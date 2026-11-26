"""Multi-scale cabinet proposals and crop classification on Apple Vision features."""
import base64
import csv
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageOps

from prepare import COATING, DATA, MECHANICAL, NORMAL, read_regions

ROOT = DATA.parent
MODEL_PATH = DATA / "detector.npz"
TYPES = ("РЩ", "ТКШ")
CONDITIONS = (NORMAL, COATING, MECHANICAL)
CELL = 8
WINDOW = 6


def manifest(split):
    with (DATA / f"{split}.csv").open(encoding="utf-8-sig", newline="") as source:
        return [{**row, "path": str((DATA / row["path"]).resolve())} for row in csv.DictReader(source)]


def read_photo(path):
    with Image.open(path) as photo:
        image = ImageOps.exif_transpose(photo).convert("RGB")
        image.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    image.info.clear()
    return image


def box_from_row(row):
    return tuple(float(row[field]) / 100 for field in ("left_pct", "top_pct", "right_pct", "bottom_pct"))


def overlaps(a, b):
    """IoU for matching, and intersection over the smaller box for nested crops."""
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9), inter / max(min(aa, ab), 1e-9)


def crop(image, box, pad=0):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    w, h = image.size
    return image.crop((int(max(0, x0 - dx) * w), int(max(0, y0 - dy) * h),
                       int(np.ceil(min(1, x1 + dx) * w)), int(np.ceil(min(1, y1 + dy) * h))))


def cells(image):
    rgb = np.asarray(image, dtype=np.float32) / 255
    h, w = rgb.shape[:2]
    gray = rgb @ np.array([.30, .59, .11], dtype=np.float32)
    dy, dx = np.gradient(gray)
    weight = np.hypot(dx, dy)
    angle = np.mod(np.arctan2(dy, dx), np.pi) * (9 / np.pi)
    bins = angle.astype(np.int32) % 9
    frac = angle - np.floor(angle)
    yy, xx = np.ogrid[:h, :w]
    index = ((yy // CELL) * (w // CELL) + xx // CELL) * 9
    count = h // CELL * (w // CELL) * 9
    hist = np.bincount((index + bins).ravel(), (weight * (1 - frac)).ravel(), minlength=count)
    hist += np.bincount((index + (bins + 1) % 9).ravel(), (weight * frac).ravel(), minlength=count)
    hist = hist.reshape(h // CELL, w // CELL, 9).astype(np.float32)
    strength = hist.sum(2, keepdims=True)
    hist /= strength + 0.2
    color = rgb.reshape(h // CELL, CELL, w // CELL, CELL, 3).mean(axis=(1, 3))
    return np.concatenate((hist, np.minimum(strength / 10, 1), color), axis=2)


def patch_descriptor(image):
    return cells(image.resize((CELL * WINDOW, CELL * WINDOW), Image.Resampling.BILINEAR)).ravel()


def random_background(image, boxes, rng, count):
    candidates = []
    aspect = image.height / image.width
    for _ in range(count * 80):
        h = float(np.exp(rng.uniform(np.log(.07), np.log(.8))))
        w = min(.9, h * aspect * rng.uniform(.55, 1.5))
        x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
        box = (x, y, x + w, y + h)
        if all(overlaps(box, truth)[1] < .18 for truth in boxes):
            candidates.append(box)
        if len(candidates) == count:
            break
    return candidates


def linear_filter(positives, negatives):
    x = np.asarray(positives + negatives, dtype=np.float32)
    labels = np.repeat([1., -1.], [len(positives), len(negatives)])
    weights = np.repeat([.5 / len(positives), .5 / len(negatives)], [len(positives), len(negatives)])
    mean = weights @ x
    scale = np.sqrt(weights @ (x - mean) ** 2 + .002)
    z = (x - mean) / scale
    gram = z.T @ (z * weights[:, None])
    gram.flat[::gram.shape[0] + 1] += .7
    coef = np.linalg.solve(gram, z.T @ (weights * labels)) / scale
    return coef.astype(np.float32), np.float32(-mean @ coef)


def proposals(image, coef, bias, limit=18):
    candidates = []
    side = min(image.size)
    for height in (.12, .17, .24, .34, .48, .68, .95, 1.3):
        for aspect in (.59, .92, 1.4):
            sy = CELL * WINDOW / (height * side)
            sx = sy / aspect
            nx = round(image.width * sx / CELL)
            ny = round(image.height * sy / CELL)
            if min(nx, ny) < WINDOW or max(nx, ny) > 160:
                continue
            size = (nx * CELL, ny * CELL)
            features = cells(image.resize(size, Image.Resampling.BILINEAR))
            windows = np.lib.stride_tricks.sliding_window_view(features, (WINDOW, WINDOW), axis=(0, 1))
            kernel = coef.reshape(WINDOW, WINDOW, -1).transpose(2, 0, 1)
            scores = np.einsum("xycij,cij->xy", windows, kernel, optimize=False) + bias
            # Keep distinct local maxima within each scale, including edge windows.
            for _ in range(min(4, scores.size)):
                y, x = np.unravel_index(np.argmax(scores), scores.shape)
                score = float(scores[y, x])
                if not np.isfinite(score):
                    break
                candidates.append((score, (x / nx, y / ny, (x + WINDOW) / nx, (y + WINDOW) / ny)))
                scores[max(0, y-2):y+3, max(0, x-2):x+3] = -np.inf
    best = []
    for score, box in sorted(candidates, reverse=True):
        if all(overlaps(box, other)[0] < .44 for _, other in best):
            best.append((score, box))
        if len(best) == limit:
            break
    full = (0., 0., 1., 1.)
    if not any(overlaps(box, full)[0] > .7 for _, box in best):
        best.append((float(patch_descriptor(image) @ coef + bias), full))
    return best


def embeddings(patches):
    with tempfile.TemporaryDirectory() as directory:
        files = []
        for i, patch in enumerate(patches):
            path = Path(directory) / f"{i}.jpg"
            patch.thumbnail((340, 340), Image.Resampling.LANCZOS)
            patch.save(path, quality=91)
            files.append(str(path))
        proc = subprocess.run([str(ROOT / ".build/vision")], input=json.dumps(files),
                              capture_output=True, text=True, check=True)
    vectors = [np.frombuffer(base64.b64decode(line), dtype=np.float32) for line in proc.stdout.splitlines()]
    if len(vectors) != len(files) or not vectors:
        raise RuntimeError("Apple Vision вернула неполный набор признаков")
    x = np.stack(vectors)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def kernel(a, b):
    return np.exp(4 * (a @ b.T - 1))


def proposal_geometry(box, score, image):
    w, h = box[2] - box[0], box[3] - box[1]
    return [np.log(max(.01, w * h)) / 3, np.log(w * image.width / (h * image.height)), score]


def local_kernel(x, geometry, ref, ref_geometry):
    diff = geometry[:, None, :] - ref_geometry[None, :, :]
    return kernel(x, ref) * np.exp(-.24 * np.sum(diff * diff, axis=2))


def box_offsets(box, target):
    w, h = box[2] - box[0], box[3] - box[1]
    tw, th = target[2] - target[0], target[3] - target[1]
    return [(target[0] + tw / 2 - box[0] - w / 2) / w,
            (target[1] + th / 2 - box[1] - h / 2) / h, np.log(tw / w), np.log(th / h)]


def refine(box, offsets):
    dx, dy, dw, dh = np.clip(offsets, -.55, .55)
    w, h = box[2] - box[0], box[3] - box[1]
    cx, cy = box[0] + w * (.5 + dx), box[1] + h * (.5 + dy)
    return np.clip([cx - w*np.exp(dw)/2, cy - h*np.exp(dh)/2,
                    cx + w*np.exp(dw)/2, cy + h*np.exp(dh)/2], 0, 1)


def fit_head(x, labels, groups, classes):
    y = np.asarray(labels)
    target = np.eye(len(classes))[np.array([classes.index(label) for label in y])]
    gram = kernel(x, x).astype(np.float64)
    counts = {key: np.sum(y == key) for key in classes}
    gram.flat[::len(x) + 1] += [.18 * counts[key] / np.mean(list(counts.values())) for key in y]
    inverse = np.linalg.inv(gram)
    coef = inverse @ target
    # Exact leave-one-site-out ridge predictions. Keep crops and flips from
    # the same site together when choosing review and background thresholds.
    oof = np.empty_like(target)
    groups = np.asarray(groups)
    for group in set(groups):
        ids = np.where(groups == group)[0]
        oof[ids] = target[ids] - np.linalg.solve(inverse[np.ix_(ids, ids)], coef[ids])
    return coef.astype(np.float32), oof


def train():
    rng = np.random.default_rng(29)
    regions = read_regions()
    examples = []
    positives, negatives = [], []
    for row in manifest("train"):
        image = read_photo(row["path"])
        annotations = regions[row["source"]]
        boxes = [box_from_row(r) for r in annotations]
        bg = random_background(image, boxes, rng, 38)
        for box in boxes:
            for pad in (0, .07, -.06):
                patch = crop(image, box, pad)
                positives.extend((patch_descriptor(patch), patch_descriptor(ImageOps.mirror(patch))))
        negatives.extend(patch_descriptor(crop(image, b)) for b in bg)
        examples.append((row, image, annotations, boxes, bg))
    first, bias = linear_filter(positives, negatives)
    patches, types, conditions, groups = [], [], [], []
    for row, image, annotations, boxes, bg in examples:
        hard = [b for _, b in proposals(image, first, bias)
                if all(overlaps(b, gt)[1] < .22 for gt in boxes)][:8]
        negatives.extend(patch_descriptor(crop(image, b)) for b in hard)
        for ann, box in zip(annotations, boxes):
            for mirrored in (False, True):
                patch = crop(image, box, .09)
                patches.append(ImageOps.mirror(patch) if mirrored else patch)
                types.append(ann["object_type"])
                conditions.append(row["condition"] if ann["role"] == "primary" else "")
                groups.append(row["site_group"])
        for box in hard[:4] + bg[:4]:
            patches.append(crop(image, box, .09))
            types.append("Фон")
            conditions.append("")
            groups.append(row["site_group"])
    detector, bias = linear_filter(positives, negatives)
    print(f"Apple Vision обрабатывает {len(patches)} обучающих фрагментов…", flush=True)
    x = embeddings(patches)
    types, conditions, groups = map(np.asarray, (types, conditions, groups))
    fg = types != "Фон"
    labeled = conditions != ""
    presence, pred = fit_head(x, np.where(fg, "объект", "фон"), groups, ("фон", "объект"))
    margin = pred[:, 1] - pred[:, 0]
    grid = np.unique(margin)
    loss = [.55 * np.mean(margin[~fg] >= t) + .45 * np.mean(margin[fg] < t) for t in grid]
    gate = grid[np.argmin(loss)]
    type_coef, _ = fit_head(x[fg], types[fg], groups[fg], TYPES)
    state_coef, state_oof = fit_head(x[labeled], conditions[labeled], groups[labeled], CONDITIONS)
    sim = x[fg] @ x[fg].T
    sim[groups[fg, None] == groups[fg][None, :]] = -1
    similarity_floor = np.quantile(sim.max(1), .03) - .025
    normal_margin = state_oof[:, 0] - state_oof[:, 1:].max(1)
    false_normal = normal_margin[(conditions[labeled] != NORMAL) & (normal_margin > 0)]
    review_margin = max(.12, float(np.quantile(false_normal, .8)) if len(false_normal) else .12)
    # Learn which proposed boxes cover a whole cabinet, and how to correct their
    # edges. The targets come exclusively from training-site bounding boxes.
    local_patches, geometry, targets = [], [], []
    for _, image, _, boxes, _ in examples:
        for score, box in proposals(image, detector, bias):
            truth = max(boxes, key=lambda gt: overlaps(box, gt)[0])
            iou, _ = overlaps(box, truth)
            local_patches.append(crop(image, box, .09))
            geometry.append(proposal_geometry(box, score, image))
            targets.append([iou, *(box_offsets(box, truth) if iou >= .25 else [0.] * 4)])
    print(f"Обучение уточнения границ на {len(local_patches)} областях…", flush=True)
    ref = embeddings(local_patches)
    geometry = np.asarray(geometry, np.float32)
    gram = local_kernel(ref, geometry, ref, geometry).astype(np.float64)
    gram.flat[::len(ref)+1] += .28
    local_coef = np.linalg.solve(gram, np.asarray(targets)).astype(np.float32)
    # A new file replaces the previous model only after a complete fit.
    with tempfile.NamedTemporaryFile(dir=DATA, suffix=".npz", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        np.savez_compressed(tmp, detector=detector, bias=bias, x=x, fg=fg, labeled=labeled,
                            presence=presence, type_coef=type_coef, state_coef=state_coef,
                            gate=gate, similarity_floor=similarity_floor, review_margin=review_margin,
                            local_ref=ref, local_geometry=geometry, local_coef=local_coef)
        tmp.replace(MODEL_PATH)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Модель сохранена. Групп обучения: {len(set(groups))}.")


class Detector:
    def __init__(self):
        with np.load(MODEL_PATH, allow_pickle=False) as store:
            self.model = {key: store[key] for key in store.files}

    def detect(self, image):
        m = self.model
        candidates = proposals(image, m["detector"], m["bias"])
        x = embeddings(crop(image, box, .09) for _, box in candidates)
        k = kernel(x, m["x"])
        presence = k @ m["presence"]
        margins = presence[:, 1] - presence[:, 0]
        known = (x @ m["x"][m["fg"]].T).max(1) >= m["similarity_floor"]
        type_scores = k[:, m["fg"]] @ m["type_coef"]
        state_scores = k[:, m["labeled"]] @ m["state_coef"]
        geometry = np.asarray([proposal_geometry(b, s, image) for s, b in candidates], np.float32)
        localization = local_kernel(x, geometry, m["local_ref"], m["local_geometry"]) @ m["local_coef"]
        ranked = np.argsort(-(localization[:, 0] + .13 * margins))
        objects = []
        for i in ranked:
            if not known[i] or margins[i] < m["gate"] or localization[i, 0] < .26:
                continue
            box = refine(candidates[i][1], localization[i, 1:])
            if any(overlaps(box, item["bbox"])[0] > .3 or overlaps(box, item["bbox"])[1] > .68
                   for item in objects):
                continue
            kind = TYPES[np.argmax(type_scores[i])]
            state = CONDITIONS[np.argmax(state_scores[i])]
            ordered = np.sort(state_scores[i])
            state_margin = float(ordered[-1] - ordered[-2])
            type_margin = abs(float(type_scores[i, 0] - type_scores[i, 1]))
            minimum = float(m["review_margin"]) if state == NORMAL else .12
            uncertain = state_margin < minimum or type_margin < .10
            objects.append(dict(bbox=[round(float(v), 4) for v in box], object_type=kind,
                                condition=state, uncertain=uncertain, appearance=x[i]))
            if len(objects) == 5:
                break
        return sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
