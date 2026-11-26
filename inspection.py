"""Inference, inspection recommendations and association of adjacent video observations."""
import base64
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from detector import Detector, overlaps, read_photo
from prepare import COATING, MECHANICAL, NORMAL, DATA

ACTIONS = {NORMAL: "Плановое наблюдение", COATING: "Проверить покрытие и коррозию",
           MECHANICAL: "Проверить целостность корпуса и креплений"}
STATUS_RANK = {"В норме": 0, "Нужна проверка": 1, "Кандидат на ремонт": 2}


@lru_cache(maxsize=1)
def model():
    return Detector()


def predict(image):
    """Only image pixels enter detection. Each accepted region gets its own result."""
    objects = model().detect(image)
    for obj in objects:
        state = obj["condition"]
        if obj["uncertain"]:
            obj["status"] = "Нужна проверка"
            obj["recommendation"] = ("Осмотреть объект вручную" if state == NORMAL else ACTIONS[state])
        elif state == NORMAL:
            obj["status"] = "В норме"
            obj["recommendation"] = ACTIONS[state]
        else:
            obj["status"] = "Кандидат на ремонт" if state == MECHANICAL else "Нужна проверка"
            obj["recommendation"] = ACTIONS[state]
        obj["review"] = obj["status"] != "В норме"
    return objects


def source_id(path):
    with open(path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest[:12]


def quality_issue(image):
    """Reject only extreme signal loss; ordinary object-free scenes go to the detector."""
    sample = image.copy()
    sample.thumbnail((256, 256), Image.Resampling.BOX)
    rgb = np.asarray(sample)
    if np.mean(np.max(rgb, axis=2) <= 12) >= .99:
        return "Почти полностью чёрный кадр"
    if np.mean(np.min(rgb, axis=2) >= 243) >= .99:
        return "Почти полностью белый кадр"
    low, high = np.percentile(rgb, [1, 99], axis=(0, 1))
    if np.max(high - low) <= 4:
        return "Почти однотонное изображение без различимых деталей"
    # Compare adjacent pixels at their native scale to distinguish strong static
    # from normal fine texture. A coarse average must also contain little structure.
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if min(gray.shape) >= 32 and gray.std() > 35:
        dx = np.square(np.diff(gray, axis=1)).mean()
        dy = np.square(np.diff(gray, axis=0)).mean()
        coarse = np.asarray(image.convert("L").resize((8, 8), Image.Resampling.BOX))
        if min(dx, dy) > 1.5 * gray.var() and coarse.std() < 8:
            return "Сильный цифровой шум по всему кадру"
    return None


def rejected(reason, *, decoded):
    return dict(status="Непригодный кадр" if decoded else "Битый кадр", review=True,
                frame_quality="Непригодное изображение" if decoded else "Ошибка декодирования",
                quality_reason=reason, recommendation="Проверить запись или повторить съёмку")


def checked_image(path):
    try:
        image = read_photo(path)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None, rejected("Декодер не смог прочитать изображение", decoded=False)
    reason = quality_issue(image)
    return image, rejected(reason, decoded=True) if reason else None


def no_object():
    return dict(status="Объект не обнаружен", review=False,
                recommendation="При наличии оборудования загрузить более крупный план")


def analyze(paths, names=None):
    rows = []
    for i, path in enumerate(paths):
        identifier = source_id(path)
        base = dict(photo=names[i] if names else Path(path).name, source_id=identifier, input_index=i)
        image, issue = checked_image(path)
        if issue:
            rows.append({**base, **issue})
            continue
        objects = predict(image)
        if not objects:
            rows.append({**base, **no_object()})
        for number, obj in enumerate(objects, 1):
            obj.pop("appearance")
            rows.append({**base, "object_id": f"{identifier}-{number:02}", **obj})
    return rows


def thumbnail(image, maximum=(640, 500)):
    thumb = image.copy()
    thumb.thumbnail(maximum, Image.Resampling.LANCZOS)
    stream = BytesIO()
    thumb.save(stream, format="JPEG", quality=78)
    return stream.getvalue()


def match_tracks(active, objects):
    candidates = []
    for a, track in enumerate(active):
        for b, obj in enumerate(objects):
            overlap, _ = overlaps(track["last_bbox"], obj["bbox"])
            sim = float(track["appearance"] @ obj["appearance"])
            same_type = obj["object_type"] == track["row"]["object_type"]
            if (same_type or sim > .90) and sim > .63 and overlap >= .22:
                candidates.append((overlap + .4 * sim, a, b))
    used = set()
    matches = {}
    for _, a, b in sorted(candidates, reverse=True):
        if a not in used and b not in matches:
            matches[b] = active[a]
            used.add(a)
    return matches


def video_rows(frames, path, name):
    identifier = source_id(path)
    tracks = []
    active = []
    unknown_frames = 0
    issues = []
    first_clear = None
    base = dict(photo=name, source_id=identifier, input_index=0)
    for frame in frames:
        stamp = round(frame["timestamp"], 3)
        if frame.get("decode_failed"):
            image, issue = None, rejected("Не удалось декодировать кадр видео", decoded=False)
        else:
            image, issue = checked_image(frame["path"])
        if issue:
            row = {**base, **issue, "timestamp": stamp}
            if image is not None:
                row["preview"] = base64.b64encode(thumbnail(image)).decode("ascii")
            issues.append(row)
            active = []
            continue
        if first_clear is None:
            first_clear = (stamp, image)
        objects = predict(image)
        matches = match_tracks(active, objects)
        active = []
        unknown_frames += int(not objects)
        preview = None
        for i, obj in enumerate(objects):
            appearance = obj.pop("appearance")
            track = matches.get(i)
            if track is None:
                track = {"row": {"object_id": f"{identifier}-{len(tracks)+1:02}", "start": stamp,
                                 "frames": 0, **base}, "rank": (-1, -1, -1)}
                tracks.append(track)
            row = track["row"]
            row["end"] = stamp
            row["frames"] += 1
            # Keep the most actionable evidence, preferring a larger, clear crop on ties.
            box = obj["bbox"]
            rank = (STATUS_RANK[obj["status"]], not obj["uncertain"], (box[2]-box[0]) * (box[3]-box[1]))
            if rank > track["rank"]:
                if preview is None:
                    preview = base64.b64encode(thumbnail(image)).decode("ascii")
                row.update(obj, timestamp=stamp, preview=preview)
                track["rank"] = rank
            track.update(last_bbox=box, appearance=appearance)
            active.append(track)
    rows = [t["row"] for t in tracks]
    if not rows and first_clear is not None:
        stamp, image = first_clear
        rows.append({**base, "timestamp": stamp,
                     "preview": base64.b64encode(thumbnail(image)).decode("ascii"), **no_object()})
    rows.extend(issues)
    rows.sort(key=lambda r: r.get("start", r["timestamp"]))
    for row in rows:
        row.update(sampled_frames=len(frames), rejected_frames=len(issues), frames_without_objects=unknown_frames)
    return rows


def analyze_video(path, name):
    try:
        proc = subprocess.run([str(DATA.parent / ".build/video"), str(path), str(path.parent)],
                              capture_output=True, text=True, check=True, timeout=120)
    except subprocess.CalledProcessError as error:
        if error.returncode != 3:
            raise
        issue = rejected("Видеодорожка отсутствует или файл не читается", decoded=False)
        return [{**issue, "photo": name, "source_id": source_id(path), "input_index": 0,
                 "status": "Видео не читается", "sampled_frames": 0, "rejected_frames": 0}]
    return video_rows(json.loads(proc.stdout)["frames"], path, name)
