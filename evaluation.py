"""Evaluate primary annotated objects using the exact same pixel inference as uploads."""
from collections import defaultdict

from detector import box_from_row, manifest, overlaps
from inspection import analyze
from prepare import NORMAL, read_regions


def evaluate():
    test = manifest("test")
    boxes = read_regions()
    rows = analyze([row["path"] for row in test])
    by_input = defaultdict(list)
    for row in rows:
        i = row["input_index"]
        row["preview_url"] = f"/demo/image/{i}"
        row["source"] = test[i]["source"]
        if "bbox" in row:
            by_input[i].append(row)
    checks = []
    annotated = matched = 0
    for i, ref in enumerate(test):
        annotations = boxes[ref["source"]]
        primary, = [box_from_row(b) for b in annotations if b["role"] == "primary"]
        annotated += len(annotations)
        distances = [(overlaps(row["bbox"], box_from_row(box))[0], j, k)
                     for j, row in enumerate(by_input[i]) for k, box in enumerate(annotations)]
        used_rows, used_boxes = set(), set()
        for iou, j, k in sorted(distances, reverse=True):
            if iou >= .5 and j not in used_rows and k not in used_boxes:
                used_rows.add(j)
                used_boxes.add(k)
        matched += len(used_boxes)
        best = max(by_input[i], key=lambda row: overlaps(row["bbox"], primary)[0], default={})
        iou = overlaps(best["bbox"], primary)[0] if best else 0
        found = iou >= .5
        evidence = best if found else {}
        checks.append(dict(source=ref["source"], photo=ref["source"].rsplit("/", 1)[-1],
                           input_index=i, source_id=next(r["source_id"] for r in rows if r["input_index"] == i),
                           expected_type=ref["object_type"], expected_condition=ref["condition"],
                           source_status=ref["source_status"], description=ref["description"],
                           expected_box=primary, iou=round(iou, 3), found=found,
                           object_id=evidence.get("object_id", ""), predicted_type=evidence.get("object_type", ""),
                           predicted_condition=evidence.get("condition", ""), decision=evidence.get("status", ""),
                           sent_to_review=bool(evidence.get("review"))))
    damaged = [r for r in checks if r["expected_condition"] != NORMAL]
    stats = dict(total=len(test), localized=sum(r["found"] for r in checks),
                 type_correct=sum(r["predicted_type"] == r["expected_type"] for r in checks),
                 condition_correct=sum(r["predicted_condition"] == r["expected_condition"] for r in checks),
                 damaged=len(damaged), damaged_review=sum(r["sent_to_review"] for r in damaged),
                 detections=sum(map(len, by_input.values())), annotated=annotated, matched=matched)
    return dict(rows=rows, checks=checks, stats=stats)
