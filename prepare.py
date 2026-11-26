"""Join the six source workbooks to images and the reviewed object boxes."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

DATA = Path(__file__).resolve().parent / "data"
DATASETS = ("Датасет-ДСС", "Датасет-ДСС 2")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
NORMAL = "Норма"
COATING = "Повреждение ЛКП"
MECHANICAL = "Механическое повреждение"


def sheet_rows(path):
    with ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(si.itertext()) for si in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    for row in sheet.findall(".//{*}row"):
        cells = {}
        for cell in row:
            value = cell.findtext("{*}v") or "".join(cell.itertext())
            if cell.get("t") == "s":
                value = strings[int(value)]
            cells[cell.get("r", "").rstrip("0123456789")] = value.strip()
        if cells.get("A"):
            yield tuple(cells.get(col, "") for col in "ABC")


def read_regions():
    with (DATA / "regions.csv").open(encoding="utf-8", newline="") as stream:
        groups = {}
        for row in csv.DictReader(stream):
            groups.setdefault(row["source"], []).append(row)
    return groups


def condition_for(status):
    status = status.casefold()
    if status.startswith("удовлет"):
        return NORMAL
    if status.startswith("повреждение лкп"):
        return COATING
    if status.startswith(("деформация", "нарушение")):
        return MECHANICAL
    raise ValueError(f"Неизвестный исходный статус {status!r}")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(downloads):
    regions = read_regions()
    rows = []
    seen = {}
    splits = {}
    for name in DATASETS:
        root = downloads / name
        if not root.is_dir():
            raise ValueError(f"Папка датасета не найдена: {root}")
        for table in sorted(root.rglob("*.xlsx")):
            entries = iter(sheet_rows(table))
            if next(entries) != ("Имя файла", "Статус", "Описание"):
                raise ValueError(f"Неожиданные колонки в {table}")
            labels = {}
            for key, status, description in entries:
                if key in labels:
                    raise ValueError(f"Повторный идентификатор {key} в {table.name}")
                labels[key] = status, description
            for path in sorted(p for p in table.parent.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
                source = path.relative_to(downloads).as_posix()
                status, description = labels.pop(path.stem)
                boxes = regions[source]
                primary, = [box for box in boxes if box["role"] == "primary"]
                folder_type = "ТКШ" if "ТКШ" in table.parent.name else "РЩ"
                if primary["object_type"] != folder_type:
                    raise ValueError(f"Тип основной рамки расходится с папкой: {source}")
                tag = primary["split"]
                group = primary["site_group"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if seen.setdefault(digest, tag) != tag or splits.setdefault(group, tag) != tag:
                    raise ValueError(f"Пересечение обучения и теста: {source}")
                rows.append(dict(source=source, path=os.path.relpath(path, DATA), site_group=group,
                                 object_type=folder_type, condition=condition_for(status),
                                 source_status=status, description=description, split=tag))
            if labels:
                raise ValueError(f"В {table.parent.name} отсутствуют фото для {sorted(labels)}")
    if set(regions) != {row["source"] for row in rows}:
        raise ValueError("Состав папок расходится с проверенной разметкой")
    for name, subset in (("manifest", rows), ("train", [r for r in rows if r["split"] == "train"]),
                         ("test", [r for r in rows if r["split"] == "test"])):
        write_csv(DATA / f"{name}.csv", subset)
        print(f"{name}.csv — {len(subset)} снимков")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=Path.home() / "Downloads",
                        help="папка, содержащая оба исходных датасета")
    prepare(parser.parse_args().datasets.expanduser().resolve())
