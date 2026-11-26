import argparse
from base64 import b64decode
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from threading import Semaphore
from urllib.parse import urlsplit

from runtime import ROOT, build_helpers, image_python

if __name__ == "__main__":
    image_python()

from detector import MODEL_PATH, manifest, read_photo, train
from evaluation import evaluate
from inspection import analyze, analyze_video, thumbnail
from prepare import DATA, prepare, write_csv

VIDEO_EXTENSIONS = {".mp4", ".mov"}
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
COMPUTE = Semaphore(1)
REPORT = {
    "object_id": "ID наблюдения", "source_id": "ID файла", "photo": "Файл",
    "source": "Путь в датасете", "object_type": "Тип объекта", "condition": "Прогноз состояния",
    "uncertain": "Прогноз неуверенный", "status": "Решение", "recommendation": "Что проверить",
    "review": "Требует осмотра", "frame_quality": "Качество кадра", "quality_reason": "Причина отбраковки",
    "start": "Начало, с", "end": "Конец, с",
    "timestamp": "Показанный кадр, с", "frames": "Кадров с объектом",
    "bbox": "Рамка x1 y1 x2 y2 (доли кадра)", "error": "Ошибка",
}


def as_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=";")
    writer.writerow(REPORT.values())
    for row in rows:
        values = []
        for key in REPORT:
            value = row.get(key, "")
            if isinstance(value, bool):
                value = "Да" if value else "Нет"
            elif isinstance(value, list):
                value = " ".join(map(str, value))
            elif isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            values.append(value)
        writer.writerow(values)
    return stream.getvalue().encode("utf-8-sig")


def process_upload(files):
    if not isinstance(files, list) or not 1 <= len(files) <= 30:
        raise ValueError("Выберите до 30 фото или один видеоролик")
    if any(not isinstance(item, dict) or not isinstance(item.get("name"), str) or
           not isinstance(item.get("data"), str) for item in files):
        raise ValueError("В запросе отсутствует имя или содержимое файла")
    names = [Path(f["name"].replace("\\", "/")).name for f in files]
    suffixes = [Path(name).suffix.lower() for name in names]
    if set(suffixes) - PHOTO_EXTENSIONS - VIDEO_EXTENSIONS:
        raise ValueError("Поддерживаются JPG, JPEG, PNG, WebP, MP4 и MOV")
    video = bool(set(suffixes) & VIDEO_EXTENSIONS)
    if video and len(files) != 1:
        raise ValueError("Выберите один видеоролик")
    limit = 60 if video else 18
    with TemporaryDirectory() as directory:
        paths = []
        size = 0
        for i, (item, suffix) in enumerate(zip(files, suffixes)):
            data = b64decode(item["data"], validate=True)
            size += len(data)
            if size > limit * 1024**2:
                raise ValueError(f"Лимит одной загрузки — {limit} МБ")
            path = Path(directory) / f"upload-{i}{suffix}"
            path.write_bytes(data)
            paths.append(path)
        with COMPUTE:
            return analyze_video(paths[0], names[0]) if video else analyze(paths, names)


def demo_image(route):
    test = manifest("test")
    if route == "/guide":
        record = test[0]
        source = Path(record["source"])
        datasets = Path(record["path"]).parents[len(source.parts)-1]
        path = datasets / "Датасет-ДСС" / "Фото.1.jpg"
        image = read_photo(path)
        return thumbnail(image, (1000, 1000))
    index = int(route.rsplit("/", 1)[-1])
    if index >= len(test):
        raise FileNotFoundError
    image = read_photo(test[index]["path"])
    return thumbnail(image, (850, 750))


class Handler(BaseHTTPRequestHandler):
    def reply(self, body, mime, status=200, attachment=False):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="mgts-report.csv"')
        self.end_headers()
        self.wfile.write(body)

    def message(self, value, status):
        self.reply(value.encode(), "text/plain; charset=utf-8", status)

    def do_GET(self):
        path = urlsplit(self.path).path
        pages = {"/": "index.html", "/about": "about.html", "/defense": "defense.html"}
        if path in pages:
            filename = pages[path]
            self.reply((ROOT / filename).read_bytes(), "text/html; charset=utf-8")
        elif re.fullmatch(r"/ui/[a-zA-Z0-9_-]+\.(?:css|js|json|webp|png|woff2)", path):
            asset = ROOT / path.lstrip("/")
            if asset.is_file():
                mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                self.reply(asset.read_bytes(), mime)
            else:
                self.message("Ресурс не найден", 404)
        elif path == "/guide" or re.fullmatch(r"/demo/image/[0-9]{1,4}", path):
            try:
                self.reply(demo_image(path), "image/jpeg")
            except FileNotFoundError:
                self.message("Исходный пример недоступен. Проверьте пути к датасетам.", 404)
        elif path == "/demo/video.mp4":
            clip = DATA / "demo-from-test-photos.mp4"
            if clip.exists():
                self.reply(clip.read_bytes(), "video/mp4")
            else:
                self.message("Демонстрационный ролик недоступен", 404)
        else:
            self.message("Страница не найдена", 404)

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            if path == "/demo":
                with COMPUTE:
                    result = evaluate()
            elif path in ("/analyze", "/report.csv"):
                size = int(self.headers.get("Content-Length", 0))
                limit = 81 if path == "/analyze" else 2
                if not 0 < size <= limit * 1024**2:
                    self.message("Проверьте размер загружаемых файлов", 413)
                    return
                payload = json.loads(self.rfile.read(size))
                if path == "/report.csv":
                    if not isinstance(payload, list) or any(not isinstance(r, dict) for r in payload):
                        raise ValueError("Некорректные строки отчёта")
                    self.reply(as_csv(payload), "text/csv; charset=utf-8", attachment=True)
                    return
                result = process_upload(payload)
            else:
                self.message("Адрес не найден", 404)
                return
            self.reply(json.dumps(result, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")
        except (ValueError, KeyError, TypeError) as error:
            self.message(str(error), 400)
        except FileNotFoundError:
            self.message("Файлы модели или тестовые данные недоступны. Проверьте инструкцию запуска.", 503)
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            self.log_error("%s", error)
            self.message("Не удалось завершить обработку. Подробности в терминале приложения.", 500)


def serve(port, lan):
    with ThreadingHTTPServer(("0.0.0.0" if lan else "127.0.0.1", port), Handler) as server:
        print(f"Сайт на Mac: http://127.0.0.1:{server.server_port}", flush=True)
        if lan:
            config = subprocess.run(["/sbin/ifconfig"], capture_output=True, text=True, check=False)
            addresses = re.findall(r"^\s+inet (\d+\.\d+\.\d+\.\d+) ", config.stdout, re.MULTILINE)
            for address in addresses:
                if not address.startswith("127."):
                    print(f"Для устройства в той же сети: http://{address}:{port}", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="собрать разметку и обучить модель")
    mode.add_argument("--test", action="store_true", help="прогнать отложенную выборку")
    parser.add_argument("--datasets", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--lan", action="store_true", help="разрешить доступ из локальной сети")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    build_helpers()
    if args.train:
        prepare(args.datasets.expanduser().resolve())
        train()
        return
    if not MODEL_PATH.exists():
        parser.exit(1, "Сначала выполните python3 app.py --train\n")
    if args.test:
        result = evaluate()
        write_csv(DATA / "test_report.csv", result["checks"])
        print(json.dumps(result["stats"], ensure_ascii=False, indent=2))
        return
    serve(args.port, args.lan)


if __name__ == "__main__":
    main()
