"""Use local macOS frameworks and an interpreter with the two image packages."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def image_python():
    if all(importlib.util.find_spec(pkg) for pkg in ("numpy", "PIL")):
        return
    candidates = (ROOT / ".venv/bin/python3",
                  Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")
    for path in candidates:
        if path.is_file() and path.resolve() != Path(sys.executable).resolve():
            if subprocess.run([str(path), "-c", "import PIL, numpy"], capture_output=True).returncode == 0:
                os.execv(str(path), [str(path), *sys.argv])
    raise SystemExit("Установите зависимости в окружении проекта: python3 -m pip install -r requirements.txt")


def build_helpers():
    build = ROOT / ".build"
    build.mkdir(exist_ok=True)
    for name, frameworks in (("vision", ["Foundation", "Vision"]),
                             ("video", ["Foundation", "AVFoundation", "CoreMedia", "CoreGraphics", "ImageIO"])):
        source = ROOT / f"{name}.m"
        output = build / name
        if output.exists() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            continue
        flags = [arg for framework in frameworks for arg in ("-framework", framework)]
        with tempfile.TemporaryDirectory(dir=build) as work:
            binary = Path(work) / name
            subprocess.run(["clang", "-fobjc-arc", "-O2", "-Wno-deprecated-declarations",
                            *flags, str(source), "-o", str(binary)], check=True)
            binary.replace(output)
