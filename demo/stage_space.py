"""Assemble a minimal, self-contained Hugging Face Space / Docker build directory.

Copies only what inference needs (model code, demo, the two served
checkpoints, pca.pkl, the reported EER table) and re-verifies every
artifact hash in the output. Nothing is pushed anywhere.

    python demo/stage_space.py --out ../twostream-space
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC_PACKAGES = ["src/__init__.py", "src/fusion", "src/stream1", "src/stream2"]
DEMO_FILES = ["demo/__init__.py", "demo/app.py", "demo/inference.py", "demo/decode.py",
              "demo/thresholds.json", "demo/requirements.txt", "demo/requirements.lock", "demo/.streamlit/config.toml"]
EXTRA_FILES = ["results/tables/table_eer.md"]
SPACE_TEMPLATE = ROOT / "demo" / "space"  # Dockerfile, README.md, .gitattributes, .dockerignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out).resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("--out must be outside the research repo")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new directory")

    from demo import inference as inf
    assets = [a for m in inf.MODELS.values() for a in (m["checkpoint"], m["pca"]) if a]
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for rel in SRC_PACKAGES + DEMO_FILES + EXTRA_FILES + [a[0] for a in assets]:
        src, dst = ROOT / rel, out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, ignore=ignore) if src.is_dir() else shutil.copy2(src, dst)
    for f in SPACE_TEMPLATE.iterdir():
        shutil.copy2(f, out / f.name)

    inf.ROOT = out
    for rel, sha in dict(assets).items():
        inf.verify_asset(rel, sha)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
    print(f"[stage] {out} ready ({size:.0f} MB), all artifact hashes verified")


if __name__ == "__main__":
    main()
