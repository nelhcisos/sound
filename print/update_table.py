#!/usr/bin/env python3
"""Пересобирает print/README.md: по строке на каждую папку модели.

Папка модели: print/<имя>/ с preview.png (или .jpg/.webp) и одним или
несколькими .stl / .3mf (если персонаж из частей — все части в этой папке).
"""
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
PREVIEW_EXT = (".png", ".jpg", ".jpeg", ".webp")
MODEL_EXT = (".stl", ".3mf", ".obj")


def link(p: Path) -> str:
    return quote(p.relative_to(ROOT).as_posix())


rows = []
for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")):
    files = sorted(d.iterdir())
    previews = [f for f in files if f.suffix.lower() in PREVIEW_EXT]
    models = [f for f in files if f.suffix.lower() in MODEL_EXT]
    prev = f'<img src="{link(previews[0])}" width="160">' if previews else "—"
    if len(models) == 1:
        stl = f"[{models[0].name}]({link(models[0])})"
    elif models:
        stl = f"[папка, {len(models)} частей]({link(d)}/)<br>" + "<br>".join(
            f"[{m.name}]({link(m)})" for m in models)
    else:
        stl = "—"
    rows.append(f"| {prev} | {d.name} | {stl} |")

body = "\n".join(rows) if rows else "| — | пока пусто | — |"
(ROOT / "README.md").write_text(
    "# Модели для печати\n\n"
    "| Превью | Модель | STL |\n|---|---|---|\n" + body + "\n\n"
    "Как добавить: папка `print/<имя>/` → `preview.png` + `.stl` "
    "(части одного персонажа — в той же папке), затем `python3 print/update_table.py`.\n",
    encoding="utf-8")
print(f"{len(rows)} моделей в таблице")
