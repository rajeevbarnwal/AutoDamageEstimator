"""infer.py – unified inference & cost‑estimation script
------------------------------------------------------
• Predicts damages/parts with YOLOv8
• Computes repair/replace cost via SQLite lookup
• Falls back to a local LLM (DeepSeek / Qwen via Ollama) when the part/damage is
  not present in the table so nothing is silently ignored.
• Accepts **one path argument** that can be
    – a single image,
    – a directory,            ➜ processes all *.jpg|png in it
    – a glob pattern (quoted) ➜ e.g. "data/**/*.jpg"
  If no argument is given, it defaults to all test images under
  data/processed/test/images/.

Usage examples
--------------
# one image
python scripts/infer.py data/img.jpg

# every image in a folder
python scripts/infer.py data/processed/test/images/

# glob pattern (MUST be quoted for zsh/bash so the shell doesn’t expand it)
python scripts/infer.py "data/processed/test/images/damage_*.jpg"
"""
from __future__ import annotations

import glob
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from ultralytics import YOLO

# === Optional LLM fallback (DeepSeek, Qwen, etc. exposed via Ollama) =============
try:
    from langchain_community.llms import Ollama  # tiny wrapper around local models
    from langchain.prompts import PromptTemplate
except ImportError:  # don’t break if user hasn’t installed langchain or ollama
    Ollama = None  # type: ignore

# --------------------------------------------------------------------------------
MODEL_PATH = "runs/detect/train3/weights/best.pt"
DB_PATH = "database/parts_costs.db"

# Cache the YOLO model & LLM so we load them only once --------------------------------
_yolo_model: YOLO | None = None
_llm = None


def _get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model


def _get_llm():
    """Return a cached local LLM instance (DeepSeek/Qwen via Ollama) or None."""
    global _llm
    if _llm is None and Ollama is not None:
        try:
            _llm = Ollama(model="deepseek-r1:latest",      # keep tag I prefer
                         base_url="http://localhost:11434") # ← My Ollama host
        except Exception as e:
            print(f"⚠️  Could not load local LLM ({e}). Unknown parts will default to ₹0.")
            _llm = False  # sentinel – don’t retry every time
    return _llm if _llm is not False else None

# ------------------------------------------------------------------------------------
# 1) Detection & severity stub
# ------------------------------------------------------------------------------------

def detect(image_path: str, conf: float = 0.1) -> List[Dict]:
    """Run YOLO prediction and return a list of detections.

    Each dict = {"class": str, "confidence": float, "severity": str}
    """
    model = _get_model()
    results = model.predict(image_path, conf=conf)

    detections: List[Dict] = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "severity": "moderate",  # TODO: plug‑in a dedicated severity model
            })
    return detections

# ------------------------------------------------------------------------------------
# 2) Cost estimation
# ------------------------------------------------------------------------------------

_PROMPT = PromptTemplate(
    template=(
        "You are an expert automobile repair estimator. "
        "Give me a *single number* in INR – no text – for repairing a {part} "
        "with {severity} damage."
    ),
    input_variables=["part", "severity"],
)


def estimate_cost(detections: List[Dict], db_path: str = DB_PATH) -> float:
    """Sum repair/replace costs for all detections.

    • Looks up known parts in SQLite table `parts`.
    • For unknown parts/damages, queries a local LLM (if available).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    total = 0.0
    for det in detections:
        part = det["class"]
        severity = det.get("severity", "moderate")

        cur.execute("SELECT repair_cost, replace_cost FROM parts WHERE part_name=?", (part,))
        row = cur.fetchone()
        if row:
            # trivial severity rule – tweak as you collect data
            repair_cost, replace_cost = row
            cost = replace_cost if severity == "severe" else repair_cost
        else:
            llm = _get_llm()
            if llm is None:
                cost = 0.0  # gracefully degrade
            else:
                try:
                    reply = llm(_PROMPT.format(part=part, severity=severity)).strip()
                    cost = float(reply.split()[0].replace(",", ""))
                except Exception as e:
                    print(f"⚠️  LLM could not estimate cost for '{part}' – {e}")
                    cost = 0.0
        total += cost

    conn.close()
    return total

# ------------------------------------------------------------------------------------
# 3) End‑to‑end helper: infer(image_path) – kept for backward compatibility
# ------------------------------------------------------------------------------------

def infer(
    image_path: str,
    *,
    conf: float = 0.25,          # <-- new
) -> Tuple[List[Dict], float]:
    """Detect + estimate cost on a single image path.

    Parameters
    ----------
    image_path : str
        Path to the image you want to analyse.
    conf : float, optional
        YOLO confidence threshold (0–1). Forwarded to `detect()`.
    """
    detections = detect(image_path, conf=conf)  # <-- forward it
    cost = estimate_cost(detections)
    return detections, cost

# ------------------------------------------------------------------------------------
# 4) CLI entry‑point – now batch‑aware
# ------------------------------------------------------------------------------------

def _gather_images(arg: str | None) -> List[str]:
    """Expand the CLI argument into a list of image paths."""
    if arg is None:
        # default – all test images
        return sorted(glob.glob("data/processed/test/images/*.[jp][pn]g"))

    p = Path(arg)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(glob.glob(str(p / "*.[jp][pn]g")))

    # treat as glob pattern – shell‑safe quoting required
    matched = glob.glob(arg, recursive=True)
    if matched:
        return sorted(matched)

    raise FileNotFoundError(f"No images matched: {arg}")


if __name__ == "__main__":
    image_list = _gather_images(sys.argv[1] if len(sys.argv) > 1 else None)
    if not image_list:
        print("⚠️  No images found – nothing to do.")
        sys.exit(1)

    for img in image_list:
        det, price = infer(img)
        rel = os.path.relpath(img)
        print(f"Image   : {rel}")
        print(f"Objects : {det}")
        print(f"Estimate: ₹{price:,.0f}")
        print("-" * 60)
