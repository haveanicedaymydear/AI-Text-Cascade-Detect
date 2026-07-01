"""Reviewer-facing Flask entry point for the DTD-LRC demo artifact.

The online demo intentionally runs the lightweight DTD stage interactively.
Full MS-LRC scoring requires transformer models, so the web demo exposes
precomputed MS-LRC evidence for fixed examples instead of loading the full
pipeline on startup.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
EXAMPLES_DIR = ROOT_DIR / "examples"
MODEL_CANDIDATES = [
    ROOT_DIR / "models" / "optimized_dtd_model.pkl",
    ROOT_DIR / "optimized_dtd_model.pkl",
]

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))


app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))
CORS(app)


class DemoDetector:
    """Small wrapper that prefers the trained DTD model and falls back gracefully."""

    def __init__(self) -> None:
        self.engine: Any = None
        self.engine_name = "uninitialized"
        self.model_path: Optional[Path] = None
        self.error: Optional[str] = None
        self.load()

    def load(self) -> None:
        try:
            from optimized_dtd_v2 import UltraFastDTD

            model_path = self._resolve_model_path()
            if model_path is None:
                raise FileNotFoundError("No optimized DTD model file was found.")

            detector = UltraFastDTD()
            detector.load_model(str(model_path))
            self.engine = detector
            self.engine_name = "trained_dtd"
            self.model_path = model_path
            self.error = None
            return
        except Exception as exc:
            self.error = str(exc)

        try:
            from dtd_lrc_final import DTDLRCCascade

            self.engine = DTDLRCCascade()
            self.engine_name = "heuristic_fallback"
        except Exception as exc:
            self.error = f"{self.error}; fallback failed: {exc}" if self.error else str(exc)
            self.engine = None
            self.engine_name = "unavailable"

    def _resolve_model_path(self) -> Optional[Path]:
        env_path = os.environ.get("DTD_MODEL_PATH")
        candidates = [Path(env_path)] if env_path else []
        candidates.extend(MODEL_CANDIDATES)
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return None

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def predict(self, text: str) -> Dict[str, Any]:
        if self.engine is None:
            raise RuntimeError(self.error or "Detector is not initialized.")

        start = time.time()
        if self.engine_name == "trained_dtd":
            raw = self.engine.predict_fast(text)
            ai_probability = float(raw["ai_probability"])
            prediction = _normalize_prediction(raw["prediction"], ai_probability)
            confidence = float(raw["confidence"])
            model_time = float(raw.get("prediction_time", 0.0))
        else:
            raw = self.engine.detect(text)
            ai_probability = float(raw.ai_probability)
            prediction = _normalize_prediction(raw.prediction, ai_probability)
            confidence = float(raw.confidence)
            model_time = float(raw.processing_time)

        elapsed_ms = (time.time() - start) * 1000.0
        stage_note = "DTD"
        uncertainty_band = 0.41 <= ai_probability <= 0.61
        if uncertainty_band:
            stage_note = "DTD (uncertain; see precomputed MS-LRC examples)"

        return {
            "prediction": prediction,
            "ai_probability": ai_probability,
            "human_probability": 1.0 - ai_probability,
            "confidence": confidence,
            "prediction_time_ms": elapsed_ms or model_time,
            "stage_used": stage_note,
            "interactive_stage": "DTD",
            "ms_lrc_mode": "precomputed_examples",
            "uncertainty_band": [0.41, 0.61],
            "is_uncertain": uncertainty_band,
            "text_length": len(text),
            "word_count": len(text.split()),
        }


def _normalize_prediction(label: str, ai_probability: float) -> str:
    label_lower = label.lower()
    if "ai" in label_lower:
        return "AI Generated"
    if "human" in label_lower:
        return "Human Written"
    return "AI Generated" if ai_probability >= 0.5 else "Human Written"


def _load_text_example(name: str) -> Optional[str]:
    path = EXAMPLES_DIR / f"{name}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _load_mslrc_examples() -> List[Dict[str, Any]]:
    path = EXAMPLES_DIR / "mslrc_precomputed.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("examples", [])


DETECTOR = DemoDetector()


@app.route("/")
def index() -> str:
    examples = {
        "human_sample": _load_text_example("human_sample"),
        "ai_sample": _load_text_example("ai_sample"),
        "uncertain_sample": _load_text_example("uncertain_sample"),
    }
    return render_template(
        "artifact_demo.html",
        examples={key: value for key, value in examples.items() if value},
    )


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "success": True,
            "status": "healthy" if DETECTOR.loaded else "degraded",
            "detector_loaded": DETECTOR.loaded,
            "detector_engine": DETECTOR.engine_name,
            "model_path": str(DETECTOR.model_path) if DETECTOR.model_path else None,
            "error": DETECTOR.error,
            "interactive_stage": "DTD",
            "ms_lrc_mode": "precomputed_examples",
            "full_pipeline_command": "python src/cascade_demo_app.py",
        }
    )


@app.route("/api/detect", methods=["POST"])
def detect() -> Any:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if len(text) < 10:
        return jsonify({"success": False, "error": "Please provide at least 10 characters."}), 400
    if len(text) > 10000:
        return jsonify({"success": False, "error": "Please provide no more than 10000 characters."}), 400

    try:
        result = DETECTOR.predict(text)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify(
        {
            "success": True,
            "result": result,
            "artifact_notice": (
                "Detector scores are probabilistic evidence. Do not use them as the sole "
                "basis for punitive academic, employment, or moderation decisions."
            ),
            "timestamp": time.time(),
        }
    )


@app.route("/api/examples", methods=["GET"])
def examples() -> Any:
    names = ["human_sample", "ai_sample", "uncertain_sample"]
    return jsonify(
        {
            "success": True,
            "examples": [
                {"id": name, "text": _load_text_example(name)}
                for name in names
                if _load_text_example(name)
            ],
        }
    )


@app.route("/api/mslrc_examples", methods=["GET"])
def mslrc_examples() -> Any:
    example_id = request.args.get("id")
    examples_data = _load_mslrc_examples()
    if example_id:
        match = next((item for item in examples_data if item.get("id") == example_id), None)
        if match is None:
            return jsonify({"success": False, "error": f"Unknown example id: {example_id}"}), 404
        return jsonify({"success": True, "example": match})
    return jsonify({"success": True, "examples": examples_data})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("DTD-LRC reviewer demo")
    print(f"Detector engine: {DETECTOR.engine_name}")
    print(f"Open: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
