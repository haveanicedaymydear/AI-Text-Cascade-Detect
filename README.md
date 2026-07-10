# DTD-LRC: Interpretable Two-Stage AI-Text Detection

This repository contains the demonstration artifact for **"DTD-LRC: An Interpretable Two-Stage Cascade for Open-World Machine-Generated Text Detection"**.

DTD-LRC combines a fast first-stage detector, **DTD**, with a deeper second-stage evidence module, **MS-LRC**, for cases where the first-stage confidence falls in an uncertainty band. The reviewer-facing web demo is intentionally lightweight: it runs DTD interactively and shows MS-LRC evidence through precomputed examples. The full local MS-LRC pipeline remains available for machines with transformer dependencies and model cache.

## Current Submission Status

A workshop-paper version of this work, **"DTD-LRC: Uncertainty-Gated Evidence Escalation for Open-World Machine-Generated Text Detection"**, has been submitted to **UncertaiNLP 2026 @ EMNLP** via OpenReview.

The UncertaiNLP-oriented paper files are available in:

```text
paper/DTD-LRC_UncertaiNLP_2026/
```

## Current Artifact Links

- Live demo: https://huggingface.co/spaces/YohanChow/DTD-LRC-AI-Text-Detector
- Direct app URL: https://yohanchow-dtd-lrc-ai-text-detector.hf.space
- Demo video: https://youtu.be/Z0NXF_Ghasg
- Artifact links file: [ARTIFACT_LINKS.md](ARTIFACT_LINKS.md)

## Actual Application Entry Points

The original Flask application in this repository is:

```text
src/web_app.py
```

It serves the lightweight DTD web/API demo on port `5000`.

For the EMNLP artifact and Hugging Face Spaces deployment, this branch adds a root entry point:

```text
app.py
```

Use `app.py` for reviewer demos and Spaces. It loads `models/optimized_dtd_model.pkl` when available, runs DTD online, exposes `/api/detect`, and exposes precomputed MS-LRC evidence through `/api/mslrc_examples`.

## Quick Start

```bash
git clone https://github.com/haveanicedaymydear/AI-Text-Cascade-Detect.git
cd AI-Text-Cascade-Detect
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Expected local URL:

```text
http://127.0.0.1:5000
```

The original lightweight app can also be started with:

```bash
python src/web_app.py
```

That command also serves `http://127.0.0.1:5000`.

## API Examples

Health check:

```bash
curl http://127.0.0.1:5000/api/health
```

Single detection:

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "The implementation of artificial intelligence systems requires comprehensive evaluation of data quality and reliability."}'
```

Precomputed MS-LRC examples:

```bash
curl http://127.0.0.1:5000/api/mslrc_examples
curl http://127.0.0.1:5000/api/mslrc_examples?id=uncertain_sample
```

Example output fields:

```json
{
  "success": true,
  "result": {
    "prediction": "AI Generated",
    "ai_probability": 0.5401541143717332,
    "human_probability": 0.4598458856282668,
    "confidence": 0.5401541143717332,
    "prediction_time_ms": 34.3,
    "stage_used": "DTD (uncertain; see precomputed MS-LRC examples)",
    "interactive_stage": "DTD",
    "ms_lrc_mode": "precomputed_examples",
    "uncertainty_band": [0.41, 0.61],
    "is_uncertain": true
  }
}
```

More examples are stored in:

```text
examples/human_sample.txt
examples/ai_sample.txt
examples/uncertain_sample.txt
examples/sample_output.json
examples/mslrc_precomputed.json
```

## Dependency Sets

Lightweight web demo:

```bash
pip install -r requirements.txt
```

This installs Flask, CORS support, NumPy, scikit-learn, and Gunicorn for deployment. It does not install transformer runtimes.

Full local MS-LRC / benchmark pipeline:

```bash
pip install -r requirements-full.txt
```

This adds `torch`, `transformers`, `accelerate`, and plotting/data-analysis libraries. Use this only when running the full local cascade or benchmark scripts.

## Demo Modes

1. **Online / reviewer demo**
   - Command: `python app.py`
   - Runs interactive DTD detection.
   - Shows MS-LRC evidence cards from `examples/mslrc_precomputed.json`, with raw JSON available in the interface.
   - Avoids loading transformer models during web startup.

2. **Original DTD Flask demo**
   - Command: `python src/web_app.py`
   - Runs the original trained DTD model on port `5000`.

3. **Full local cascade**
   - Install: `pip install -r requirements-full.txt`
   - Command: `python src/cascade_demo_app.py`
   - Default URL: `http://127.0.0.1:5001`
   - Requires transformer model downloads/cache and more memory than the lightweight demo.

## Hugging Face Spaces Deployment

This branch includes a Docker-based Space configuration:

```text
Dockerfile
requirements.txt
app.py
templates/artifact_demo.html
examples/
```

Create a Hugging Face Space with SDK `Docker`, point it at this repository/branch, and use the included `Dockerfile`. The container sets `PORT=7860`, while local execution defaults to `PORT=5000`.

If deploying manually, the Space should run:

```bash
python app.py
```

## System Idea

```text
Input text
  |
  v
Stage 1: DTD
  - TF-IDF features
  - lexical, syntactic, punctuation, repetition, and style features
  - fast classifier -> AI probability p1
  |
  | if p1 is outside [0.41, 0.61]
  v
Return Stage-1 decision

If p1 is inside [0.41, 0.61]:
  |
  v
Stage 2: MS-LRC
  - NLL/byte scoring across model families and scales
  - ladder response curve features
  - family-scale response matrix
  - second-stage decision plus interpretable evidence
```

## Evaluation Summary

Headline metrics from the current project report:

| Component | Metric | Value |
|---|---:|---:|
| Stage 1 DTD | AUC | 0.9899 |
| Stage 1 DTD | F1 | 0.96 |
| Stage 1 DTD | Precision / Recall | 0.96 / 0.96 |
| Stage 1 DTD | Average latency | 38 ms |
| Gate | Stage-2 routing band | [0.41, 0.61] |
| Stage 2 MS-LRC smoke | AUC | 0.9622 |
| Stage 2 MS-LRC smoke | Accuracy | 0.9333 |
| Stage 2 MS-LRC smoke | F1 | 0.9375 |
| Cascade smoke | Stage-2 usage | 31 / 120 |

These results are internal system-validation results for the demonstration artifact. Direct numerical comparison with external detectors should only be reported when all baselines are rerun under the same data and evaluation protocol.

## Limitations and Ethical Use

DTD-LRC is a probabilistic aid, not an authority. It should not be used as the sole basis for punitive academic, employment, or moderation decisions.

Known limitations include:

- false positives on short, formulaic, translated, heavily edited, creative, or stylistically unusual human writing
- sensitivity to domain shift and generator shift
- calibration drift under unseen distributions
- reduced reliability on multilingual or code-mixed text unless separately evaluated
- high compute cost for full MS-LRC because it requires multiple language-model evaluations

Recommended use is evidence-assisted review: surface uncertainty, confidence, and diagnostic evidence for human review rather than replacing judgment.
