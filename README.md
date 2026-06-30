# DTD-LRC: Interpretable Two-Stage AI-Text Detection

This repository contains the demonstration system for **DTD-LRC: An Interpretable Two-Stage Cascade for Open-World Machine-Generated Text Detection**.

DTD-LRC is designed for practical machine-generated text detection under changing domains, generators, and writing styles. Instead of relying on a single detector score, it combines a fast first-stage detector with a deeper second-stage ladder-response module for uncertain cases.

## System Idea

The system uses a confidence-gated cascade:

```text
Input text
   │
   ▼
Stage 1: DTD
   ├─ TF-IDF features
   ├─ lexical, syntactic, punctuation, repetition, and structural/style features
   └─ fast classifier → AI probability p1
   │
   ├─ if p1 is outside [0.41, 0.61] → return Stage-1 decision
   │
   └─ if p1 is inside  [0.41, 0.61] → invoke Stage 2
          │
          ▼
Stage 2: MS-LRC
   ├─ NLL/byte scoring across model families and scales
   ├─ ladder response curve features
   ├─ family-scale response matrix
   └─ second-stage decision + interpretable evidence
```

## Components

### Stage 1: DTD

DTD is a lightweight first-stage detector. It uses:

- TF-IDF vectors
- lexical features
- sentence-structure features
- punctuation patterns
- repetition indicators
- structural and style markers

These features are concatenated into a feature vector and passed to a fast classifier that returns:

- predicted label
- AI probability
- human probability
- confidence score
- runtime

### Stage 2: MS-LRC

MS-LRC is invoked only for uncertain cases. It treats language models as measurement instruments and computes length-normalized negative log-likelihood (`NLL/byte`) across multiple scales and families.

The current research prototype uses Qwen-style scale ladders and extracts features such as:

- early drop
- late drop
- overall drop
- drop ratio
- concavity flag
- cross-family spread

The goal is not to rely on a single perplexity threshold, but to inspect how predictability changes across model capacity and family variants.

## Demonstration Modes

For reviewer-facing demonstration, the recommended deployment strategy is:

1. **Lightweight online demo**  
   Runs the interactive front end and fast DTD detection reliably.

2. **Precomputed MS-LRC examples**  
   Shows ladder response curves and family-scale NLL matrices for selected examples.

3. **Full installable pipeline**  
   Allows local execution of the complete MS-LRC scoring workflow when model cache and hardware are available.

This design avoids forcing a public web demo to load multiple large language models while still keeping the full pipeline reproducible.

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

These results are intended as internal system validation for the demonstration paper. Direct numerical comparison with external detectors should only be reported if all baselines are rerun under the same data and evaluation protocol.

## Quick Start

> The exact command may need to be adjusted to match the local project entry point.

```bash
git clone https://github.com/haveanicedaymydear/AI-Text-Cascade-Detect.git
cd AI-Text-Cascade-Detect
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## API

Expected endpoints in the demonstration prototype:

```text
GET  /api/health
POST /api/detect
```

Example detection request:

```bash
curl -X POST http://127.0.0.1:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Paste a paragraph here for detection."}'
```

Expected response fields include:

```json
{
  "label": "Human-written Text or AI-generated Text",
  "ai_probability": 0.32,
  "human_probability": 0.68,
  "runtime_ms": 44.5,
  "stage_used": "DTD or MS-LRC"
}
```

## Reviewer Checklist

Before submission, the repository should contain:

- [ ] stable online demo link or installable package
- [ ] demo video link
- [ ] `requirements.txt`
- [ ] clear run command
- [ ] example inputs and outputs
- [ ] screenshots
- [ ] evaluation result files or reproduction scripts
- [ ] license
- [ ] limitations and ethical-use statement

## Limitations and Ethical Use

DTD-LRC is a probabilistic aid, not an authority. It should not be used as the sole basis for punitive academic or moderation decisions.

Known limitations include:

- false positives on short, formulaic, creative, or stylistically unusual human writing
- sensitivity to domain and generator shift
- high compute cost for the full MS-LRC 3-by-3 ladder
- possible calibration shift under unseen distributions

The recommended use is evidence-assisted review: the system should surface uncertainty and diagnostic evidence rather than replace human judgment.

## Paper Draft

A working EMNLP System Demonstration draft is placed under:

```text
paper/emnlp_demo_draft.tex
paper/references.bib
```

The draft assumes the official EMNLP/ACL style files are added before compilation.

## Citation

Citation information will be added after submission.
