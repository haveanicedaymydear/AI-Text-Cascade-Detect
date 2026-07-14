# Reproducibility and Claim Boundaries

This document separates the public artifact, internal validation results, and claims that still require broader external evaluation.

## Publicly reproducible components

The repository exposes:

- the lightweight DTD web/API entry point;
- the original DTD Flask application;
- precomputed MS-LRC evidence examples for reviewer-facing inspection;
- the full local cascade entry point for machines with transformer dependencies and model cache;
- example inputs and outputs;
- paper files and artifact links;
- lightweight and full dependency specifications.

## Artifact modes

### Lightweight reviewer mode

```bash
pip install -r requirements.txt
python app.py
```

This mode performs interactive Stage-1 DTD inference and surfaces precomputed Stage-2 MS-LRC evidence. It is designed for low-friction inspection and deployment.

### Full local cascade mode

```bash
pip install -r requirements-full.txt
python src/cascade_demo_app.py
```

This mode requires transformer model downloads or cache, substantially more memory, and longer execution time.

## Reported validation results

The README reports the following internal artifact-validation results:

- Stage-1 DTD AUC: 0.9899;
- Stage-1 DTD F1: 0.96;
- Stage-1 average latency: 38 ms;
- Stage-2 MS-LRC smoke-evaluation AUC: 0.9622;
- Stage-2 MS-LRC smoke-evaluation accuracy: 0.9333;
- Stage-2 MS-LRC smoke-evaluation F1: 0.9375;
- Stage-2 routing: 31 of 120 examples.

These values should be interpreted as project-level validation, not as a universal performance guarantee.

## Comparison boundary

Direct comparison with external AI-text detectors is only valid when all systems are rerun under the same:

- dataset split;
- generator families and versions;
- preprocessing;
- calibration procedure;
- threshold selection rule;
- evaluation metrics;
- compute environment.

The repository therefore does not claim state-of-the-art performance based solely on unmatched numbers from other papers.

## Open-world boundary

The phrase *open-world* refers to evaluation across generator families and distributions beyond a single closed training source. It does not imply reliable detection of every future model, language, domain, editing process, or adversarial transformation.

## Ethical boundary

DTD-LRC is a probabilistic decision-support artifact. It should not be used as the sole basis for punitive academic, employment, moderation, or authorship decisions.

Recommended use is evidence-assisted review with:

- human judgment;
- uncertainty inspection;
- provenance and contextual evidence;
- an appeal or correction path;
- explicit acknowledgement of false-positive risk.
