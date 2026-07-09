# DTD-LRC_UncertaiNLP_2026

This directory contains the UncertaiNLP 2026 @ EMNLP version of the DTD-LRC paper.

## Target Venue

Third Workshop on Uncertainty-Aware NLP @ EMNLP 2026 (UncertaiNLP 2026)

Current status: manuscript in preparation. Do not describe the paper as submitted until the OpenReview submission is actually completed.

## Paper Framing

This version reframes the original EMNLP System Demonstration draft as an uncertainty-aware NLP paper:

- Original demo framing: interpretable two-stage cascade system for AI-text detection.
- UncertaiNLP framing: uncertainty-gated evidence escalation for open-world AI-text detection.

The central claim is that low-confidence first-stage predictions should trigger evidence escalation rather than unsupported binary decisions.

## Files

- `main.tex` — UncertaiNLP-oriented LaTeX draft.
- `references.bib` — references used by the draft.
- `openreview_fields.md` — suggested OpenReview title, keywords, TL;DR, and research areas.

## Compile Notes

The draft uses the official ACL/EMNLP style package. When moving to Overleaf, include:

- `main.tex`
- `references.bib`
- official ACL/EMNLP style files such as `acl.sty` and `acl_natbib.bst`
- the figure files referenced in `main.tex`

The current figure paths assume this directory sits under `paper/` and can reference `../figures/...`. Adjust paths as needed in Overleaf.
