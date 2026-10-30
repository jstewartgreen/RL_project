# SimPLe world model (Pong reproduction)

Goal: world model env' for SimPLe. Spec: docs/SPEC.md (authoritative). Full paper: docs/paper.md.
Commands: `pytest tests/` · `python train_wm.py --config configs/smoke.yaml` (must finish in <2 min)
Docs: spec in SPEC.md (authoritative), phases in PLAN.md, full paper in docs/paper.md.
Current phase: P0. Only implement SPEC sections tagged with the current phase or earlier.

Rules:
- Read-only: reference/ (T2T code), teammates/, docs/paper.md, docs/arxiv_src/.- SPEC.md: you may resolve a [TBD] only by citing T2T (file and line) and tagging it [T2T].
  Any other change to SPEC.md needs my approval. Never change a sourced value.
- PLAN.md: tick checkboxes as exit criteria are met. Ask before changing "Current phase".
- Never change a SPEC hyperparameter in code; log any deviation in DEVIATIONS.md with the reason.
- If the spec is ambiguous, stop and ask rather than guess.
- Never edit tests to make them pass.
- Cite the paper section in a comment for every non-obvious design choice.