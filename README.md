# sem-defect-modal

SEM defect-detection pipeline for FIB-SEM cross-sections of battery electrodes, running on Modal.

See `docs/SPEC.md` for the authoritative build spec.

## Layout

- `sem/` — core logic, pure functions, no Modal imports (CPU-testable).
- `modal_app.py` — thin Modal wrapper (functions, classes, endpoints).
- `tests/` — pytest suite, CPU-only.
- `requirements.lock` — pinned dependencies (uv pip freeze).

## Local dev

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.lock
pytest
```

## QC evaluation foundation

The CPU-only QC interfaces, deterministic `classical_v1` baseline, similarity
analysis, and stem-level split manifest use Python 3.10 and the pinned
`requirements.txt` dependencies. Set `SEM_DATA_ROOT` and `SEM_WORK_ROOT` to
override the data and derived-output locations (defaults are
`/home/ubuntu/data/sem` and `/home/ubuntu/work/qc`).

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/stem_similarity.py
.venv/bin/python scripts/run_method.py classical_v1
.venv/bin/python scripts/make_split.py
```

Predictions, similarity reports, and overlays are written under
`$SEM_WORK_ROOT`; the deterministic manifest and its summary are generated in
`data/splits/`. Similarity results do not automatically merge groups: only
explicit entries in `configs/groups_override.yaml` change split grouping.
