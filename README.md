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
