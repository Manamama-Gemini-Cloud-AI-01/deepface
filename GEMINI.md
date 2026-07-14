# DeepFace Project - Operational Context

This document tracks persistent project-specific knowledge to prevent reinventing wheels.

## NumPy Serialization Issue (PR #1614)
- **Status**: PR #1614 (feat(api): Add NumpyJSONProvider to handle NumPy types) is submitted and awaiting review.
- **Goal**: Implement a global `NumpyJSONProvider` in `deepface/api/src/app.py` to fix serialization errors in Flask.
- **Persistence**: The implementation is preserved in the branch `fix/numpy-serialization-v3` on the fork `Manamama-Gemini-Cloud-AI-01/deepface`.
- **Regression Test**: A regression test `test_numpy_serialization_regression` is included in `tests/unit/test_api.py`.
