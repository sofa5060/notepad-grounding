# Reduction Tracking

File-by-file audit and reduction status for `src/notepad_grounding/`.

| # | File | Lines | Status | Notes |
|---|------|-------|--------|-------|
| 1 | `api.py` | 33 | done | Removed ApiError → plain Exception, trimmed docstring |
| 2 | `capture.py` | 12 | done | Removed unused require_windows parameter |
| 3 | `cli.py` | 84 | done | Removed ApiError import + dead except branch |
| 4 | ~~`click_grid.py`~~ | — | **deleted** | Merged into geometry.py + images.py |
| 5 | `geometry.py` | 80 | done | Unified: 1-based row/col, id_fmt param, added cell_by_id + offset_point |
| 6 | `images.py` | 172 | done | Absorbed draw_click_grid, draw_full_click_marker, crop_around_point |
| 7 | `locate.py` | 539 | done | Removed _cell_by_id, replaced ClickGridCell → GridCell, updated imports |
| 8 | `desktop.py` | 157 | pending | pyautogui helpers |
| 9 | `desktop_interactions.py` | 90 | pending | High-level interaction steps |
| 10 | `env.py` | 26 | pending | .env loader |
| 11 | `flow.py` | 351 | pending | Orchestrator |
| 12 | `main.py` | 10 | pending | Entry point |
| 13 | `models.py` | 66 | pending | Dataclasses + Pydantic models |
| 14 | `prompts.py` | 118 | pending | LLM prompt builders |
| 15 | `reviewers.py` | 247 | pending | OpenAI reviewer |
| 16 | `vision.py` | 133 | pending | OpenAI vision client |
