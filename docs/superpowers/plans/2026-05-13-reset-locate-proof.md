# Reset Locate Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reset the repo to a small, query-based desktop icon grounding proof that captures or loads a screenshot, draws grid/OCR/candidate debug images, and returns deterministic coordinates for the best candidate.

**Architecture:** Use a flat Python package while the proof is still unstable. The first implementation is deterministic: Windows OCR or test-supplied OCR words create label boxes, grid cells are visible debug context and a placement prior, fuzzy query matching ranks candidates, and the selected center coordinate is written to JSON. LLM/ScreenSeekeR work stays out of production code until this proof is reliable.

**Tech Stack:** Python 3.11+, `uv`, Pillow, `mss`, Windows OCR through PyWinRT on Windows, `pytest`.

---

## File Structure

- Delete old layered placeholder modules:
  - `src/notepad_grounding/automation/`
  - `src/notepad_grounding/grounding/`
  - `src/notepad_grounding/data/`
  - `src/notepad_grounding/config.py`
- Delete old debug outputs and old docs:
  - `output/debug/*.png`
  - `docs/assignment-analysis.md`
  - `docs/project-context.md`
  - `docs/project-plan.md`
  - old `docs/superpowers/plans/2026-05-12-milestones-1-2.md`
- Keep:
  - `pyproject.toml`
  - `src/notepad_grounding/__init__.py`
- Create:
  - `src/notepad_grounding/capture.py`: screenshot capture and screenshot loading.
  - `src/notepad_grounding/ocr.py`: OCR data types, Windows OCR backend, grouping.
  - `src/notepad_grounding/grounding.py`: grid construction, candidate inference, fuzzy scoring, locate orchestration.
  - `src/notepad_grounding/annotate.py`: draw grid, OCR labels, candidates, selected target.
  - `src/notepad_grounding/main.py`: `locate` CLI.
  - `tests/test_grounding.py`: grid, candidate, scoring, and locate result unit tests.
  - `tests/test_cli.py`: CLI parser smoke tests.
- Rewrite:
  - `README.md`: current proof only.
  - `AGENTS.md`: current scope and reset rules only.

## Task 1: Cleanup

**Files:**
- Delete: old placeholder/debug modules and old docs listed above.
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Remove clutter**

Use `apply_patch` delete hunks for tracked placeholder files and docs. Use `rm` only for ignored generated artifacts such as `output/debug/*.png` and caches.

- [ ] **Step 2: Rewrite instructions**

`AGENTS.md` must say:
- use `uv`;
- run final automation in Windows;
- current milestone is `locate --query`;
- grid is a debug view and prior, not a hard guarantee;
- Windows OCR is preferred;
- LLM work is optional and disabled until deterministic proof works.

## Task 2: Grounding Tests

**Files:**
- Create: `tests/test_grounding.py`

- [ ] **Step 1: Write failing grid test**

```python
from notepad_grounding.grounding import build_grid


def test_build_grid_covers_screen_with_expected_cell_size():
    cells = build_grid((1920, 1080), cell_width=96, cell_height=96, taskbar_height=48)

    assert cells[0].box == (0, 0, 96, 96)
    assert cells[-1].box[3] <= 1032
    assert len(cells) > 100
```

Run: `uv run pytest tests/test_grounding.py::test_build_grid_covers_screen_with_expected_cell_size -v`
Expected: FAIL because `notepad_grounding.grounding` does not exist.

- [ ] **Step 2: Write candidate/scoring tests**

```python
from notepad_grounding.ocr import OcrLine
from notepad_grounding.grounding import infer_candidates, locate_from_lines


def test_infer_candidates_places_icon_above_label():
    line = OcrLine(text="Notepad", confidence=100, box=(700, 460, 760, 478))

    candidates = infer_candidates([line], screen_size=(1920, 1080), query="Notepad")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.label_text == "Notepad"
    assert candidate.icon_box[3] <= line.box[1]
    assert candidate.score > 0.8


def test_locate_from_lines_selects_best_fuzzy_query_match():
    lines = [
        OcrLine(text="Recycle Bin", confidence=100, box=(20, 80, 90, 100)),
        OcrLine(text="Note pad", confidence=100, box=(700, 460, 760, 478)),
    ]

    result = locate_from_lines(lines, screen_size=(1920, 1080), query="Notepad")

    assert result is not None
    assert result.candidate.label_text == "Note pad"
    assert result.center == result.candidate.icon_center
```

Run: `uv run pytest tests/test_grounding.py -v`
Expected: FAIL because implementation is missing.

## Task 3: Minimal Grounding Implementation

**Files:**
- Create: `src/notepad_grounding/grounding.py`
- Create: `src/notepad_grounding/ocr.py`

- [ ] **Step 1: Implement data types and grid**

Create immutable dataclasses:
- `OcrLine(text: str, confidence: float, box: Box)`
- `GridCell(index: int, row: int, col: int, box: Box)`
- `Candidate(id: int, label_text: str, label_box: Box, icon_box: Box, combined_box: Box, score: float, score_notes: list[str])`
- `LocateResult(query: str, center: tuple[int, int], candidate: Candidate, candidates: list[Candidate], grid: list[GridCell])`

Implement `build_grid(screen_size, cell_width=96, cell_height=96, taskbar_height=48)`.

- [ ] **Step 2: Implement fuzzy scoring and candidate inference**

Use `difflib.SequenceMatcher` over normalized alphanumeric text.
Score exact normalized query match as `1.0`; fuzzy matches use the ratio.
Infer icon box above the label with a conservative 64px icon area clamped to the screen.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_grounding.py -v`
Expected: PASS.

## Task 4: OCR, Capture, Annotation, CLI

**Files:**
- Create: `src/notepad_grounding/capture.py`
- Create: `src/notepad_grounding/annotate.py`
- Modify: `src/notepad_grounding/main.py`
- Create/modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI smoke test**

```python
from notepad_grounding.main import build_parser


def test_locate_parser_accepts_query_and_image():
    parser = build_parser()
    args = parser.parse_args(["locate", "--query", "Notepad", "--image", "screen.png"])

    assert args.command == "locate"
    assert args.query == "Notepad"
    assert str(args.image) == "screen.png"
```

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL until `build_parser` is rewritten.

- [ ] **Step 2: Implement CLI**

Command:

```text
uv run notepad-grounding locate --query Notepad --out-dir output/debug
uv run notepad-grounding locate --query Notepad --image output/debug/screen.png --out-dir output/debug
```

Outputs:
- `<timestamp>-raw.png` when capturing live screenshot;
- `<timestamp>-grid.png`;
- `<timestamp>-ocr.png`;
- `<timestamp>-candidates.png`;
- `<timestamp>-result.json`.

- [ ] **Step 3: Implement annotation**

`annotate.py` draws:
- light gray grid cells with row/column labels every few cells;
- green OCR label boxes;
- yellow candidate combined boxes with score labels;
- blue selected icon box and center crosshair.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v`
Expected: PASS.

## Task 5: Docs

**Files:**
- Rewrite: `README.md`
- Rewrite: `AGENTS.md`

- [ ] **Step 1: README includes current workflow**

Document:
- setup with `uv sync`;
- Windows run command;
- replay from image command;
- output artifact meanings;
- why grid is not a guarantee;
- why LLM is deferred;
- next milestone after locate proof.

- [ ] **Step 2: Verify docs mention no deleted commands**

Run: `rg "candidate-proof|ocr-proof|screenshot-proof|template|verifier|notepad.py|posts.py" README.md AGENTS.md docs`
Expected: no matches except this plan if searched globally.

## Task 6: Final Verification

- [ ] **Step 1: Run full tests**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Run CLI help**

Run: `uv run notepad-grounding --help`
Expected: shows `locate`.

- [ ] **Step 3: Inspect git diff**

Run: `git status --short`
Expected: deleted clutter, new flat modules, rewritten docs.

