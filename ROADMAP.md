# Notepad Grounding Roadmap

> Working memory for the take-home project. Keep this file focused on the current goal: make `locate` reliable, explainable, and easy to review.

## Current Goal

Build a reliable query-based command:

```text
uv run notepad-grounding locate --query Notepad --out-dir output
```

The command should visually find a desktop icon or shortcut, save useful artifacts, and return a deterministic center coordinate. The LLM can choose labeled grid cells and final crop-local boxes; code owns all screen-coordinate math.

## Status Legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[?]` Needs investigation or decision

## Current Working Baseline

- `[x]` Primary `llm-visual` locate flow exists.
- `[x]` Coarse-to-fine grid search exists.
- `[x]` Final crop-local bbox detection exists.
- `[x]` Final bbox self-review uses `previous_response_id`.
- `[x]` Full-screen bbox overlay artifact exists.
- `[x]` `grid-ocr` fallback experiment is preserved.
- `[x]` Reviewer module exists for post-action automation validation.
- `[x]` CLI supports `locate --query`.
- `[x]` Artifacts are saved to output directories.
- `[x]` `.env` support exists.
- `[x]` Default model is configurable through `OPENAI_MODEL`.

## Stage 1: Add a Judge During Grid Grounding

**Problem:** The current grid descent is fragile. If the model picks the wrong cell, every later crop is based on the wrong area. Re-running the same prompt can also repeat the same cached mistake.

**Goal:** Validate each selected grid crop before committing to it.

- `[ ]` Add a structured judge result for grid validation:
  - `contains_target: bool`
  - `confidence: float`
  - `rationale: str`
  - optional `visible_evidence: str`
- `[ ]` After `choose_cell`, crop the selected cell from the original image with the normal padding.
- `[ ]` Send that selected crop to the judge with the query and ask whether it contains either:
  - the requested icon/app visual, or
  - a desktop item label matching the query.
- `[ ]` If the judge says `contains_target=true`, continue the existing flow.
- `[ ]` If the judge says `contains_target=false`, continue the original chooser conversation with `previous_response_id` and tell it the selected cell was rejected.
- `[ ]` Re-prompt the chooser to select a different valid cell, ideally passing rejected cell IDs and the judge rationale.
- `[ ]` Cap retries per grid round so failures end cleanly instead of looping.
- `[ ]` Save judge artifacts:
  - selected crop image
  - judge decision JSON
  - retry prompt metadata
  - rejected cell list
- `[ ]` Make the judge model configurable separately from the chooser model if useful.
- `[ ]` Default to the same OpenAI vision model until another vision-capable provider is proven in Windows testing.
- `[ ]` Add focused tests around the retry/control-flow logic with fake clients.

**Acceptance criteria:**

- A wrong selected cell can be rejected without restarting the whole locate command.
- The next chooser request continues the same conversation instead of replaying the same cached request.
- The result JSON shows which cells were accepted or rejected.
- Locate fails with a clear error after bounded retries.

## Stage 2: Simplify the Code Structure

**Problem:** The repo currently feels heavier than the assignment needs. There are many small files and many tests for a take-home proof.

**Goal:** Keep the working behavior while making the project smaller and easier to read.

- `[ ]` Trace the actual runtime path for `locate`.
- `[ ]` Decide which modules are genuinely part of the assignment proof.
- `[ ]` Keep flow isolation where it helps clarity:

```text
src/notepad_grounding/
  shared/
  flows/
    llm_visual_search/
    grid_ocr/
```

- `[ ]` Remove dead code or unused helpers.
- `[ ]` Merge tiny shared modules only when it reduces real friction.
- `[ ]` Keep `grid-ocr` as a fallback experiment, but make it clearly secondary.
- `[ ]` Reduce tests to a small, high-value set:
  - CLI smoke tests
  - deterministic geometry / bbox mapping tests
  - judge retry behavior tests
  - final result serialization tests
- `[ ]` Remove tests that only lock down unimportant implementation details.
- `[ ]` Make README and AGENTS match the final structure.

**Important decision:** Do not delete useful tests just to make the repo look smaller. Prefer fewer, better tests over no tests.

## Stage 3: Produce Assignment Evidence Screenshots

**Problem:** The assignment expects annotated screenshots showing successful detection in different desktop locations and conditions.

**Goal:** Generate a small, convincing evidence set from Windows.

- `[ ]` Generate top-left scenario:
  - Notepad shortcut in top-left
  - normal screen size
  - small desktop icons
  - detection overlay saved
- `[ ]` Generate center scenario:
  - Notepad shortcut near center
  - larger icon size
  - detection overlay saved
- `[ ]` Generate bottom-right scenario:
  - Notepad shortcut near bottom-right
  - changed wallpaper or busier background
  - detection overlay saved
- `[ ]` Add optional extra scenario if time allows:
  - different resolution or display scaling
  - dark/light theme switch
  - partially obscured desktop icon
  - multiple similar desktop items
- `[ ]` Store final evidence under `docs/screenshots/`.
- `[ ]` Add short notes explaining each scenario and the returned center coordinate.
- `[ ]` Confirm final runtime testing happens in Windows, not macOS.

**Acceptance criteria:**

- At least 3 annotated screenshots are in the repo.
- The screenshots cover top-left, center, and bottom-right positions.
- Each screenshot has a matching result artifact or coordinate note.

## Stage 4: Search by Icon Semantics and Label

**Problem:** The current prompt can over-focus on the label text. The assignment wants an extensible visual grounding system, not just OCR-like label matching.

**Goal:** Treat the query as both a possible label and a semantic app/icon target.

- `[ ]` Update chooser prompts to search for the desktop item matching the query by:
  - visual icon/app semantics, and
  - visible label text when available.
- `[ ]` Update final bbox prompt to box only the icon graphic, not the label.
- `[ ]` Add judge wording that accepts either visual evidence or a matching label, but asks for the reason.
- `[ ]` Test a renamed Notepad shortcut, such as `Notes`.
- `[ ]` Test a case where the label is hard to read or not useful.
- `[ ]` Test similar names, such as Notepad vs Notepad++.
- `[ ]` Decide how to handle multiple matching candidates:
  - return candidates and rank them, or
  - ask the LLM to pick the best candidate with rationale.
- `[ ]` Document the behavior in README as query-based visual grounding, not label-only search.

**Acceptance criteria:**

- `--query Notepad` can still find the Notepad shortcut when the label is not exactly `Notepad`.
- The final click center is computed from the icon graphic bbox.
- README clearly explains the icon-plus-label search behavior.

## Coverage Matrix

| Scenario | Status | Stage | Notes |
|---|---:|---:|---|
| Icon at top-left | `[x]` manual / `[ ]` artifact | 3 | Need final annotated screenshot |
| Icon at center | `[x]` manual / `[ ]` artifact | 3 | Need final annotated screenshot |
| Icon at bottom-right | `[x]` manual / `[ ]` artifact | 3 | Need final annotated screenshot |
| Small icon size | `[x]` | 3 | Reconfirm in final evidence run |
| Medium icon size | `[x]` | 3 | Reconfirm in final evidence run |
| Large icon size | `[ ]` | 3 | Add if time allows |
| Light theme | `[ ]` | 3 | Not explicitly tested |
| Dark theme | `[ ]` | 3 | Not explicitly tested |
| Solid background | `[?]` | 3 | Not explicitly tested |
| Busy/custom background | `[x]` | 3 | Tested manually with drawings |
| Partially obscured icon | `[ ]` | 3/4 | Useful robustness case |
| Multiple matching icons | `[ ]` | 4 | Needs selection strategy |
| Similar names | `[ ]` | 4 | Notepad vs Notepad++ |
| Renamed shortcut | `[ ]` | 4 | Tests visual semantics |
| Different resolution | `[ ]` | 3 | Only 1920x1080 is assumed |
| API unavailable | `[ ]` | Later | Relevant only to automation flow |
| Existing files in target directory | `[x]` | Later | Existing automation clears directory |
| Retry logic after actions | `[x]` | Later | Existing automation has retry/reviewer |
| Wrong app recovery | `[x]` | Later | Existing automation closes wrong app |

## Interview Prep

### Architecture Answers

- `[x]` Coarse-to-fine grid is used because it keeps coordinate math deterministic while letting the LLM make visual choices.
- `[x]` The design aligns with ScreenSeekeR / ScreenSpot-Pro: reduce the search area before precise localization.
- `[x]` LLM vision is used because template matching is brittle across wallpapers, themes, icon sizes, and renamed shortcuts.
- `[x]` OCR-only is kept as a fallback experiment because labels help, but the assignment needs visual icon grounding.

### Robustness Answers To Prepare

- `[ ]` Explain the new judge layer for rejecting bad grid choices.
- `[ ]` Explain bounded retries and artifact logging.
- `[ ]` Explain how multiple similar icons would be ranked.
- `[ ]` Explain when the approach still fails:
  - icon fully hidden
  - ambiguous identical shortcuts
  - model/provider outage
  - unsupported resolution/scaling not tested

### Performance Answers To Prepare

- `[ ]` Measure average locate time on Windows.
- `[ ]` Record number of LLM calls per successful locate.
- `[ ]` Mention optimization options:
  - smaller/faster model for judging
  - fewer grid rounds after confidence is high
  - cached last-known positions
  - parallel or cheaper candidate checks

## Later / Parking Lot

These were already in previous notes and should not be lost, but they are not the next priority unless the assignment scope changes.

- `[ ]` Full automation loop:
  - fetch posts
  - open Notepad
  - type content
  - save files
  - close and repeat
- `[ ]` Graceful handling for JSONPlaceholder/API failure.
- `[ ]` Replace-file popup handling.
- `[ ]` Keep or split OCR into a separate future project.
- `[ ]` Explore a pure OCR version for comparison.
- `[ ]` Consider a local vision model to reduce cost/latency.
- `[ ]` Consider synthetic training data or a small detector if the task grows beyond the take-home.
- `[ ]` Consider memory of last known icon positions as an optimization, not as the primary proof.

## Quick Checklist Before Final Submission

- `[ ]` Stage 1 judge/retry flow implemented and tested.
- `[ ]` Stage 2 cleanup completed without breaking `locate`.
- `[ ]` Stage 3 screenshots generated in Windows.
- `[ ]` Stage 4 prompt behavior tested with renamed/similar labels.
- `[ ]` README updated and accurate.
- `[ ]` `.env.example` matches required config.
- `[ ]` `uv run pytest -v` passes.
- `[ ]` `uv run notepad-grounding locate --query Notepad --out-dir output` works in Windows.
- `[ ]` Repo pushed to GitHub.

## Decisions Log

| Date | Decision | Context |
|---|---|---|
| 2026-05-13 | Keep `llm-visual` as the primary flow | OCR remains a fallback experiment |
| 2026-05-13 | Add a judge before committing to each selected grid crop | Prevents wrong-cell descent and repeated cached mistakes |
| 2026-05-13 | Use `previous_response_id` for correction loops | Keeps conversation context when rejecting a choice or refining a bbox |
| 2026-05-13 | Keep screen-coordinate math in deterministic code | LLM may choose cells or crop-local bboxes only |
| 2026-05-13 | Final evidence must be produced on Windows | macOS/Parallels host behavior is not enough for final testing |

---

*Last updated: 2026-05-13*
