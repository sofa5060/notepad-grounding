# Notepad Grounding — Project Plan

> This document tracks what has been implemented, what needs to be done next, and what we plan to add in the future.

---

## Status Legend

- `[ ]` — Not started
- `[~]` — In progress
- `[x]` — Done
- `[?]` — Needs investigation / decision

---

## 1. Done (Completed)

### LLM Visual Search Flow (Primary)
- `[x]` Coarse-to-fine grid search (`choose_cell`) with 3-4 rounds
- `[x]` Iterative bbox refinement (`locate_icon_with_validation`) — LLM returns crop-local bbox, draws red box, LLM self-reviews in same conversation
- `[x]` Double-click on computed center
- `[x]` Recovery from wrong-app clicks — Alt+F4 then Escape (safe for apps + desktop)
- `[x]` Pop-up handling — press Enter on unexpected dialogs
- `[x]` Error retry — 3 attempts with 1.0s delay per post (matches PDF requirement)
- `[x]` Reviewer validates every action via screenshots (instructor + Pydantic)

### Automation Workflow (Per PDF Section 3)
- `[x]` Full loop: screenshot → ground → double-click → type → save → close → repeat
- `[x]` Fetches first 10 posts from JSONPlaceholder API
- `[x]` Types content in format: `Title: {title}\n\n{body}`
- `[x]` Saves as `post_{id}.txt` in `Desktop/tjm-project`
- `[x]** Closes Notepad after EACH post, then re-opens for next (matches PDF workflow)**
- `[x]` Target directory auto-cleared at start to avoid replace dialogs
- `[x]` LLM reviewer validates every step:
  - `[x]` Post-double-click: is the correct app open?
  - `[x]` Post-type: is the expected text in the editor?
  - `[x]` Post-save: does the file exist? Handles pop-ups.
  - `[x]` Post-close: is the file still there?

### Scenarios Already Tested Manually
- `[x]` Small desktop icons (96px)
- `[x]` Medium desktop icons (standard)
- `[x]` Icons moved to different desktop positions
- `[x]` Non-solid wallpaper background (with drawings)
- `[x]` Multiple items on desktop
- `[x]` Notepad already open (click Notepad on taskbar)

### CLI & Project
- `[x]` Minimal CLI: `locate --query` and `automate --query`
- `[x]` Environment config via `.env` (API key, model, etc.)
- `[x]** uv configuration**`
- `[x]` All artifacts saved to output directory
- `[x]` Comprehensive README with architecture and usage
- `[x]` Pushed to GitHub

---

## 2. CRITICAL GAPS — Must Fix Before Interview

### A. Deliverable: 3 Annotated Screenshots
**PDF Requirement:** "At least 3 annotated screenshots showing: Icon detected in top-left area, Icon detected in bottom-right area, Icon detected in center of screen"

- `[ ]` Generate screenshot with Notepad icon in **top-left** + detection overlay
- `[ ]` Generate screenshot with Notepad icon in **bottom-right** + detection overlay
- `[ ]` Generate screenshot with Notepad icon in **center** + detection overlay
- `[ ]` Add these to repo (e.g., `docs/screenshots/`)

> **This is a hard deliverable. Do not skip.**

### B. Broaden Search to Visual Icon (Not Just Label)
**PDF Requirement:** "The assignment is for a notepad, but it should work for any icon or button even if we don't have the exact image or text beforehand."

- `[ ]` Update LLM prompt: "Find the Notepad **icon** (the blue notepad graphic), not just the text label"
- `[ ]` Test with renamed shortcut (e.g., "Notes") — should still find it by visual
- `[ ]` Test with no label visible — should still work
- `[ ]` Document this flexibility in README

### C. Handle Multiple Matching Icons
**PDF Requirement:** "Handle cases where: Multiple matching icons exist" + Bonus: "Detect multiple desktop icons and select the correct one"

- `[ ]` When multiple Notepad icons exist, use LLM to select the correct one
- `[ ]` Return list of candidates and let LLM rank them
- `[ ]` Test with multiple Notepad shortcuts on desktop

---

## 3. Test Scenarios from PDF — Coverage Matrix

| # | Scenario | Status | Notes |
|---|----------|--------|-------|
| 1 | Icon at **top-left** | `[x]` | Tested manually, need screenshot |
| 2 | Icon at **bottom-right** | `[x]` | Tested manually, need screenshot |
| 3 | Icon at **center** | `[x]` | Tested manually, need screenshot |
| 4 | **Small** icon size (96px) | `[x]` | Tested |
| 5 | **Medium** icon size | `[x]` | Tested |
| 6 | **Large** icon size | `[ ]` | **NOT TESTED** |
| 7 | **Light** desktop theme | `[ ]` | **NOT TESTED** |
| 8 | **Dark** desktop theme | `[ ]` | **NOT TESTED** |
| 9 | **Solid color** background | `[?]` | Not explicitly tested |
| 10 | **Custom / busy** background | `[x]` | Tested with drawings |
| 11 | **Partially obscured** by windows | `[ ]` | **NOT TESTED** |
| 12 | **Multiple** matching icons | `[ ]` | **NOT TESTED** |
| 13 | Similar names (Notepad vs Notepad++) | `[ ]` | **NOT TESTED** |
| 14 | API unavailable / graceful degradation | `[ ]` | **NOT HANDLED** |
| 15 | Existing files in target directory | `[x]` | Cleared at start |
| 16 | Retry logic (3 attempts, 1s delay) | `[x]` | `max_retries=3, retry_delay=1.0` |
| 17 | Window title validation (Notepad launched) | `[x]` | Reviewer validates |

---

## 4. Interview Discussion Topics — Be Ready To Answer

### Approach & Design
- `[x]** Why coarse-to-fine grid over alternatives?**`
  - **Answer:** Inspired by ScreenSeekeR (ScreenSpot-Pro paper) — "strategically reducing the search area enhances accuracy." We use a cascaded search: full screen → grid cell → crop → finer grid → bbox refinement. This is more robust than single-shot detection.
- `[x]** Why LLM vision instead of template matching / OCR?**`
  - **Answer:** Template matching breaks with different backgrounds/themes/sizes. OCR only reads text, not icons. LLM vision understands visual semantics ("the blue notepad icon") and generalizes to arbitrary icons without retraining.

### Failure Cases
- `[x]** When would detection fail?**`
  - **Answer:** (1) Icon completely obscured by window, (2) Very busy background with similar colors, (3) Extreme icon sizes we haven't tested, (4) Multiple identical icons, (5) Network/API failure.
- `[ ]** How would you improve it?**`
  - **Answer ideas:** (1) Add OCR fallback for text labels, (2) Use local vision model (e.g., Qwen-VL) to reduce latency/cost, (3) Pre-train a small icon detector on synthetic data, (4) Add memory/cache of last known positions.

### Performance
- `[ ]** How long does detection take?**`
  - **Action:** Measure and document average time per detection.
  - **Answer so far:** ~4-6 LLM calls × 2-3s = ~10-15s per detection. Bbox refinement adds 2-3 more calls.
- `[ ]** Optimization strategies?**`
  - **Answer ideas:** (1) Use smaller/faster model for coarse rounds, (2) Cache screenshots, (3) Parallelize grid overlay generation, (4) Reduce image resolution for coarse rounds.

### Robustness
- `[ ]** Different Windows themes (light/dark)?**`
  - **Status:** Not tested. Need to test.
- `[ ]** Different icon view sizes?**`
  - **Status:** Tested small + medium, not large.
- `[ ]** Custom backgrounds (busy vs solid)?**`
  - **Status:** Tested busy, not solid.
- `[ ]** Multiple similar icons?**`
  - **Status:** Not tested. Need logic to disambiguate.
- `[ ]** Icons with similar names (Notepad vs Notepad++)?**`
  - **Status:** Not tested. Visual search should help here.

### Scaling
- `[x]** Detect any arbitrary desktop icon?**`
  - **Answer:** Yes, change the query text. But we should make this more explicit — the LLM should search for visual icon features, not just text.
- `[ ]** Work on different resolutions?**`
  - **Status:** Only tested on 1920×1080. Grid cell sizes are proportional but need verification.
- `[ ]** Alternative approaches with more time?**`
  - **Answer ideas:** (1) Pure OCR pipeline (no LLM, deterministic, offline), (2) Fine-tuned YOLO model for icon detection, (3) Hybrid: OCR for text labels + LLM for icons, (4) Use accessibility APIs as fallback (though assignment forbids this).

### Paper Reference
- `[x]` Read ScreenSpot-Pro paper (arXiv:2504.07981)
- `[x]` Key insight: "Strategically reducing the search area enhances accuracy"
- `[x]` Their method: ScreenSeekeR — cascaded search guided by planner
- `[x]` **Our approach is aligned:** We do cascaded search (grid → crop → finer grid → bbox), which matches the paper's finding. We can cite this in the interview.

---

## 5. Cleanup (Before or After Interview)

The project currently feels over-structured with too many small files and test files. It needs simplification.

### A. Reduce File Count & Complexity
- `[ ]` **Inspect the flow** — trace the actual execution path and remove unused abstractions
- `[ ]` **Merge small modules** where it makes sense (e.g., geometry + images)
- `[ ]` **Remove boilerplate** that was added "just in case"
- `[ ]` **Reduce test count** — keep only tests that validate actual behavior, not every utility function
- `[ ]` **Target:** Make the project look like a human wrote it, not 10 AI agents

### B. Simplify Project Structure

Current:
```
src/notepad_grounding/
  shared/
    llm.py
    images.py
    geometry.py
    automation.py
    reviewer.py
    schemas.py
    env.py
  flows/
    llm_visual_search/
      flow.py
    grid_ocr/
      ...
  cli.py
```

Target (simpler):
```
src/notepad_grounding/
  locate.py      # Core visual grounding logic
  automate.py    # Full automation loop
  llm.py         # LLM client + schemas
  utils.py       # Screenshots, image drawing, geometry
  desktop.py     # Mouse, keyboard, window helpers
  reviewer.py    # LLM reviewer
  cli.py
```

- `[ ]` Consolidate `images.py` + `geometry.py` + `env.py` into `utils.py`
- `[ ]` Consolidate `automation.py` into `desktop.py`
- `[ ]` Consolidate `llm.py` + `schemas.py` into `llm.py`
- `[ ]` Remove `shared/` and `flows/` subdirectories

### C. Code Quality
- `[ ]` Remove dead code (OCR flow if not used)
- `[ ]` Add inline comments explaining WHY, not WHAT
- `[ ]` Reduce test files from ~10 to ~3 focused test files
- `[ ]` Make imports and dependencies clearer

---

## 6. Future Projects (Post-Interview / Weekend)

### A. Split LLM and OCR into Separate Projects
The current repo has both LLM visual search and grid OCR. They should be separate repositories.

- `[ ]` **Project 1: `notepad-grounding`** — LLM-based visual search (current, simplified)
- `[ ]` **Project 2: `notepad-grounding-ocr`** — Pure OCR-based approach (no LLM, deterministic)
  - `[ ]` Use Tesseract / easyocr / pytesseract
  - `[ ]` Build a deterministic grid overlay
  - `[ ]` Match text labels to icon positions
  - `[ ]` No API calls needed, fully offline
  - `[ ]` Can be a good fallback or comparison benchmark

### B. OCR Flow Implementation (Weekend Task)
- `[ ]` Research best OCR library for Windows desktop icons
- `[ ]` Implement icon detection via OCR (text labels)
- `[ ]` Map OCR text positions to icon centers
- `[ ]` Test on same scenarios as LLM flow
- `[ ]` Compare accuracy and speed vs LLM flow

---

## 7. Quick Checklist Before Interview

- [ ] **3 annotated screenshots** generated and in repo
- [ ] **Large icon size** tested
- [ ] **Light theme** tested
- [ ] **Dark theme** tested
- [ ] **Partially obscured icon** tested
- [ ] **Multiple icons** tested
- [ ] **Renamed shortcut** tested (visual search, not label)
- [ ] **API degradation** handled (try/except around fetch)
- [ ] **Performance measured** and documented
- [ ] **Code cleaned up** and readable
- [ ] **README accurate** and up-to-date
- [ ] **Repo pushed** to GitHub
- [ ] (Optional) **OCR version** working as a backup demo

---

## 8. Notes & Decisions Log

| Date | Decision | Context |
|------|----------|---------|
| 2025-05-13 | Use Alt+F4 + Escape for closing wrong apps | Alt+F4 closes apps; Escape dismisses shutdown dialog on desktop |
| 2025-05-13 | Use `previous_response_id` for LLM self-correction in bbox refinement | Keeps conversation context, avoids re-uploading images |
| 2025-05-13 | Use `instructor` library for structured LLM outputs | Eliminates JSON parsing errors, guarantees Pydantic schema |
| 2025-05-13 | Coarse-to-fine grid matches ScreenSeekeR approach | ScreenSpot-Pro paper says "reducing search area enhances accuracy" |
| 2025-05-13 | Close Notepad after each post, re-open for next | Matches PDF workflow exactly |

---

*Last updated: 2025-05-13*
