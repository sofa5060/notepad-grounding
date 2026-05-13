# Notepad Grounding — Project Roadmap

> Updated working document for the take-home interview project.

## Current Status

The core `locate` command works. Recent focus has been on reliability fixes for Windows testing.

## Status Legend

- `[x]` Done
- `[~]` In progress
- `[ ]` Not started
- `[?]` Needs decision

---

## Completed ✅

### Core Features
- `[x]` Primary `llm-visual` locate flow
- `[x]` Coarse-to-fine grid search
- `[x]` Final crop-local bbox detection
- `[x]` Bbox self-review using `previous_response_id`
- `[x]` Full-screen bbox overlay (`full-detection.png`)
- `[x]` CLI: `locate --query` and `automate --query`
- `[x]` Artifacts saved to output directories
- `[x]` `.env` support
- `[x]` Configurable model via `OPENAI_MODEL`
- `[x]` API fallback to dummy data when JSONPlaceholder is down
- `[x]` Timing measurements printed to console
- `[x]` Alt+F4 + Escape recovery for wrong app clicks

### Reliability Fixes
- `[x]` **Auto-correct invalid cell_id**: When LLM returns wrong round (e.g., `R2-*` instead of `R1-*`), catch it and retry with correction prompt via `previous_response_id`

### Manual Testing Done
- `[x]` Bottom-right position + small icons + busy background
- `[x]` Center position + medium icons + busy background  
- `[x]` Top-left position + small icons + busy background
- `[x]` Multiple desktop positions tested
- `[x]` Non-solid wallpaper (busy/cluttered background)

---

## Still To Do 🔧

### 1. Fix Center Position Reliability

**Problem:** Image 2 (center position) showed the LLM returning invalid `cell_id 'R2-2-2'` when valid cells were `R1-*`. The auto-correct fix was just pushed but needs testing on Windows.

**Action:**
- `[ ]` Pull latest code on Windows
- `[ ]` Retest center position with medium icons
- `[ ]` Confirm auto-correct works and produces clean `full-detection.png`

### 2. Generate Final Screenshot Artifacts

The PDF requires **3 annotated screenshots in the repo**. We have tested manually but need to save the actual `full-detection.png` files.

| # | Position | Icon Size | Status |
|---|----------|-----------|--------|
| 1 | Bottom-right | Small | `[x]` Tested manually |
| 2 | Center | Medium | `[~]` Needs retest with fix |
| 3 | Top-left | Small | `[x]` Tested manually |

**Action:**
- `[ ]` After center fix is confirmed, copy 3 `full-detection.png` files to `docs/screenshots/`
- `[ ]` Add notes explaining each scenario

### 3. Test Icon Size Variations

| Size | Status | Notes |
|------|--------|-------|
| Small (96px) | `[x]` Tested | Works |
| Medium | `[x]` Tested | Works |
| Large | `[ ]` Not tested | Quick test if time allows |

### 4. Test Theme Variations

| Theme | Status |
|-------|--------|
| Light | `[x]` Implicitly tested (all screenshots are light theme) |
| Dark | `[ ]` Not tested | Optional bonus |

### 5. Full Automation End-to-End Test

**Action:**
- `[ ]` Run `uv run notepad-grounding automate --query Notepad` once on Windows
- `[ ]` Verify all 10 posts are saved to `Desktop/tjm-project`
- `[ ]` Check success rate (should be high with the cell_id fix)

### 6. Optional Robustness Tests (If Time)

- `[ ]` Icon partially obscured by window
- `[ ]` Multiple Notepad shortcuts on desktop
- `[ ]` Renamed shortcut (e.g., "Notes" instead of "Notepad")
- `[ ]` Different resolution (only 1920x1080 assumed)

---

## Coverage Matrix

| Scenario | Status | Notes |
|----------|--------|-------|
| Icon at top-left | `[x]` | Tested, needs artifact saved |
| Icon at center | `[~]` | Tested but failed, fix pushed, needs retest |
| Icon at bottom-right | `[x]` | Tested, needs artifact saved |
| Small icon size | `[x]` | ✅ |
| Medium icon size | `[x]` | ✅ |
| Large icon size | `[ ]` | Optional |
| Light theme | `[x]` | Implicitly tested |
| Dark theme | `[ ]` | Optional bonus |
| Busy/custom background | `[x]` | ✅ All tests used busy background |
| Partially obscured icon | `[ ]` | Optional |
| Multiple matching icons | `[ ]` | Optional |
| Renamed shortcut | `[ ]` | Optional |
| API unavailable fallback | `[x]` | Code ready |
| Retry logic (3 attempts, 1s delay) | `[x]` | ✅ |
| Wrong app recovery | `[x]` | ✅ Alt+F4 + Escape |

---

## Quick Checklist Before Interview

- [ ] Retest center position with latest fix
- [ ] Generate 3 annotated screenshots and add to repo
- [ ] Run full automation at least once
- [ ] Measure average timing (for interview discussion)
- [ ] `uv run pytest -v` passes
- [ ] `uv run notepad-grounding locate --query Notepad --out-dir output` works
- [ ] Repo pushed to GitHub
- [ ] README is up to date

---

## Interview Discussion Topics (Prepare Answers)

### Architecture
- `[x]` Why coarse-to-fine grid? → Matches ScreenSpot-Pro paper
- `[x]` Why LLM vision? → Brittle across wallpapers/themes/sizes
- `[x]` Code owns all screen-coordinate math

### Robustness
- `[x]` Auto-correct invalid cell_id via conversation continuity
- `[x]` 3 retries with 1s delay
- `[ ]` Judge layer (not yet implemented — simpler approach works)
- `[ ]` Performance: measure locate time on Windows

### Known Limitations
- `[x]` Icon fully hidden → would fail
- `[x]` Unsupported resolution/scaling → not tested
- `[x]` Multiple identical shortcuts → not handled

---

*Last updated: 2026-05-13*
