# Assignment Analysis — Vision-Based Desktop Automation

## Source Document
PDF: `/Users/sofa/Downloads/Take home project.pdf`

## Critical Findings (Changed Our Understanding)

### The Real Goal Is NOT "Find Notepad"

> *"We are looking for the most flexible implementation, where we can bypass things like unexpected pop-ups without knowing what they look like in advance. The assignment is for a notepad, but it should work for any icon or button even if we don't have the exact image or text beforehand."*

**This means template matching is disqualified.** Any approach that requires:
- A reference image of the Notepad icon ❌
- Hardcoded knowledge of what Notepad looks like ❌
- Training data specific to one icon ❌

...is the wrong approach. The interviewer will test with arbitrary icons/buttons.

### Why OCR-First Is Actually Correct

The assignment explicitly references **ScreenSpot-Pro** (arXiv:2504.07981) and says:
> *"An exact implementation of the paper is recommended, but other approaches are also acceptable."*

ScreenSpot-Pro's approach:
1. Detect UI elements via vision
2. Generate candidate regions
3. Score/rank candidates
4. Select the best match for a text query

This is **OCR → candidates → scoring → selection**. Our architectural direction is correct.

### What "General Visual Grounding" Means

The system must handle query-based detection:
- Query: `"Notepad"` → find Notepad icon
- Query: `"Save"` → find Save button in a dialog
- Query: `"OK"` → find OK button on an unexpected popup

This requires:
1. **Text detection** (OCR) — read what's visible
2. **Spatial reasoning** — know text belongs to an icon above it
3. **Fuzzy matching** — match noisy OCR to the query
4. **Visual verification** — confirm the region actually contains an icon

### Why Our Current OCR Fails

Desktop icon labels have a specific visual signature that standard OCR doesn't handle well:
- **White text with dark drop shadow** (designed for human eyes on any wallpaper)
- Small font (~11px)
- On textured/colored wallpaper
- With an icon image directly above

Current preprocessing (`grayscale + contrast + upscale`) doesn't amplify this signature.

### The Interview Test Scenario

> *"During the interview, we will: Move the Notepad icon to different desktop positions. Run your executable. Verify it correctly locates and clicks the icon. Discuss what happens if detection fails."*

This means:
- The icon WILL move between runs
- We need 3+ annotated screenshots proving detection at different positions
- We need to explain failure modes intelligently

## Proven Architecture (From Paper + Assignment)

```
Screenshot
  ↓
[Preprocessing: amplify white text + shadow signature]
  ↓
[Multi-Engine OCR: Windows OCR + Tesseract (full + tiled)]
  ↓
[Merge + Deduplicate all detections]
  ↓
[Desktop Inventory Prior: list ~/Desktop filenames]
  ↓
[Grid Inference: compute desktop icon grid from detected labels]
  ↓
[Candidate Generation: icon box above each label, plus inferred grid cells]
  ↓
[Scoring: label_match(query) + inventory_prior + geometry + visual_consistency]
  ↓
[Selection: return center of best candidate]
  ↓
[Optional MLLM Fallback: when top candidates are ambiguous]
```

## Key Implementation Priorities

### 1. Text-Specific Preprocessing
Instead of generic contrast enhancement, create preprocessing that specifically amplifies the "white text with dark shadow" signature of desktop icon labels.

### 2. Desktop Inventory Prior
Query `C:\Users\<user>\Desktop` for actual filenames. Use these as:
- Expected label set ("Notepad" should exist)
- Fuzzy match targets for noisy OCR ("Note pad" → "Notepad")
- False positive rejection ("CleanShot" is real, random OCR fragments are not)

### 3. Grid-Aware Candidate Generation
Windows desktop icons follow a regular grid (when auto-arrange is on). From detected labels, compute grid spacing. Generate candidates for ALL grid cells, not just where OCR worked. This handles icons with unreadable labels.

### 4. Robust Scoring
Combine multiple signals:
- `label_score`: fuzzy match between query and OCR text
- `inventory_score`: does OCR text match a known desktop item?
- `geometry_score`: does the region look like an icon+label?
- `grid_score`: is the position on the desktop grid?
- `visual_score`: does the icon region have icon-like visual properties?

### 5. MLLM Fallback (Optional)
When top 2 candidates have similar scores, send crops to GPT-4o/Claude for disambiguation. The MLLM chooses between candidates; coordinates remain deterministic.

## Discussion Topics to Prepare

The assignment lists explicit discussion topics. We should design the system to have good answers for each:

| Topic | Our Answer |
|-------|-----------|
| **Icon detection approach** | OCR-first with desktop inventory priors and grid inference |
| **Failure cases** | Very busy wallpaper, extremely small icons, icons behind windows, non-English labels |
| **Performance** | ~2-3 seconds per detection (OCR bottleneck) |
| **Different themes** | Text preprocessing adapts to light/dark text; grid inference is theme-independent |
| **Different icon sizes** | Multi-scale template matching within candidate regions |
| **Multiple similar icons** | Scoring ranks by label match + visual consistency |
| **Similar names** | "Notepad++" gets lower score than exact "Notepad" match |
| **Scaling to any icon** | Query-based: same pipeline works for any text label |
| **Scaling to any resolution** | Grid inference adapts to any screen size |
| **What would you do differently** | Train a small YOLO/CNN on synthetic desktop icons for faster detection |
| **MLLM for OCR** | Instead of traditional OCR, use GPT-4o/Claude to read text from screenshot crops. More robust for small/shadowed text but slower and API-dependent |

## Current Status vs. Target

| Component | Status | Target |
|-----------|--------|--------|
| Screenshot capture | ✅ Done | ✅ Done |
| Basic OCR (Tesseract + Windows) | ✅ Done | ✅ Done |
| Word grouping | ✅ Done | ✅ Done |
| Candidate inference (icon above label) | ✅ Done | ✅ Done |
| **Text-specific preprocessing** | ❌ Rejected — destroyed text for Windows OCR | **Replaced by raw image** |
| **Desktop inventory prior** | ❌ Missing | **Priority 1** |
| **Grid inference** | ❌ Missing | **Priority 2** |
| Scoring | ❌ Empty | Priority 4 |
| Automation (click, type, save) | ❌ Empty | Priority 5 |
| MLLM verifier | ❌ Empty | Optional |
