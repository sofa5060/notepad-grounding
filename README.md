# notepad-grounding

Vision-based Windows desktop icon grounding and automation.

Given a query like `"Notepad"`, the system uses an LLM to visually locate the desktop icon, click it, type content, save files, and close the app — with validation at every step.

## How It Works

### Architecture Overview

The system has three layers:

1. **Visual Grounding** — finds the icon on the desktop using LLM vision
2. **Automation** — clicks, types, saves, closes using `pyautogui`
3. **Reviewer** — validates every action by sending screenshots back to the LLM

### Step-by-Step Flow

#### 1. Visual Grounding: `locate` command

```powershell
uv run notepad-grounding locate --query Notepad
```

The grounding uses a **coarse-to-fine grid search**:

1. **Screenshot** — captures the desktop (or loads an image)
2. **Coarse grid** — draws a 3×4 grid over the screenshot, labels each cell (`R1-1-1`, `R1-1-2`, etc.)
3. **LLM picks a cell** — sends the grid image to the LLM, which returns the cell ID containing the icon (no pixel coordinates!)
4. **Grid judge validates the crop** — crops the selected cell with padding and asks a second vision call whether that crop actually contains the target icon or label
5. **Retry if rejected** — if the judge rejects the crop, the chooser is told which cell was wrong and must pick a different cell in the same conversation
6. **Crop** — once accepted, crops the selected region with padding
7. **Repeat** — draws a finer 3×3 grid on the crop, LLM picks again
8. **Marked click-point refinement** — once the crop is small enough:
   - Code draws a 7×7 yellow grid over the crop
   - Row numbers are drawn on the left and column numbers above the image, outside the screenshot content
   - LLM chooses the row/column cell whose center should be clicked
   - A judge checks the selected cell crop before accepting it
   - Code crops around that cell and draws a finer 5×5 row/column grid
   - LLM chooses the final cell ID
9. **Center calculation** — code maps the selected point ID back to screen coordinates
10. **Bbox fallback** — if marked-point selection fails, the existing bbox refinement flow still runs automatically

All artifacts (grids, crops, detections) are saved under `output/llm_visual_search/<timestamp>/`.

#### 2. Full Automation: `automate` command

```powershell
uv run notepad-grounding automate --query Notepad
```

For each of the first 10 posts from JSONPlaceholder:

1. **Ground** — run visual search to get icon center `(x, y)`
2. **Double-click** — `pyautogui.doubleClick(x, y)` to launch Notepad
3. **Review** — take screenshot, ask LLM: *"Is Notepad open?"*
   - If wrong app opened (e.g., Steam): close it with `Alt+F4`, retry
4. **Type** — type the post content: `Title: {title}\n\n{body}`
5. **Review** — *"Does the text look correct?"*
6. **Save** — `Ctrl+Shift+S` → type full path `C:\Users\...\Desktop\tjm-project\post_{id}.txt` → Enter
7. **Review** — *"Did save succeed? Any pop-ups?"*
   - If "Replace file?" dialog: click Yes/Replace
8. **Close** — `Ctrl+Shift+W` to close Notepad
9. **Review** — *"Is Notepad closed? Desktop visible?"*
10. **Repeat** — fresh screenshot → ground again → next post

If any step fails, the system retries up to 3 times with a 1-second delay.

### Validation Layers

The system uses three separate validation layers:

| Layer | When it runs | What it checks |
|-------|--------------|----------------|
| Grid judge | After every selected grid cell | The selected crop contains the requested icon/app visual or matching label |
| Marked click grid | At final precision | The LLM chooses labeled row/column cells instead of raw pixel coordinates |
| Bbox reviewer | Fallback after point selection failure | The red rectangle is tight around only the icon graphic |
| Automation reviewer | After click/type/save/close actions | The desktop state matches the expected outcome |

If the grid judge rejects a selected cell, the chooser is corrected in the same conversation and must choose another cell. If all judge retries fail, `locate` stops with saved judge crop/result artifacts instead of continuing toward a likely wrong click.

### The Automation Reviewer Pattern

After **every action**, a screenshot is sent to the LLM reviewer:

```
Action: "Double-clicked Notepad icon at (850, 420)"
Expected: "Notepad window is open and active"
→ LLM returns: {status, action_needed, rationale}
```

| Status | Meaning | Recovery |
|--------|---------|----------|
| `success` | Everything is correct | Continue |
| `wrong_app` | Wrong window opened | Close with `Alt+F4`, retry from grounding |
| `pop_up` | Unexpected dialog | Handle it (click Replace, Yes, etc.) |
| `error` | Something else is wrong | Wait longer or retry |
| `retry` | Not sure yet | Wait and re-check |

This makes the system robust against:
- Wrong icon clicks
- Replace file dialogs
- Windows that don't open
- Windows that don't close

### Structured LLM Outputs

Instead of parsing raw JSON text, the system uses the **`instructor`** library with Pydantic models:

```python
from pydantic import BaseModel

class ReviewResultModel(BaseModel):
    status: str           # success, wrong_app, pop_up, error, retry
    action_needed: str    # what to do next
    rationale: str        # explanation

# Guaranteed structured output — no parsing errors
result = client.chat.completions.create(
    model="gpt-4o",
    response_model=ReviewResultModel,
    messages=[...],
)
```

## Setup

### Requirements

- Windows 10/11 (for live capture and automation)
- Python 3.11+
- `uv` for dependency management
- `OPENAI_API_KEY` in `.env`

### Install

```powershell
git clone <repo-url>
cd notepad-grounding
uv sync
```

### Configuration

```powershell
copy .env.example .env
```

Edit `.env`:

```text
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-4o
OPENAI_JUDGE_MODEL=
```

Default model is `gpt-5.4`. Change `OPENAI_MODEL` to use a different one. Set `OPENAI_JUDGE_MODEL` only if you want the grid judge to use a different OpenAI vision model than the chooser.

### Before Running

1. Create a desktop shortcut named **"Notepad"**
2. Make sure the desktop is visible (no windows covering it)
3. The target save directory `Desktop/tjm-project` is created and cleared automatically

## CLI Commands

### `locate` — Find an icon

```powershell
# Live capture + ground
uv run notepad-grounding locate --query Notepad

# Replay a saved screenshot
uv run notepad-grounding locate --query Notepad --image screenshots/desktop.png
```

**Output:**
```
flow=llm-visual
output_dir=output/llm_visual_search/20250115-143022
result=output/llm_visual_search/20250115-143022/result.json
found=true
center=850,420
```

Artifacts saved:
- `00-source.png` — original screenshot
- `01-grid.png`, `01-selected.png` — round 1 grid
- `01-judge-crop-attempt-1.png` — selected crop sent to the grid judge
- `01-judge-result-attempt-1.json` — structured judge verdict
- `02-grid.png`, `02-selected.png` — round 2 grid
- `final-crop.png` — final cropped region
- `click-points-01.png` — coarse final row/column grid with labels outside the image
- `click-points-01-result.json` — selected coarse grid cell
- `click-grid-01-judge-*.json/png` — judge artifacts for the selected coarse cell
- `click-points-02-crop.png` — refinement crop around the accepted coarse cell
- `click-points-02.png` — fine row/column grid with labels outside the image
- `click-points-02-result.json` — selected fine grid cell and mapped coordinates
- `click-grid-02-judge-*.json/png` — judge artifacts for the selected fine cell
- `click-point-final.png` — final selected click location on the final crop
- `click-point-full.png` — final selected click location on the full screenshot
- `click-point-final.json` — final click result
- `click-point-error.json` — written only if point selection fails and bbox fallback starts
- `bbox-initial-result.json` — initial bbox model output
- `bbox-review-01.png` — red bbox image sent back for review
- `bbox-review-01-result.json` — bbox reviewer response for that iteration
- `bbox-final-result.json` — final bbox used for center/click calculation
- `final-detection.png` — detection with red bbox, written when bbox fallback is used
- `result.json` — full result with center coordinates

### `automate` — Full workflow

```powershell
# Default: 10 posts, 3 retries
uv run notepad-grounding automate --query Notepad
```

**Output:**
```
INFO: Starting automation for query='Notepad'
INFO: Fetching 10 posts from JSONPlaceholder...
INFO: [post_1.txt] Attempt 1/3: grounding icon...
INFO: [post_1.txt] Icon found at (850, 420)
INFO: [post_1.txt] Double-clicked at (850, 420)
INFO: [REVIEW] Double-clicked... | Expected: Notepad window is open
INFO: [REVIEW] status=success action_needed=proceed rationale=Notepad is active
INFO: [post_1.txt] Reviewer confirmed: Notepad is open
...
flow=automate
output_dir=output/automation/20250115-143022
result=output/automation/20250115-143022/result.json
total_posts=10
succeeded=10
failed=0
```

## Project Structure

```text
src/notepad_grounding/
  main.py                          # CLI entry point
  shared/
    automation.py                  # Mouse/keyboard helpers (pyautogui)
    api.py                         # JSONPlaceholder fetch
    capture.py                     # Screenshot capture (mss)
    click_points.py                # Marked click-grid overlays and mapping
    env.py                         # .env file loader
    geometry.py                    # Grid cell math, box operations
    images.py                      # Image drawing (grid, bbox)
    llm.py                         # OpenAI client + structured outputs
    reviewer.py                    # LLM state reviewer
    schemas.py                     # Pydantic models for LLM outputs
  flows/
    llm_visual_search/
      flow.py                      # Coarse-to-fine grounding + marked-point/bbox final precision
    automation/
      runner.py                    # Full Notepad automation loop
    grid_ocr/                      # Fallback OCR flow (deprecated)
```

## Key Design Decisions

### 1. No Pixel Coordinates from LLM
The LLM picks **labeled grid cells** during coarse search and **labeled click points** during final precision. Code owns all screen-coordinate math. This is deterministic and reliable.

### 2. Marked Click-Grid Precision with Bbox Fallback
The final click target is selected from a labeled row/column grid whose labels sit outside the image content, then mapped to screen coordinates in code. The older bbox refinement path remains available as fallback while the grid method is tested.

### 3. Reviewer Validates Every Action
A separate LLM call after every action catches errors early:
- Wrong app opened → close and retry
- Pop-up appeared → handle it
- Text not typed → retry

### 4. Structured Outputs with Pydantic
Using `instructor` library guarantees the LLM returns valid JSON matching a Pydantic schema. No manual parsing, no JSON errors.

### 5. macOS Parallels Compatibility
When running on macOS controlling a Windows VM:
- `Ctrl` is mapped to `Command` for keyboard shortcuts
- `Alt+F4` closes the active window
- `Ctrl+Shift+S` opens Save As dialog
- `Ctrl+Shift+W` closes Notepad tab

## Development

```bash
# Run tests
uv run pytest -v

# Check CLI help
uv run notepad-grounding --help
uv run notepad-grounding locate --help
uv run notepad-grounding automate --help
```

## Notes

- **Resolution:** Designed for `1920×1080` at `100%` scale. Other resolutions may need tuning.
- **Icon names:** The query should match the visible text label (e.g., `"Notepad"`, not `"notepad"`)
- **Cost:** Each automation run makes ~40-60 LLM calls (grounding rounds + reviews). Monitor your OpenAI usage.
- **Speed:** Each post takes ~15-30 seconds depending on LLM response time.

## License

Take-home project.
