# notepad-grounding

Vision-based Windows desktop automation for locating and using the Notepad shortcut.

Running the project captures the desktop, visually grounds the requested icon, opens Notepad, writes posts from JSONPlaceholder, saves them to `Desktop/tjm-project`, and closes Notepad between posts. The default target is `Notepad`.

## Run

```powershell
uv run notepad-grounding
```

Override the target query:

```powershell
uv run notepad-grounding --query Notepad
```

Debug only the visual grounding step:

```powershell
uv run notepad-grounding locate --query Notepad
```

## How It Works

The system keeps screen-coordinate math deterministic. The LLM only chooses from labeled visual options.

1. Capture a desktop screenshot.
2. Draw a coarse labeled grid over the current image.
3. Ask the vision model which cell contains the requested icon or shortcut.
4. Ask the target crop reviewer whether the selected crop actually contains the target icon, app visual, or matching label.
5. If rejected, continue the same chooser conversation and forbid rejected cells.
6. Repeat coarse-to-fine cropping until the target crop is small enough.
7. Draw a 7x7 row/column click grid, then a 5x5 refinement grid.
8. Map the final selected grid-cell center back to full-screen coordinates in code.
9. If click-grid precision fails, run the bbox reviewer fallback and click the reviewed bbox center.
10. During automation, use the desktop state reviewer after click/type/save/close actions.

## Validation Layers

| Reviewer | When it runs | What it checks |
| --- | --- | --- |
| Target crop reviewer | After selected coarse/click grid crops | The crop contains the requested icon/app visual or label |
| Bbox reviewer | Bbox fallback only | The red bbox is tight and centered on the clickable icon graphic |
| Desktop state reviewer | After automation actions | The desktop state matches the expected result |

## Configuration

Copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-5.4
OPENAI_REVIEWER_MODEL=
```

`OPENAI_REVIEWER_MODEL` is optional. If unset, reviewers use legacy `OPENAI_JUDGE_MODEL` when present, then `OPENAI_MODEL`, then the default `gpt-5.4`.

## Requirements

- Windows 10/11 for live capture and automation
- Python 3.11+
- `uv`
- Desktop shortcut named `Notepad`
- Target display: `1920x1080`, `100%` scale

Avoid Parallels shared folders for final Windows testing. Clone into a normal Windows path such as:

```text
C:\Users\<user>\Desktop\notepad-grounding
```

## Output

Default automation output:

```text
output/automation/<timestamp>/
```

Each nested locate run saves artifacts under:

```text
output/automation/<timestamp>/locate/<timestamp>/
```

Important locate artifacts:

- `00-source.png` — original screenshot
- `01-grid.png`, `01-selected.png` — coarse grid and selected cell
- `01-target-review-crop-attempt-1.png` — selected crop sent to the target reviewer
- `01-target-review-result-attempt-1.json` — structured target reviewer result
- `final-crop.png` — final cropped region
- `click-points-01.png` — coarse final row/column grid
- `click-points-02.png` — fine row/column grid
- `click-point-final.png` — final selected click point on the final crop
- `click-point-full.png` — final selected click point on the full screenshot
- `click-point-final.json` — final click result
- `bbox-*.json/png`, `final-detection.png` — written when bbox fallback runs
- `result.json` — final structured locate result

## Project Structure

```text
src/notepad_grounding/
  cli.py          # command line interface
  flow.py         # central product flow orchestrator
  desktop_interactions.py # high-level desktop interaction/review steps
  locate.py       # visual grounding pipeline
  vision.py       # OpenAI vision chooser client and parsers
  reviewers.py    # target, bbox, and desktop reviewers
  prompts.py      # prompt builders
  models.py       # dataclasses and Pydantic response models
  click_grid.py   # row/column click-grid overlays and mapping
  geometry.py     # boxes and coarse grid math
  images.py       # image drawing/cropping helpers
  capture.py      # screenshot capture
  desktop.py      # mouse/keyboard/window helpers
  api.py          # JSONPlaceholder fetch
src/grid_ocr/
  ocr.py          # preserved side OCR experiment, not the main runtime path
  grounding.py
  annotate.py
```

## Development

```bash
uv run pytest -v
uv run notepad-grounding --help
uv run notepad-grounding locate --help
```

## Notes

- The main command runs the full automation; `locate` is a debug command.
- OCR is preserved as a separate side package under `src/grid_ocr`.
- Bbox precision remains as fallback while the click-grid method is tested on Windows.
- The implementation does not use Windows desktop item APIs, shell list view APIs, or accessibility APIs to obtain icon positions.
