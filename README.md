# notepad-grounding

Windows desktop grounding proof for a take-home project.

Current goal:

```text
Given a desktop screenshot and a query such as "Notepad",
visually locate the target icon and return a deterministic click center.
```

The default flow is now **LLM visual search**. OCR remains in the repo as a fallback experiment, but it is no longer the main path.

## Runtime

Final testing must run inside Windows 10/11, ideally the Windows VM used for the interview.

Recommended setup:

- Windows 10/11
- Resolution: `1920 x 1080`
- Display scale: `100%`
- A desktop shortcut named `Notepad`
- `OPENAI_API_KEY` set for the LLM visual flow
- Clone into a normal Windows folder, for example:

```text
C:\Users\<user>\Desktop\notepad-grounding
```

Avoid Parallels shared folders such as `C:\Mac\Home\...` for final testing.

## Setup

```powershell
uv sync
```

Create a local `.env` file from the example:

```powershell
copy .env.example .env
```

Then edit `.env`:

```text
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-5.4
```

The default LLM is `gpt-5.4`. You can change `OPENAI_MODEL` later if we decide to test a stronger or cheaper vision model.

## Default Locate Flow

Run inside Windows with the desktop visible:

```powershell
uv run notepad-grounding locate --query Notepad --out-dir output
```

Replay an existing screenshot:

```powershell
uv run notepad-grounding locate --query Notepad --image output/screen.png --out-dir output
```

The default `llm-visual` flow:

1. Captures or loads the screenshot.
2. Draws a coarse grid over the image.
3. Sends the grid image to the LLM.
4. The LLM returns a grid cell ID, not pixel coordinates.
5. The code crops that selected region from the original screenshot.
6. The process repeats with finer crops.
7. The final click center is computed by code from the selected cell box.

Outputs are grouped by flow and timestamp:

```text
output/llm_visual_search/<timestamp>/
  00-source.png
  01-grid.png
  01-selected.png
  02-grid.png
  02-selected.png
  03-grid.png
  03-selected.png
  result.json
```

## Fallback OCR Flow

The previous OCR experiment is preserved:

```powershell
uv run notepad-grounding locate --query Notepad --flow grid-ocr --out-dir output
```

Use it only for comparison or fallback. It runs OCR over overlapping tiles and saves grid/OCR/candidate debug images under `output/grid_ocr/`.

## Project Layout

```text
src/notepad_grounding/
  main.py
  shared/
    capture.py
    geometry.py
    images.py
    llm.py
  flows/
    llm_visual_search/
      flow.py
    grid_ocr/
      annotate.py
      grounding.py
      ocr.py
```

## Development Checks

```bash
uv run pytest -v
uv run notepad-grounding locate --help
```

## Next Milestones

1. Validate `llm-visual` on Windows with the icon in top-left, center, and bottom-right positions.
2. Inspect saved grid/crop images and tune the number of rounds or crop padding if needed.
3. Add the Notepad launch/save automation only after visual location is reliable.
