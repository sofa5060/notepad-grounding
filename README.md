# notepad-grounding

Vision-based Windows desktop automation that locates the Notepad icon using LLM vision and runs a full automation workflow.

## Setup

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-5.4
```

Then install dependencies:

```powershell
uv sync
```

## Run

```powershell
uv run notepad-grounding
```

## How It Works

1. Capture a full desktop screenshot
2. Send to LLM with structured output — model returns center point and bounding box
3. Confidence gating (0.85 minimum) with up to 3 retries; run is skipped if all attempts fail
4. Capture and save a fresh pre-click screenshot, then double-click the icon center
5. Wait for a window change and capture the post-click screenshot
6. Ask the LLM whether the expected app opened using the before/after image pair
7. Continue only after a positive structured verdict, then type, save, and close
8. Repeat for 10 posts

## Error Handling

| Scenario | Handling |
|----------|----------|
| Low confidence / locate API error | Retry up to 3 times; skip post if all attempts fail |
| API unavailable (posts) | Retry up to 3 times, 5s delay; fallback to dummy data |
| Window fails to open | Timeout after 8 seconds |
| App verification fails or returns false | Close with Alt+F4, wait up to 5 seconds, then retry |
| Unverified app fails to close | Abort the run before typing or clicking again |
| Normal Notepad close fails | Alt+F4 fallback |

## Requirements

- Windows 10/11, 1920x1080, 100% scale
- Python 3.11+
- `uv`
- Desktop shortcut named `Notepad`

## Project Structure

```text
src/notepad_grounding/
  main.py         # entry point and orchestration
  llm.py          # LLM vision locating and app-open verification
  automation.py   # desktop automation (click, type, save, close)
  api.py          # JSONPlaceholder post fetching with dummy fallback
```

## Output

Per-attempt artifacts are saved under `output/notepad_grounding/post_<NN>/run_<NN>/`:

- `screenshot.png` — original screenshot
- `response.json` — structured LLM response
- `annotated.png` — screenshot with bounding box, crosshair, and confidence label
- `open-before.png` — desktop immediately before the double-click
- `open-after.png` — desktop after the opened window settles
- `open-verification.json` — structured `opened_expected_app` verdict when parsing succeeds

## Development

```bash
uv run pytest -v
```
