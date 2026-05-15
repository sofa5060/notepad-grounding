# notepad-grounding

Vision-based Windows desktop automation that locates the Notepad icon using LLM vision and runs a full automation workflow.

## Run

```powershell
uv run notepad-grounding
```

## How It Works

1. Capture a full desktop screenshot
2. Send to LLM with structured output — model returns center point and bounding box
3. Confidence gating (0.7 minimum) with up to 3 retries on low confidence or API errors
4. Double-click the icon center
5. Type post content, save via Ctrl+Shift+S, close via Ctrl+Shift+W
6. Repeat for 10 posts

## Error Handling

| Scenario | Handling |
|----------|----------|
| Low confidence / API error | Retry up to 3 times, 1s delay |
| API unavailable (posts) | Retry up to 3 times, 5s delay; fallback to dummy data |
| Window fails to open | Timeout after 8 seconds |
| Close fails | Alt+F4 fallback |

## Configuration

Copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=sk-your-real-key
OPENAI_MODEL=gpt-5.4
```

## Requirements

- Windows 10/11, 1920x1080, 100% scale
- Python 3.11+
- `uv`
- Desktop shortcut named `Notepad`

## Project Structure

```text
src/notepad_grounding/
  main.py         # entry point and orchestration
  llm.py          # LLM vision icon locating with structured output
  automation.py   # desktop automation (click, type, save, close)
  api.py          # JSONPlaceholder post fetching with dummy fallback
```

## Output

Per-run artifacts saved under `output/notepad_grounding/<NN>/`:

- `screenshot.png` — original screenshot
- `response.json` — structured LLM response
- `annotated.png` — screenshot with bounding box, crosshair, and confidence label

## Development

```bash
uv run pytest -v
```
