# AGENTS.md

## Project

This repository is for a take-home interview project: **Vision-Based Desktop Automation with Dynamic Icon Grounding**.

The current milestone is a query-based `locate` command that visually finds a desktop icon and returns a deterministic center coordinate.

## Tooling

- Use `uv` for Python dependency management and running scripts.
- Use `bun` instead of `npm` if JavaScript tooling is introduced.
- Final runtime testing must happen inside Windows, not macOS.

## Runtime Assumptions

- Target OS: Windows 10/11.
- Target display: `1920x1080`, `100%` scale.
- The user will create a desktop shortcut named `Notepad`.
- `OPENAI_API_KEY` is required for the default LLM visual flow.
- `.env` is supported and ignored by git. Use `.env.example` as the template.
- Default model: `gpt-5.4`, configurable with `OPENAI_MODEL`.
- Clone/run from a normal Windows path such as:

```text
C:\Users\<user>\Desktop\notepad-grounding
```

Avoid Parallels shared folders such as `C:\Mac\Home\...` for final testing.

## Flow Structure

Keep flows isolated:

```text
src/notepad_grounding/
  shared/
  flows/
    llm_visual_search/
    grid_ocr/
```

Tests should mirror flow ownership:

```text
tests/
  shared/
  flows/
    llm_visual_search/
```

## Default Implementation Direction

Primary command:

```text
uv run notepad-grounding locate --query Notepad --out-dir output
```

Default flow: `llm-visual`.

The LLM visual flow:

1. Capture or load a screenshot.
2. Draw a labeled grid over the current image/crop.
3. Ask the LLM which grid cell contains the target.
4. Require the LLM to return a cell ID, not coordinates.
5. Crop the selected cell from the original screenshot with padding.
6. Repeat for a few rounds.
7. Once the crop is small enough, ask the LLM for a crop-local `icon_bbox` around the icon graphic.
8. Map the crop-local bbox back to screen coordinates in deterministic code.
9. Compute the final click center from the icon bbox center.
10. Save every grid/crop/result artifact.

The key rule: the LLM may choose among labeled cells and may return crop-local boxes during the final precision step; code owns all screen-coordinate math.

## Fallback Flow

`grid-ocr` is preserved as a fallback experiment:

```text
uv run notepad-grounding locate --query Notepad --flow grid-ocr --out-dir output
```

Do not make OCR the primary path again unless the user explicitly asks.

## Exclusions

Do not use Windows desktop item APIs, shell list view APIs, or accessibility APIs to obtain icon positions. That bypasses the vision-grounding requirement.

Do not ask the LLM for full-screen pixel coordinates.

Do not implement JSONPlaceholder fetching, Notepad typing, saving, or closing until visual location is reliable.
