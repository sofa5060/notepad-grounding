# notepad-grounding

Small Windows desktop grounding proof for a take-home project. The current goal is narrow:

```text
Given a screenshot and a visible label query such as "Notepad",
find the best desktop icon candidate and return a deterministic click center.
```

This reset intentionally avoids a large module tree. The app first needs to prove that it can see the desktop, show the grid/OCR/candidate reasoning, and locate the target reliably. Full Notepad automation comes after this proof works.

## Runtime

Final testing must run inside Windows 10/11, ideally the Windows VM used for the interview.

Recommended setup:

- Windows 10/11
- Resolution: `1920 x 1080`
- Display scale: `100%`
- A desktop shortcut named `Notepad`
- Clone into a normal Windows folder, for example:

```text
C:\Users\<user>\Desktop\notepad-grounding
```

Avoid Parallels shared folders such as `C:\Mac\Home\...` for final testing.

## Setup

Use `uv`:

```powershell
uv sync
```

## Locate Proof

Run inside Windows with the desktop visible:

```powershell
uv run notepad-grounding locate --query Notepad --out-dir output/debug
```

Replay an existing screenshot:

```powershell
uv run notepad-grounding locate --query Notepad --image output/debug/screen.png --out-dir output/debug
```

The command writes:

- `<timestamp>-raw.png`: live screenshot, only when no `--image` is supplied
- `<timestamp>-grid.png`: overlapping OCR tile boxes used by the detector
- `<timestamp>-ocr.png`: OCR label boxes
- `<timestamp>-candidates.png`: inferred icon candidates, scores, and selected center
- `<timestamp>-result.json`: query, screen size, grid settings, selected candidate, and center coordinate

The default path already upscales each OCR tile before Windows OCR, then maps detected boxes back to real screen coordinates. This is useful for small or low-contrast labels.

The overlap is important. It lets labels/icons that sit on a tile boundary appear fully inside at least one neighboring crop, then the OCR boxes are mapped back to screen coordinates and merged.

Advanced OCR tuning flags exist, but they are intentionally hidden from normal help so the VM test command stays simple. Tune them only when a debug image shows a specific failure.

## Current Approach

The first implementation is deterministic:

1. Capture or load a screenshot.
2. Split the screenshot into overlapping OCR tiles.
3. Upscale each tile for Windows OCR.
4. Map tile-local OCR boxes back to screen coordinates.
5. Deduplicate overlapping tile detections.
6. Group text into visible labels, including vertically wrapped icon names and close inline fragments from neighboring tiles.
7. Infer an icon box above each plausible label.
8. Score candidates with fuzzy label matching against the query.
9. Return the center of the best candidate if its score is high enough.
10. Save debug images for every step.

The visible grid image draws the same overlapping tile grid used by OCR, so grid boundaries in `grid.png` show the real OCR crops.

The grid is not treated as a guarantee because Windows allows manual icon placement; Auto Arrange and icon size are user settings.

## Why No LLM Yet

The assignment references ScreenSpot-Pro / ScreenSeekeR. The useful lesson for this project is search-space reduction and explainable visual grounding, not direct LLM coordinate prediction.

An LLM verifier can be added later, but only after deterministic candidates exist. If used, it should choose among numbered candidate IDs. It should not output pixel coordinates.

## Development Checks

```bash
uv run pytest -v
uv run notepad-grounding --help
```

## Next Milestones

1. Validate `locate --query Notepad` on Windows with the icon in top-left, center, and bottom-right positions.
2. Use the grid/OCR/candidate images to inspect OCR tile coverage and candidate geometry.
3. Tune default OCR tile size/overlap in code only if the three required Windows screenshots show a repeatable failure.
4. Add a small optional ScreenSeekeR-lite experiment: planner suggests regions, deterministic code still returns coordinates.
5. Add Notepad launch/save automation only after location proof is reliable.
