# notepad-grounding

Vision-based desktop automation project for dynamically grounding the Windows Notepad desktop shortcut.

This repository is currently scoped to **Milestone 1** and **Milestone 2** only:

- scaffold a uv-managed Python project;
- create the package structure for the later grounding pipeline;
- expose a CLI entry point;
- capture a Windows desktop screenshot;
- save a raw screenshot and an annotated coordinate-proof screenshot.

It does not yet run OCR, detect icon candidates, launch Notepad, fetch JSONPlaceholder posts, or ask an LLM for coordinates.

## Runtime Assumptions

Final runtime testing should happen inside a Windows 11 VM, not on macOS.

Recommended Windows setup:

- Windows 10/11, preferably Windows 11 in Parallels.
- Display resolution: `1920 x 1080`.
- Display scale: `100%`.
- Parallels graphics: `Scaled`.
- Clone the repo into a normal Windows folder such as:

```text
C:\Users\<user>\Desktop\notepad-grounding
```

Avoid running from Parallels shared folders like `C:\Mac\Home\...`.

## Setup

Install Python and `uv` inside Windows, then run:

```powershell
uv sync
```

On macOS, the package can be installed and unit-tested, but the actual screenshot proof is expected to fail unless explicitly run as a non-Windows smoke test.

## Screenshot Proof

Run this inside the Windows VM:

```powershell
uv run notepad-grounding screenshot-proof --out-dir output/debug
```

Expected outputs:

```text
output/debug/<timestamp>-desktop-raw.png
output/debug/<timestamp>-desktop-annotated.png
```

The annotated image draws fixed reference boxes around the top-left origin, top-right corner, bottom-left corner, and screen center. It also writes metadata for the runtime OS, captured screenshot size, expected size, and size status.

For stricter validation of the required runtime size:

```powershell
uv run notepad-grounding screenshot-proof --out-dir output/debug --strict-size
```

If the capture is not `1920x1080`, the command exits non-zero in strict mode.

## Development Checks

Run focused tests:

```bash
uv run pytest -v
```

Check the CLI wiring:

```bash
uv run notepad-grounding --help
```

## Current Architecture

The package follows the planned candidate-first grounding layout:

```text
src/notepad_grounding/
  main.py
  config.py
  automation/
    desktop.py
    notepad.py
  grounding/
    annotations.py
    candidates.py
    detector.py
    ocr.py
    scoring.py
    template.py
    verifier.py
  data/
    posts.py
```

Only `automation.desktop`, `grounding.annotations`, and the CLI contain real runtime behavior for this milestone. The remaining modules are placeholders for later milestones.

## Grounding Direction

The project will use a deterministic candidate-first pipeline:

1. capture a desktop screenshot;
2. OCR icon labels;
3. infer icon candidate boxes above labels;
4. score candidates with label, geometry, and template evidence;
5. optionally ask an LLM verifier to choose among numbered candidates;
6. click the deterministic center of the selected candidate.

The LLM verifier, if added later, must choose among candidate IDs. It should not directly predict pixel coordinates.
