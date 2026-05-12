# AGENTS.md

## Project
This repository is for a take-home interview project: **Vision-Based Desktop Automation with Dynamic Icon Grounding**.

The target runtime is **Windows 10/11 at 1920x1080**, tested inside a Windows VM on a MacBook, likely Parallels. The app must locate the Notepad desktop shortcut dynamically, launch it, write posts from JSONPlaceholder, save files, close Notepad, and repeat for the first 10 posts.

## Tooling
- Use `uv` for Python dependency management and running scripts.
- The user has a global preference to use `bun` instead of `npm` if any JavaScript tooling is introduced, but this project should primarily be Python.
- Final runtime testing must happen inside Windows, not macOS, because screenshot capture and mouse/keyboard automation must control the Windows desktop.

## Development Workflow
- The repo is authored on macOS at `/Users/sofa/Sofa/Coding/notepad-grounding`.
- The repo should be pushed to GitHub and cloned inside Windows to a normal Windows folder such as:
  `C:\Users\<user>\Desktop\notepad-grounding`
- Avoid running from Parallels shared folders like `C:\Mac\Home\...` because Git ownership, filesystem behavior, and automation paths can get weird.

## Agreed Technical Direction
Use a **deterministic candidate-first grounding pipeline** with an optional LLM verifier disabled by default.

Do not make the first implementation a pure LLM coordinate predictor. The interview risk is too high: direct MLLM grounding can hallucinate coordinates and is unreliable for tiny desktop icons.

Preferred flow:
1. Capture desktop screenshot.
2. Use OCR to find desktop icon labels.
3. Infer icon candidate boxes above detected labels.
4. Score candidates using label matching, geometry, and later icon/template similarity.
5. Optionally rerank top candidates with an LLM verifier, but do not ask the LLM for pixel coordinates.
6. Return the deterministic center coordinate of the chosen candidate.
7. Save annotated screenshots showing boxes, scores, and selected target.

## Paper Context
The assignment references `arXiv:2504.07981`, **ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use**.

Key interpretation:
- The paper introduces both a benchmark, **ScreenSpot-Pro**, and an algorithmic framework, **ScreenSeekeR**.
- ScreenSeekeR separates:
  - a **planner model** such as GPT-4o, which suggests likely search areas using GUI knowledge; and
  - a **grounder model** such as OS-Atlas, UGround, or SeeClick, which predicts bounding boxes/points.
- The key idea to borrow is not "ask GPT to click." It is "reduce the search space before grounding."
- Known risks of recursive cropping/search:
  - context loss from over-tight crops
  - cascading errors if an early crop is wrong
  - oversized boxes that still contain many distractors
  - tiny boxes that cut off icon labels
  - coordinate remapping bugs from crop-local coordinates back to screen coordinates
  - non-deterministic model behavior

## Architecture Preference
Keep modules small and testable. Suggested package layout:

```text
src/notepad_grounding/
  __init__.py
  main.py
  config.py

  automation/
    desktop.py        # screenshot, click, type, hotkeys
    notepad.py        # launch validation, save, close

  grounding/
    detector.py       # orchestration: screenshot -> candidates -> scores -> result
    ocr.py            # OCR label detection
    candidates.py     # infer icon boxes from labels
    scoring.py        # weighted evidence scoring
    template.py       # optional Notepad icon template similarity
    verifier.py       # optional LLM candidate verifier
    annotations.py    # draw debug screenshots

  data/
    posts.py          # JSONPlaceholder fetch + fallback sample data

tests/
  test_candidates.py
  test_scoring.py
  test_coordinate_mapping.py
```

## Scoring Direction
Treat candidate selection as weighted evidence, not one fragile detector.

Initial scoring:
- `label_score`: OCR/fuzzy match to `Notepad`.
- `geometry_score`: icon-like layout, label under icon, reasonable dimensions, not taskbar.

Soon after:
- `template_score`: compare candidate icon crop to one or more reference Notepad icons using OpenCV/perceptual similarity.

Optional later:
- `semantic_icon_score`: CLIP/OpenCLIP style similarity if dependencies are practical on Windows ARM.
- `llm_verifier_score`: present numbered candidate crops and ask the model which candidate is Notepad.

The LLM verifier must choose among candidate IDs. It should not output click coordinates.

## Deliverables To Preserve
The final repo should include:
- source code with clear structure
- `pyproject.toml` / uv configuration
- README with setup, run instructions, approach, failure cases, and discussion points
- at least 3 annotated screenshots:
  - Notepad icon in top-left area
  - Notepad icon in bottom-right area
  - Notepad icon in center of screen

## Next Milestone
Build the smallest useful proof:
1. Scaffold a uv Python project.
2. Implement screenshot capture on Windows.
3. Run OCR over the desktop screenshot.
4. Infer desktop icon candidates from OCR labels.
5. Save an annotated screenshot with label boxes and inferred icon candidate boxes.

Do not start with full Notepad automation. First prove that the system can see and box desktop icon candidates.

For the milestone summary, see `docs/project-plan.md`. For deeper background and paper notes, see `docs/project-context.md`.
