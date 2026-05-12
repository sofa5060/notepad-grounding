# Project Plan

## Goal
Build a Python Windows desktop automation app that dynamically finds the Notepad desktop shortcut, launches it, writes the first 10 JSONPlaceholder posts, saves each one as `post_{id}.txt` in `Desktop/tjm-project`, closes Notepad, and repeats from a fresh screenshot.

The important interview focus is **visual grounding**: the app must find the Notepad icon even when its desktop position changes.

## Chosen Approach
Use a **candidate-first grounding pipeline**:

1. Capture the desktop screenshot.
2. Detect desktop icon label text with OCR.
3. Infer icon candidate boxes above labels.
4. Score each candidate using deterministic signals.
5. Optionally ask an LLM verifier to choose among numbered candidates.
6. Click the deterministic center of the selected candidate.

Do not rely on a pure LLM coordinate prediction as the primary implementation.

## Why This Approach
Direct MLLM grounding is flexible but risky: small desktop icons can be hard to localize, coordinates can be imprecise, and outputs can vary between runs.

The candidate-first approach makes the problem easier and safer:

- OCR and geometry generate measurable candidates.
- Template/icon similarity can add deterministic visual evidence.
- An LLM, if used, only chooses among candidate IDs.
- Debug screenshots can show exactly why a candidate was selected.

## Planned Architecture

```text
src/notepad_grounding/
  main.py
  config.py

  automation/
    desktop.py
    notepad.py

  grounding/
    detector.py
    ocr.py
    candidates.py
    scoring.py
    template.py
    verifier.py
    annotations.py

  data/
    posts.py

tests/
  test_candidates.py
  test_scoring.py
  test_coordinate_mapping.py
```

## Milestones

### 1. Project Skeleton
- Create uv Python project.
- Add package layout.
- Add CLI entry point.
- Add README with Windows setup notes.

### 2. Screenshot + Annotation
- Capture Windows desktop screenshot.
- Save raw screenshot.
- Save annotated screenshot with test boxes.

### 3. OCR Candidate Generation
- Run OCR over screenshot.
- Detect label boxes.
- Infer icon boxes above labels.
- Save annotated candidate screenshot.

### 4. Deterministic Scoring
- Add fuzzy label scoring for `Notepad`.
- Add geometry scoring.
- Select best candidate and return center coordinate.

### 5. Template Similarity
- Add reference Notepad icon templates.
- Compare candidate icon crops across sizes/scales.
- Integrate template score.

### 6. Notepad Automation
- Double-click selected candidate.
- Validate Notepad launched.
- Type content.
- Save file.
- Close Notepad.

### 7. Full Workflow
- Fetch first 10 posts from JSONPlaceholder.
- Create `Desktop/tjm-project`.
- Repeat screenshot -> ground -> launch -> write -> save -> close.
- Handle existing files and API failures gracefully.

### 8. Robustness
- Retry detection up to 3 attempts with 1-second delays.
- Improve failure messages.
- Add optional LLM verifier behind config flag.

### 9. Deliverables
- Add README explanation of approach, failure cases, and paper relation.
- Capture annotated screenshots for:
  - top-left Notepad icon
  - center Notepad icon
  - bottom-right Notepad icon

## Immediate Next Step
Start with milestones 1 and 2. Prove screenshot capture and annotation inside the Windows VM before building the full automation loop.
