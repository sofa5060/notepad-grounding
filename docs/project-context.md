# Notepad Grounding Project Context

## Current State
The project is an empty Git repository created at:

```text
/Users/sofa/Sofa/Coding/notepad-grounding
```

The user plans to edit from macOS, push to GitHub, and clone/run inside a Windows VM.

## Assignment Summary
Build a Python application that runs on Windows 10/11 at 1920x1080 and dynamically locates the Notepad desktop shortcut using computer vision. It must launch Notepad through the desktop icon, fetch the first 10 posts from JSONPlaceholder, write each post to Notepad, save it as `post_{id}.txt` inside `Desktop/tjm-project`, close Notepad, then repeat from a fresh screenshot.

The key evaluation target is visual grounding. The app should not hardcode icon coordinates.

## Runtime Environment Plan
- Use Parallels on Mac to run Windows 11.
- Configure Parallels graphics to **Scaled**.
- Configure Windows display settings:
  - resolution: `1920 x 1080`
  - scale: `100%`
- Create a desktop shortcut named `Notepad`.
- Install Git, Python, and `uv` inside Windows.
- Clone the GitHub repo into a real Windows path, not a Parallels shared Mac folder.

## Core Design Decision
Use **candidate-first visual grounding**.

Instead of asking an LLM or model to directly return a coordinate from the full screenshot, first generate a set of likely desktop icon candidates. Then score and select among those candidates.

This gives a better failure mode:
- if detection fails, we can inspect candidate boxes and scores;
- if an LLM is used, it only chooses among numbered candidates;
- click coordinates remain deterministic.

## Candidate Generation Idea
Desktop icons generally consist of:
- an icon image;
- a text label below it.

The first detector should:
1. run OCR over the desktop screenshot;
2. find text labels that look like desktop icon labels;
3. infer the icon region above each label;
4. create candidate boxes containing both the icon and label;
5. merge duplicate candidates when OCR was run over overlapping crops.

Use overlapping tiles if full-screen OCR performs poorly. Avoid hard tile boundaries without overlap because labels may be split across tiles.

## Desktop Inventory Label Prior
Because the automation runs on the target Windows desktop, the app can later query the Desktop directory and collect known item names before interpreting OCR results.

This should be used as a **label prior**, not as a coordinate source:
- filesystem/Desktop inventory can tell us likely labels such as `Notepad`;
- OCR can tell us where text-like regions appear in the screenshot;
- fuzzy matching can connect noisy OCR strings such as `otepad`, `Note pad`, or truncated labels to known desktop item names;
- final click coordinates must still come from screenshot-derived candidate boxes.

This helps reduce false positives from OCR noise while preserving the assignment's visual-grounding requirement.

## Scoring Plan
Each candidate should collect evidence:

```text
total_score =
  label_score * 0.45 +
  template_score * 0.30 +
  semantic_icon_score * 0.20 +
  geometry_score * 0.05
```

The exact weights can change after testing.

### Label Score
Use OCR text and fuzzy matching.

Examples:
- `Notepad` -> high confidence
- `notepad` -> high confidence
- `Note pad` -> medium/high confidence
- OCR text that fuzzy-matches a known Desktop item name -> boosted confidence
- `Notepad++` -> suspicious, not exact target
- unrelated labels -> low confidence

### Template Score
Compare the inferred icon crop to one or more reference Notepad icons at multiple scales.

Potential methods:
- OpenCV template matching
- perceptual hash similarity
- edge/shape similarity

This is deterministic but can be brittle across Windows versions, icon sizes, scaling, themes, and shortcut overlays.

### Semantic Icon Score
Optional. Use CLIP/OpenCLIP-style image-text or image-image embedding similarity if practical on Windows ARM.

This may tolerate icon variation better than template matching, but it adds heavier dependencies and uncertainty.

### LLM Verifier
Optional and disabled by default.

Use only after candidate generation:

```text
Here are numbered candidate crops. Which candidate contains the Notepad desktop shortcut?
Return JSON with candidate_id and confidence.
```

The verifier must not output coordinates. Coordinates come from the selected candidate box.

## Paper Notes
The referenced paper is **ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use**.

Important takeaways:
- It defines **GUI grounding** as mapping an instruction and screenshot to a precise UI target location.
- It introduces **ScreenSpot-Pro**, a benchmark for high-resolution professional GUI grounding.
- It proposes **ScreenSeekeR**, a visual search framework.
- ScreenSeekeR uses:
  - a planner model to infer likely search regions; and
  - a grounder model to locate boxes/points inside those regions.
- Its practical insight is search-space reduction: high-resolution screenshots make small targets hard, so crop/search in likely regions.

Important limitations discussed:
- cropping can lose context;
- early wrong crops cause cascading errors;
- boxes can be too large or too small;
- target icon and label can be separated by a bad crop;
- model outputs can be non-deterministic;
- coordinate remapping can break;
- direct MLLM grounding is not reliable enough for a demo.

## Milestones

### Milestone 1: Project Skeleton
- Create `pyproject.toml` with uv configuration.
- Create package structure under `src/notepad_grounding`.
- Add a CLI entry point.
- Add README with setup assumptions.

### Milestone 2: Screenshot + Annotation
- Capture a Windows desktop screenshot.
- Save raw screenshot under a debug/output folder.
- Draw test annotations on it.

### Milestone 3: OCR Candidate Generation
- Run OCR on screenshot.
- Detect text labels.
- Infer icon boxes above labels.
- Save annotated screenshot with OCR boxes and inferred candidate boxes.

### Milestone 4: Deterministic Scoring
- Implement fuzzy label scoring.
- Implement geometry scoring.
- Select best Notepad candidate.
- Return center coordinate.

### Milestone 5: Template Similarity
- Add Notepad reference templates.
- Compare candidate icon crops at multiple scales.
- Integrate template score into total score.

### Milestone 6: Notepad Automation
- Double-click selected candidate.
- Validate Notepad launched.
- Type content.
- Save as `Desktop/tjm-project/post_{id}.txt`.
- Close Notepad.

### Milestone 7: Full Workflow
- Fetch first 10 posts from JSONPlaceholder.
- Gracefully fall back if API is unavailable.
- Repeat screenshot -> ground -> launch -> write -> save -> close.
- Handle existing output files predictably.

### Milestone 8: Robustness + Optional LLM Verifier
- Retry detection up to 3 times with 1-second delays.
- Add clearer failure messages.
- Optionally implement numbered-candidate LLM verifier behind a config flag.

### Milestone 9: Deliverables
- Capture and commit/share three annotated screenshots:
  - top-left icon location
  - center icon location
  - bottom-right icon location
- Finish README discussion:
  - why this approach
  - failure cases
  - performance
  - how to extend to arbitrary icons/buttons
  - relation to ScreenSeekeR

## Recommended Immediate Next Step
Start with Milestone 1 and Milestone 2. Do not implement the API or Notepad save loop first. The grounding candidate screenshot is the technical core and should be proven early.
