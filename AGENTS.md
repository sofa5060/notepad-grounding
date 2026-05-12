# AGENTS.md

## Project

This repository is for a take-home interview project: **Vision-Based Desktop Automation with Dynamic Icon Grounding**.

The current milestone is deliberately small: implement and debug a query-based `locate` command that finds a desktop icon candidate from a screenshot and returns a deterministic center coordinate.

Do not rebuild the large placeholder architecture until the locate proof works on Windows.

## Tooling

- Use `uv` for Python dependency management and running scripts.
- Use `bun` instead of `npm` if JavaScript tooling is ever introduced.
- The project should remain Python-first.
- Final runtime testing must happen inside Windows, not macOS.

## Runtime Assumptions

- Target OS: Windows 10/11.
- Target display: `1920x1080`, `100%` scale.
- The user will create a desktop shortcut named `Notepad`.
- Clone/run from a normal Windows path such as:

```text
C:\Users\<user>\Desktop\notepad-grounding
```

Avoid Parallels shared folders such as `C:\Mac\Home\...` for final testing.

## Current Implementation Direction

Primary command:

```text
uv run notepad-grounding locate --query Notepad --out-dir output/debug
```

Current deterministic flow:

1. Capture or load a screenshot.
2. Run Windows OCR.
3. Group visible text into labels.
4. Infer candidate icon boxes above plausible labels.
5. Score candidates with fuzzy query matching.
6. Return the selected candidate center.
7. Save debug images for grid, OCR labels, candidates, and selected center.

Grid overlays are for debugging and future scoring priors. Do not assume Windows desktop icons are always locked to the grid; Auto Arrange and Align to Grid are user-controlled settings, and icon sizes can change.

## Debugging Expectations

Every locate run should produce visual artifacts:

- grid overlay, so cell size can be inspected;
- OCR overlay, so text detection failures are visible;
- candidate overlay, so inferred icon boxes and scores are visible;
- JSON result, so coordinates and settings are inspectable.

Prefer simple, readable algorithms until screenshots show the failure mode clearly.

## LLM / Paper Context

The assignment references `arXiv:2504.07981`, **ScreenSpot-Pro** and **ScreenSeekeR**.

Use the paper as inspiration for search-space reduction:

- planner suggests possible regions;
- grounder/candidate code works inside those regions;
- final coordinates come from deterministic boxes.

Do not use a pure LLM coordinate predictor as the main solution. If an LLM is added later, it should choose among numbered candidate IDs and remain disabled by default.

## Near-Term Scope

Implement and validate only:

- query-based locate;
- grid/OCR/candidate debug images;
- top-left, center, bottom-right Notepad screenshot proofs.

Do not implement JSONPlaceholder fetching, Notepad typing, saving, or closing until the locate proof is reliable.
