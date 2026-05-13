# Future Requirements & Architecture Notes

## Problem Summary

The current automation has several edge cases that break the flow:

1. **Replace file pop-up** — When saving a file that already exists, Windows shows a "Do you want to replace?" dialog. Not handled.
2. **Wrong icon clicked** — The grounding module found coordinates, but clicking them opened the wrong app (e.g., Steam). Not detected.
3. **Wrong app left open** — After opening the wrong app, it was never closed. The automation didn't recover.
4. **No validation after each step** — We don't verify that the expected outcome actually happened.

## Proposed Architecture: LLM Reviewer / Validator

Instead of a rigid linear flow, add an **LLM-based reviewer** that validates each step using screenshots.

### Concept

After every action, take a screenshot and ask the LLM:

> "I just performed [ACTION]. Looking at the current screen, did it succeed? Is this the expected state?"

The LLM returns:
- `status`: "success", "unexpected_pop_up", "wrong_app", "error"
- `action_needed`: what to do next (e.g., "click Replace button", "close this window and retry", "proceed")
- `rationale`: explanation

### Validation Checkpoints

| Step | Action | Validation Question |
|------|--------|---------------------|
| 1 | Ground + click icon | "Is the active window Notepad?" |
| 2 | Type content | "Does the text area contain the expected content?" |
| 3 | Save file | "Did the save succeed? Any pop-ups?" |
| 4 | Close Notepad | "Is Notepad closed? Is the desktop visible?" |
| N | Any unexpected state | "What is on screen and what should we do?" |

### Recovery Behaviors

| Detected State | Recovery Action |
|---------------|-----------------|
| Wrong app opened | Close the wrong app (`Alt+F4` or `Ctrl+W`), go back to desktop, retry grounding |
| Replace file pop-up | Click "Replace" or "Yes" button |
| Save As dialog still open | The path wasn't typed correctly — retry typing path |
| Notepad not visible after click | Wait longer or retry click |
| Desktop not visible after close | The wrong window is still open — close it |

### Orchestrator vs Reviewer

The user prefers a **reviewer/validator** pattern rather than a full orchestrator:

- The existing flow knows the sequence: ground → click → type → save → close
- The **reviewer** is a validation layer that checks the screenshot after each step
- If validation passes → continue to next step
- If validation fails → reviewer suggests recovery action → execute it → re-validate

### Benefits

1. **Robustness** — Handles unexpected states without hard-coding every edge case
2. **Flexibility** — Works for any pop-up, any wrong app, any dialog
3. **Human-like** — Mimics how a human would look at the screen and decide what to do
4. **Generalizable** — The same reviewer can validate any desktop automation task

## Implementation Ideas

### Option A: Inline validation (simpler)
After each action in `runner.py`, call a validation function:

```python
# After click
validate_state(expected="notepad_open", image=capture_desktop())

# After save
validate_state(expected="file_saved", image=capture_desktop())
```

### Option B: Separate reviewer module (cleaner)
Create `flows/reviewer/reviewer.py` with a `ReviewClient` that:
- Takes a screenshot + context ("I just clicked the Notepad icon")
- Returns a structured decision

```python
class ReviewClient:
    def review_state(self, *, action: str, expected: str, image: Image) -> ReviewResult:
        ...
```

### Structured Outputs with Pydantic

Instead of parsing raw JSON text, we use OpenAI's **structured outputs** (`beta.chat.completions.parse`) with Pydantic models:

```python
from pydantic import BaseModel, Field

class ReviewResultModel(BaseModel):
    status: str = Field(..., description="One of: success, wrong_app, pop_up, error, retry")
    action_needed: str = Field(..., description="Recovery action to take")
    rationale: str = Field(..., description="Explanation")

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[...],
    response_format=ReviewResultModel,
)
result = completion.choices[0].message.parsed  # guaranteed Pydantic object
```

**Benefits:**
- No JSON parsing errors
- Guaranteed schema compliance
- Type-safe throughout the pipeline
- No manual validation needed

### Prompt Design

```
You are a desktop automation reviewer. I just performed: {action_description}

Expected state: {expected_state_description}

Look at the screenshot and determine:
1. Is the expected state achieved?
2. Is there an unexpected pop-up, dialog, or wrong window open?
3. If something is wrong, what is the exact recovery action?
```

The response format is enforced by the Pydantic schema, not by prompt text.

## Next Steps

1. Design the `ReviewClient` interface
2. Implement the prompt and response parser
3. Add validation calls after each step in `runner.py`
4. Implement recovery actions (close wrong window, handle pop-ups)
5. Test with real edge cases

## Notes

- The reviewer needs the same vision capabilities as the grounding flow
- Could reuse `OpenAIVisionClient` with a different prompt
- Consider caching/screenshot artifacts for debugging
- Need to handle timeout if reviewer can't determine state
