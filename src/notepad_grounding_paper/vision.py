from __future__ import annotations

import json
import os
from openai import OpenAI
from PIL import Image
from notepad_grounding_paper.env import load_env_file
from notepad_grounding_paper.images import image_to_data_url
from notepad_grounding_paper.models import CellChoice
from notepad_grounding_paper.prompts import build_cell_choice_prompt
from notepad_grounding_paper.prompts import build_choice_correction_prompt
from notepad_grounding_paper.prompts import build_click_grid_prompt
from notepad_grounding_paper.prompts import build_revise_cell_choice_prompt

DEFAULT_OPENAI_MODEL = "gpt-5.4"

class OpenAIVisionClient:
    def __init__(self, *, model: str | None = None) -> None:
        load_env_file()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")
        self._client = OpenAI()
        self._model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    def choose_cell(self, *, query: str, image: Image.Image, cell_ids: list[str]) -> CellChoice:
        return self._choose_cell_id(
            prompt=build_cell_choice_prompt(query=query, cell_ids=cell_ids),
            image=image,
            valid_cell_ids=cell_ids,
        )

    def revise_cell_choice(
        self,
        *,
        query: str,
        image: Image.Image,
        cell_ids: list[str],
        rejected_cell_ids: list[str],
        reviewer_rationale: str,
        previous_response_id: str | None,
    ) -> CellChoice:
        remaining = [cell_id for cell_id in cell_ids if cell_id not in set(rejected_cell_ids)]
        return self._choose_cell_id(
            prompt=build_revise_cell_choice_prompt(
                query=query,
                rejected_cell_ids=rejected_cell_ids,
                reviewer_rationale=reviewer_rationale,
                valid_cell_ids=remaining,
            ),
            image=image,
            valid_cell_ids=remaining,
            previous_response_id=previous_response_id,
        )

    def choose_click_grid_cell(
        self,
        *,
        query: str,
        image: Image.Image,
        cell_ids: list[str],
        rejected_cell_ids: list[str] | None = None,
        previous_response_id: str | None = None,
    ) -> CellChoice:
        rejected_cell_ids = rejected_cell_ids or []
        valid_cell_ids = [cell_id for cell_id in cell_ids if cell_id not in set(rejected_cell_ids)]
        return self._choose_cell_id(
            prompt=build_click_grid_prompt(query=query, cell_ids=cell_ids, rejected_cell_ids=rejected_cell_ids),
            image=image,
            valid_cell_ids=valid_cell_ids,
            previous_response_id=previous_response_id,
        )

    def _ask_model(self, *, prompt: str, image: Image.Image, previous_response_id: str | None = None):
        request = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_to_data_url(image), "detail": "high"},
                    ],
                }
            ],
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        return self._client.responses.create(**request)

    def _choose_cell_id(
        self,
        *,
        prompt: str,
        image: Image.Image,
        valid_cell_ids: list[str],
        previous_response_id: str | None = None,
        max_retries: int = 2,
    ) -> CellChoice:
        response = self._ask_model(prompt=prompt, image=image, previous_response_id=previous_response_id)
        for attempt in range(max_retries + 1):
            try:
                choice = parse_cell_choice(response.output_text, valid_cell_ids=valid_cell_ids)
                return CellChoice(
                    cell_id=choice.cell_id,
                    confidence=choice.confidence,
                    rationale=choice.rationale,
                    response_id=response.id,
                )
            except ValueError as exc:
                if attempt == max_retries:
                    raise ValueError(f"LLM failed to return a valid cell_id after {max_retries} retries") from exc
                response = self._ask_model(
                    prompt=build_choice_correction_prompt(error=str(exc), valid_cell_ids=valid_cell_ids),
                    image=image,
                    previous_response_id=response.id,
                )


def parse_cell_choice(text: str, *, valid_cell_ids: list[str]) -> CellChoice:
    payload = _load_json(text)
    cell_id = str(payload.get("cell_id", "")).strip()
    if cell_id not in valid_cell_ids:
        raise ValueError(f"LLM returned invalid cell_id {cell_id!r}; expected one of {valid_cell_ids}")
    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
    rationale = str(payload.get("rationale", "")).strip()
    return CellChoice(cell_id=cell_id, confidence=confidence, rationale=rationale)


def _load_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {text}") from exc
