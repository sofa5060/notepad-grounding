from PIL import Image

from notepad_grounding.grounding.annotations import Box, annotate_screenshot
from notepad_grounding.grounding.annotations import annotate_candidate_proof
from notepad_grounding.grounding.candidates import IconCandidate
from notepad_grounding.grounding.ocr import OcrLine


def test_annotate_screenshot_draws_boxes_without_resizing(tmp_path):
    image = Image.new("RGB", (120, 80), "white")
    boxes = [Box(10, 10, 40, 30, "sample")]

    output_path = tmp_path / "annotated.png"
    result = annotate_screenshot(image, boxes=boxes, output_path=output_path)

    assert result == output_path
    assert output_path.exists()

    annotated = Image.open(output_path)
    assert annotated.size == (120, 80)
    assert annotated.getpixel((10, 10)) != (255, 255, 255)


def test_annotate_candidate_proof_draws_label_icon_and_combined_boxes(tmp_path):
    image = Image.new("RGB", (180, 140), "white")
    label = OcrLine("Notepad", 90, Box(70, 90, 120, 108, "Notepad"))
    candidate = IconCandidate(
        candidate_id=1,
        label_text="Notepad",
        label_box=label.box,
        icon_box=Box(78, 40, 112, 84, "icon"),
        combined_box=Box(68, 34, 122, 108, "#1 Notepad"),
        confidence_notes=[],
    )

    output_path = tmp_path / "candidate-proof.png"
    annotate_candidate_proof(
        image,
        labels=[label],
        candidates=[candidate],
        output_path=output_path,
    )

    annotated = Image.open(output_path)
    assert annotated.size == (180, 140)
    assert annotated.getpixel((68, 34)) == (255, 210, 0)
    assert annotated.getpixel((78, 50)) == (0, 120, 255)
    assert annotated.getpixel((70, 90)) == (0, 180, 80)
