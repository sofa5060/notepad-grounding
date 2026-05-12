from PIL import Image

from notepad_grounding.grounding.annotations import Box, annotate_screenshot


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
