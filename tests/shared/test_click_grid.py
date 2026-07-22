from PIL import Image

from notepad_grounding_paper.images import build_grid_cells
from notepad_grounding_paper.images import cell_by_id
from notepad_grounding_paper.images import offset_point
from notepad_grounding_paper.images import crop_around_point
from notepad_grounding_paper.images import draw_click_grid
from notepad_grounding_paper.images import draw_full_click_marker


def test_crop_around_point_and_offset_point_are_deterministic():
    image = Image.new("RGB", (200, 100), "white")

    (crop, crop_box) = crop_around_point(image, center=(190, 90), size=(60, 60))
    selected = cell_by_id(build_grid_cells((0, 0, crop.width, crop.height), rows=5, cols=5, id_fmt="rc"), "R3C3")

    assert crop.size == (60, 60)
    assert crop_box == (140, 40, 200, 100)
    assert offset_point(selected.center, offset=(crop_box[0], crop_box[1])) == (170, 70)


def test_build_click_grid_cells_creates_row_column_ids_and_centers():
    cells = build_grid_cells((0, 0, 70, 70), rows=7, cols=7, id_fmt="rc")

    assert len(cells) == 49
    assert cells[0].id == "R1C1"
    assert cells[-1].id == "R7C7"
    assert cells[0].box == (0, 0, 10, 10)
    assert cells[24].id == "R4C4"
    assert cells[24].center == (35, 35)


def test_draw_click_grid_places_labels_outside_image(tmp_path):
    image = Image.new("RGB", (70, 70), "white")
    cells = build_grid_cells((0, 0, image.width, image.height), rows=7, cols=7, id_fmt="rc")
    output_path = tmp_path / "grid.png"

    draw_click_grid(image, cells, output_path=output_path, selected_cell_id="R4C4")

    assert output_path.exists()
    rendered = Image.open(output_path)
    assert rendered.width > image.width
    assert rendered.height > image.height
    assert cell_by_id(cells, "R4C4").center == (35, 35)


def test_draw_click_grid_uses_red_lines_and_yellow_selected_cell(tmp_path):
    image = Image.new("RGB", (70, 70), "white")
    cells = build_grid_cells((0, 0, image.width, image.height), rows=7, cols=7, id_fmt="rc")
    output_path = tmp_path / "grid.png"

    draw_click_grid(image, cells, output_path=output_path, selected_cell_id="R4C4")

    rendered = Image.open(output_path)
    assert rendered.getpixel((28, 28)) == (255, 0, 0)
    assert rendered.getpixel((58, 58)) == (255, 210, 0)


def test_draw_full_click_marker_uses_red_marker_and_large_label(tmp_path):
    image = Image.new("RGB", (120, 120), "white")
    output_path = tmp_path / "full.png"

    draw_full_click_marker(image, point=(60, 60), output_path=output_path, label="click_point")

    rendered = Image.open(output_path)
    assert rendered.getpixel((60, 60)) == (255, 0, 0)
    red_label_pixels = 0
    for y in range(0, 50):
        for x in range(rendered.width):
            (r, g, b) = rendered.getpixel((x, y))
            if r > 200 and g < 80 and b < 80:
                red_label_pixels += 1
    assert red_label_pixels > 20
