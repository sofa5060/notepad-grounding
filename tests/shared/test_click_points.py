from PIL import Image

from notepad_grounding.shared.click_points import build_click_points
from notepad_grounding.shared.click_points import crop_around_point
from notepad_grounding.shared.click_points import draw_click_points
from notepad_grounding.shared.click_points import offset_point
from notepad_grounding.shared.click_points import point_by_id


def test_build_click_points_creates_stable_ids_and_respects_margin():
    points = build_click_points((100, 100), rows=9, cols=9, margin=10)

    assert len(points) == 81
    assert points[0].id == "P01"
    assert points[-1].id == "P81"
    assert points[0].center == (10, 10)
    assert points[-1].center == (90, 90)
    assert all(10 <= point.center[0] <= 90 for point in points)
    assert all(10 <= point.center[1] <= 90 for point in points)


def test_build_click_points_supports_fine_5_by_5_grid():
    points = build_click_points((50, 50), rows=5, cols=5, margin=5)

    assert len(points) == 25
    assert [point.id for point in points[:3]] == ["P01", "P02", "P03"]
    assert points[12].center == (25, 25)


def test_crop_around_point_and_offset_point_are_deterministic():
    image = Image.new("RGB", (200, 100), "white")

    crop, crop_box = crop_around_point(image, center=(190, 90), size=(60, 60))
    selected = point_by_id(build_click_points(crop.size, rows=5, cols=5, margin=10), "P13")

    assert crop.size == (60, 60)
    assert crop_box == (140, 40, 200, 100)
    assert offset_point(selected.center, offset=(crop_box[0], crop_box[1])) == (170, 70)


def test_draw_click_points_writes_overlay(tmp_path):
    image = Image.new("RGB", (100, 100), "white")
    points = build_click_points(image.size, rows=3, cols=3, margin=10)
    output_path = tmp_path / "points.png"

    draw_click_points(image, points, output_path=output_path, selected_point_id="P05")

    assert output_path.exists()
