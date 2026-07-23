from PIL import Image

from grid_ocr import main as grid_main
from grid_ocr.ocr import OcrLine


def test_main_saves_artifacts_and_prints_center(monkeypatch, tmp_path, capsys):
    image = Image.new("RGB", (300, 200), "white")
    lines = [OcrLine(text="Notepad", confidence=100, box=(100, 120, 160, 138))]
    monkeypatch.setattr(grid_main, "capture_desktop", lambda: image)
    monkeypatch.setattr(grid_main, "extract_ocr_lines_from_grid", lambda _image: lines)
    monkeypatch.setattr(grid_main, "OUT_DIR", tmp_path)

    grid_main.main()

    assert (tmp_path / "screenshot.png").exists()
    assert (tmp_path / "ocr.png").exists()
    assert (tmp_path / "candidates.png").exists()
    assert capsys.readouterr().out.strip() == "130,82"
