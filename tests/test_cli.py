from notepad_grounding.main import build_parser


def test_locate_parser_accepts_query_and_image():
    parser = build_parser()
    args = parser.parse_args(["locate", "--query", "Notepad", "--image", "screen.png"])

    assert args.command == "locate"
    assert args.query == "Notepad"
    assert str(args.image) == "screen.png"
    assert args.flow == "llm-visual"
    assert args.ocr_mode == "grid"
    assert args.ocr_scale == 2
    assert args.ocr_tile_width == 320
    assert args.ocr_tile_height == 240
    assert args.ocr_tile_overlap == 48


def test_locate_parser_accepts_hidden_ocr_tuning_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "locate",
            "--query",
            "Notepad",
            "--flow",
            "grid-ocr",
            "--image",
            "screen.png",
            "--ocr-scale",
            "2",
            "--ocr-tile-width",
            "320",
            "--ocr-tile-height",
            "240",
            "--ocr-tile-overlap",
            "48",
        ]
    )

    assert args.command == "locate"
    assert args.query == "Notepad"
    assert args.flow == "grid-ocr"
    assert str(args.image) == "screen.png"
    assert args.ocr_scale == 2
    assert args.ocr_mode == "grid"
    assert args.ocr_tile_width == 320
    assert args.ocr_tile_height == 240
    assert args.ocr_tile_overlap == 48
