from notepad_grounding.main import build_parser


def test_locate_parser_accepts_query_and_image():
    parser = build_parser()
    args = parser.parse_args(["locate", "--query", "Notepad", "--image", "screen.png"])

    assert args.command == "locate"
    assert args.query == "Notepad"
    assert str(args.image) == "screen.png"
