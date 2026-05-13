from notepad_grounding.main import build_parser


def test_locate_parser_accepts_query_and_image():
    parser = build_parser()
    args = parser.parse_args(["locate", "--query", "Notepad", "--image", "screen.png"])

    assert args.command == "locate"
    assert args.query == "Notepad"
    assert str(args.image) == "screen.png"


def test_automate_parser_accepts_query():
    parser = build_parser()
    args = parser.parse_args(["automate", "--query", "Notepad"])

    assert args.command == "automate"
    assert args.query == "Notepad"
