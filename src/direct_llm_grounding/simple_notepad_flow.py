from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import pyautogui
import requests

from direct_llm_grounding.notepad_direct import ask_llm_for_coordinates
from direct_llm_grounding.notepad_direct import capture_desktop
from direct_llm_grounding.notepad_direct import draw_debug_box
from direct_llm_grounding.notepad_direct import load_env_file
from direct_llm_grounding.notepad_direct import parse_coordinate_guess

QUERY = "Notepad"
RUNS = 10
DELAY_BETWEEN_RUNS_SECONDS = 5
API_URL = "https://jsonplaceholder.typicode.com/posts"
OUTPUT_DIR = Path("output/simple_notepad_flow")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DUMMY_POSTS: list[dict] = [
    {"id": 1, "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit", "body": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto"},
    {"id": 2, "title": "qui est esse", "body": "est rerum tempore vitae\nsequi sint nihil reprehenderit dolor beatae ea dolores neque\nfugiat blanditiis voluptate porro vel nihil molestiae ut reiciendis\nqui aperiam non debitis possimus qui neque nisi nulla"},
    {"id": 3, "title": "ea molestias quasi exercitationem repellat qui ipsa sit aut", "body": "et iusto sed quo iure\nvoluptatem occaecati omnis eligendi aut ad\nvoluptatem doloribus vel accusantium quis pariatur\nmolestiae porro eius odio et labore et velit aut"},
    {"id": 4, "title": "eum et est occaecati", "body": "ullam et saepe reiciendis voluptatem adipisci\nsit amet autem assumenda provident rerum culpa\nquis hic commodi nesciunt rem tenetur doloremque ipsam iure\nquis sunt voluptatem rerum illo velit"},
    {"id": 5, "title": "nesciunt quas odio", "body": "repudiandae veniam quaerat sunt sed\nalias aut fugiat sit autem sed est\nvoluptatem omnis possimus esse voluptatibus quis\nest aut tenetur dolor neque"},
    {"id": 6, "title": "dolorem eum magni eos aperiam quia", "body": "ut aspernatur corporis harum nihil quis provident sequi\nmollitia nobis aliquid molestiae\nperspiciatis et ea nemo ab reprehenderit accusantium quas\nvoluptate dolores velit et doloremque molestiae"},
    {"id": 7, "title": "magnam facilis autem", "body": "dolore placeat quibusdam ea quo vitae\nmagni quis enim qui quis quo nemo aut saepe\nquidem repellat excepturi ut quia\nsunt ut sequi eos ea sed quas"},
    {"id": 8, "title": "dolorem dolore est ipsam", "body": "dignissimos aperiam dolorem qui eum\nfacilis quibusdam animi sint suscipit qui sint possimus cum\nquaerat magni maiores excepturi\nipsam ut commodi dolor voluptatum modi aut vitae"},
    {"id": 9, "title": "nesciunt iure omnis dolorem tempora et accusantium", "body": "consectetur animi nesciunt iure dolore\nquis quis cursus aut quam aperiam sequi eum\nquo fugit voluptatem reprehenderit\narchitecto dolores possimus quia quidem id maiores"},
    {"id": 10, "title": "optio molestias id quia eum", "body": "quo et expedita modi cum officia vel magni\ndoloribus qui repudiandae\nvero nisi sit\nquos veniam quod sed accusamus veritatis error"},
]


def run() -> None:
    load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or set it in the shell.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = get_target_directory()
    logger.info("clearing notes folder: %s", target_dir)
    reset_target_directory(target_dir)

    posts = fetch_posts(limit=RUNS)
    for index, post in enumerate(posts, start=1):
        filename = filename_for_post(post, index=index)
        full_path = target_file_for_post(target_dir, post, index=index)
        logger.info("[%d/%d] locating %s", index, len(posts), QUERY)

        screenshot = capture_desktop()
        screenshot_path = OUTPUT_DIR / f"{index:02d}-screenshot.png"
        screenshot.save(screenshot_path)

        raw_response = ask_llm_for_coordinates(screenshot)
        (OUTPUT_DIR / f"{index:02d}-response.txt").write_text(raw_response, encoding="utf-8")

        guess = parse_coordinate_guess(raw_response, image_size=screenshot.size)
        draw_debug_box(screenshot, guess).save(OUTPUT_DIR / f"{index:02d}-annotated.png")
        logger.info("[%s] double-clicking %s at (%d, %d)", filename, QUERY, guess.x, guess.y)

        open_notepad_at(guess.x, guess.y)
        paste_text(format_post_content(post))
        save_notepad_as(full_path)
        close_notepad()

        logger.info("[%s] saved to %s", filename, full_path)
        if index < len(posts):
            logger.info("waiting %d seconds before next run", DELAY_BETWEEN_RUNS_SECONDS)
            time.sleep(DELAY_BETWEEN_RUNS_SECONDS)


def fetch_posts(*, limit: int = RUNS) -> list[dict]:
    try:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        posts = response.json()
        if not isinstance(posts, list):
            raise RuntimeError(f"Unexpected API response shape: {type(posts).__name__}")
        return posts[:limit]
    except Exception as exc:
        logger.warning("API unavailable (%s). Using fallback dummy data.", exc)
        return _DUMMY_POSTS[:limit]


def format_post_content(post: dict) -> str:
    title = str(post.get("title", "")).strip()
    body = str(post.get("body", "")).strip()
    return f"Title: {title}\n\n{body}"


def filename_for_post(post: dict, *, index: int) -> str:
    post_id = post.get("id", index)
    return f"post_{post_id}.txt"


def target_file_for_post(target_dir: Path, post: dict, *, index: int) -> Path:
    return target_dir / filename_for_post(post, index=index)


def get_target_directory() -> Path:
    return Path.home() / "Desktop" / "tjm-project"


def reset_target_directory(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in target_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def open_notepad_at(x: int, y: int) -> None:
    pyautogui.moveTo(x, y, duration=0.3)
    pyautogui.doubleClick()
    time.sleep(2.0)


def paste_text(text: str) -> None:
    set_clipboard_text(text)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)


def save_notepad_as(full_path: Path) -> None:
    pyautogui.hotkey("ctrl", "shift", "s")
    time.sleep(1.0)
    paste_text(str(full_path))
    pyautogui.press("enter")
    time.sleep(1.0)


def close_notepad() -> None:
    pyautogui.hotkey("alt", "f4")
    time.sleep(1.0)


def set_clipboard_text(text: str) -> None:
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    root.destroy()


if __name__ == "__main__":
    run()
