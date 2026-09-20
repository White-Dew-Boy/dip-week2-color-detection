"""图片读取，以及 IMG_PATH 配置解析。

单独成模块的原因：cv2.imread 在中文路径下会失败，而本项目目录正好是中文路径，
所以统一走 np.fromfile + imdecode；两个直方图脚本和主程序都复用这里的函数。
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from dotenv import load_dotenv

# 项目根目录（本文件所在目录），用于把相对路径解析成绝对路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# 自动补全后缀时按这个顺序找（.JPG 等大写写法靠大小写不敏感匹配覆盖）
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def _find_case_insensitive(directory: str, wanted_name: str) -> str | None:
    """在 directory 里按名字（忽略大小写）找文件，找到返回完整路径。"""
    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    wanted = wanted_name.casefold()
    for entry in entries:
        if entry.casefold() == wanted and os.path.isfile(os.path.join(directory, entry)):
            return os.path.join(directory, entry)
    return None


def _search_by_name(directory: str, stem: str, recursive: bool) -> str | None:
    """在目录（可含子目录）里找名为 stem + 任一图片后缀的文件。"""
    wanted = {f"{stem}{ext}".casefold() for ext in IMAGE_EXTENSIONS}
    if recursive:
        for root, _, files in os.walk(directory):
            for name in files:
                if name.casefold() in wanted:
                    return os.path.join(root, name)
        return None

    try:
        entries = os.listdir(directory)
    except OSError:
        return None
    for entry in entries:
        if entry.casefold() in wanted and os.path.isfile(os.path.join(directory, entry)):
            return os.path.join(directory, entry)
    return None


def list_available_images() -> list[str]:
    """列出项目里能找到的图片（相对项目根目录），用于报错时提示。"""
    found = []
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "__pycache__")]
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                found.append(os.path.relpath(os.path.join(root, name), PROJECT_DIR))
    return sorted(found)


def resolve_image_path(path: str) -> str:
    """把用户给的图片路径解析成实际存在的文件，支持省略后缀。

    按下面的顺序尝试，命中即返回（所以写全路径时行为与以前完全一致）：
        1. 原样返回（路径存在）
        2. 依次补上 IMAGE_EXTENSIONS 里的后缀
        3. 同目录、同目录的子目录里按名字找（忽略大小写，覆盖 .JPG / .PNG）
        4. 项目目录整体搜一遍（忽略大小写）

    例：IMG_PATH = "assets/cap" 能找到 assets/cap.png；
        IMG_PATH = "flowers" 能找到 assets/flowers.PNG。
    """
    given = path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)

    # 1. 原样（写全了就走这条，不改变原有行为）
    if os.path.isfile(given):
        return given

    # 2. 补后缀
    for ext in IMAGE_EXTENSIONS:
        candidate = given + ext
        if os.path.isfile(candidate):
            return candidate

    # 3. 去掉已有后缀，按文件名匹配（忽略大小写）
    directory = os.path.dirname(given) or PROJECT_DIR
    if os.path.isdir(directory):
        stem = os.path.splitext(os.path.basename(given))[0]
        for exact_name in (os.path.basename(given), stem):
            hit = _find_case_insensitive(directory, exact_name)
            if hit:
                return hit
        hit = _search_by_name(directory, stem, recursive=True)
        if hit:
            return hit

    # 4. 整个项目里搜一遍（这样只写 "cap" 也能找到 assets/cap.png）
    hit = _search_by_name(PROJECT_DIR, os.path.splitext(os.path.basename(given))[0],
                          recursive=True)
    if hit:
        return hit

    available = list_available_images()
    hint = f"；可用图片：{', '.join(available)}" if available else ""
    raise FileNotFoundError(f"找不到图片 -> {given}{hint}")


def get_image_path(cli_path: str | None = None) -> str:
    """确定要处理的图片：命令行参数优先，其次 .env 里的 IMG_PATH。

    返回绝对路径；路径可以省略图片后缀，找不到时抛 FileNotFoundError。
    """
    if cli_path:
        raw = cli_path
    else:
        load_dotenv(os.path.join(PROJECT_DIR, ".env"))
        # .env 里可能写成 IMG_PATH = "assets/xxx.png"，所以要把引号去掉
        raw = os.getenv("IMG_PATH", "").strip().strip('"').strip("'")

    if not raw:
        raise FileNotFoundError("未指定图片：请传入 --image，或在 .env 中设置 IMG_PATH")

    return resolve_image_path(raw)


def load_image(path: str):
    """读取图片，支持中文路径。读取失败返回 None。"""
    if not path or not os.path.isfile(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
