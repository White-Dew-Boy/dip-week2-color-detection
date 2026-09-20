"""可视化：把识别结果画成方便看的图（轮廓标注、彩色掩膜、对比拼图）。

这里都是“画图”的纯函数，输入识别结果、输出图片数组，不涉及文件读写和命令行。
"""

from __future__ import annotations

import cv2
import numpy as np

from color_specs import ColorSpec

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 轮廓颜色：单色识别时用本色；无法确定颜色时按序号从这里取
_PALETTE = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
    (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255),
]


def draw_label(image: np.ndarray, text: str, origin, bgr, scale: float = 0.6) -> None:
    """就地写一行带半透明底板的大字，保证在复杂背景上也看得清。"""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    y0, y1 = max(0, y - h - 6), min(image.shape[0], y + 6)
    x0, x1 = max(0, x - 4), min(image.shape[1], x + w + 4)
    if y1 > y0 and x1 > x0:
        patch = image[y0:y1, x0:x1]
        patch[:] = cv2.addWeighted(patch, 0.35, np.zeros_like(patch), 0.65, 0)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, bgr, 2, cv2.LINE_AA)


def draw_regions(image: np.ndarray, regions, bgr=WHITE, prefix: str = "") -> np.ndarray:
    """在原图上画出区域轮廓、包围盒和面积标注。"""
    canvas = image.copy()
    for index, region in enumerate(regions, start=1):
        color = _PALETTE[(index - 1) % len(_PALETTE)] if bgr is None else bgr
        cv2.drawContours(canvas, [region["contour"]], -1, color, 2)
        x, y, w, h = region["bbox"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        cv2.putText(canvas, f"{prefix}#{index} {region['area']:.0f}px",
                    (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def colorize_mask(mask: np.ndarray, color: ColorSpec) -> np.ndarray:
    """把二值掩膜染成该颜色的实色图（便于和原图并排对比）。"""
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    canvas[mask > 0] = color.bgr
    return canvas


def to_bgr(image: np.ndarray) -> np.ndarray:
    """灰度图补成三通道，否则 hconcat 会因为类型不一致直接报错。"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def tile_images(tiles, titles=None, gap: int = 4, background=WHITE) -> np.ndarray:
    """把若干张图缩放到同一高度后横向拼接，并在左上角写标题。"""
    if titles is None:
        titles = [""] * len(tiles)
    tiles = [to_bgr(tile) for tile in tiles]
    target_h = min(tile.shape[0] for tile in tiles)

    resized = []
    for tile in tiles:
        scale = target_h / tile.shape[0]
        resized.append(cv2.resize(tile, (max(1, int(tile.shape[1] * scale)), target_h),
                                  interpolation=cv2.INTER_AREA))

    for tile, title in zip(resized, titles):
        if not title:
            continue
        # 深色底图用白字、浅色底图（如 mask）用黑字
        text_color = BLACK if float(tile.mean()) > 127 else WHITE
        cv2.putText(tile, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    text_color, 2, cv2.LINE_AA)

    separator = np.full((target_h, gap, 3), background, dtype=np.uint8)
    parts = []
    for tile in resized:
        parts.extend([tile, separator])
    return cv2.hconcat(parts[:-1])


def fit_for_display(image: np.ndarray, max_width: int | None = 1600) -> np.ndarray:
    """图片太宽时等比缩小，避免窗口超出屏幕（只影响显示，与识别 / 保存无关）。

    max_width 传 None 或 0 表示不限制，按原始分辨率显示（画质不变）。
    """
    if not max_width or image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    return cv2.resize(image, (max_width, max(1, int(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
