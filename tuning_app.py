"""交互式阈值微调窗口（可选功能，非识别主流程）。

滑块初值就是 color_specs.py 里固定下来的阈值；拖动只影响本次显示，
按 s 可以把当前阈值存成 JSON，方便回头改代码里的默认值。
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from color_specs import HUE_MAX, SAT_MAX, VAL_MAX, ColorSpec
from mask_postprocess import threshold_hsv, to_hsv

WINDOW = "Trackbars"
_BARS = ("H Min", "S Min", "V Min", "H Max", "S Max", "V Max")


def _read_bars() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """读取 6 个滑块的值，并保证下限不大于上限（否则 inRange 全黑）。"""
    h_min, s_min, v_min = (cv2.getTrackbarPos(name, WINDOW) for name in _BARS[:3])
    h_max, s_max, v_max = (cv2.getTrackbarPos(name, WINDOW) for name in _BARS[3:])

    if h_min > h_max:
        h_min, h_max = h_max, h_min
    if s_min > s_max:
        s_min, s_max = s_max, s_min
    if v_min > v_max:
        v_min, v_max = v_max, v_min
    return (h_min, s_min, v_min), (h_max, s_max, v_max)


def run_tuner(image: np.ndarray, spec: ColorSpec, save_path: str) -> None:
    """打开微调窗口；按 s 保存阈值到 save_path，按 q / ESC 退出。"""
    cv2.namedWindow(WINDOW)

    def empty(_value):
        pass

    for name, value, upper in zip(_BARS, (*spec.lower, *spec.upper),
                                  (HUE_MAX, SAT_MAX, VAL_MAX) * 2):
        cv2.createTrackbar(name, WINDOW, int(value), upper, empty)

    print(f"微调模式（{spec.name}）：拖动滑块调整阈值，按 s 保存、按 q / ESC 退出。")

    hsv = to_hsv(image)          # 滤波与色彩空间转换只做一次，循环里只用阈值
    while True:
        lower, upper = _read_bars()
        mask = threshold_hsv(hsv, spec, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)

        cv2.imshow("image", image)
        cv2.imshow("mask", mask)
        cv2.imshow("res", result)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            bounds = {"color": spec.name, "lower": list(lower), "upper": list(upper)}
            with open(save_path, "w", encoding="utf-8") as fp:
                json.dump(bounds, fp, ensure_ascii=False, indent=2)
            print(f"已保存阈值到 {os.path.abspath(save_path)}: {bounds}")

    cv2.destroyAllWindows()
