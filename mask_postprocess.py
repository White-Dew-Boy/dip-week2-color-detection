"""掩膜（mask）的生成与后处理：二值化、阈值细化、形态学去噪、连通域分析。

流程上属于“拿到 HSV 阈值之后、得到目标区域之前”的那一段，和颜色定义、
可视化、命令行都无关，方便单独调试。
"""

from __future__ import annotations

import cv2
import numpy as np

from color_specs import ColorSpec, get_color

# 中值滤波核大小（奇数）：压制彩色噪点，比高斯滤波更干净且不糊边缘；<=1 表示不滤波
MEDIAN_BLUR_KSIZE = 5
# 开 / 闭运算的结构元大小
MORPH_KSIZE = 5
# 小于该像素面积的色块视为噪声丢弃
MIN_AREA = 100

# 自适应细化所需的最少粗筛像素，太少就不细化，避免被个别噪点带偏
MIN_SAMPLES = 300


def to_hsv(image: np.ndarray, blur_ksize: int = MEDIAN_BLUR_KSIZE) -> np.ndarray:
    """BGR -> HSV，先做中值滤波去噪。"""
    if blur_ksize and blur_ksize > 1:
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        image = cv2.medianBlur(image, ksize)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def threshold_hsv(
    hsv: np.ndarray,
    color: ColorSpec | str,
    lower=None,
    upper=None,
) -> np.ndarray:
    """按颜色阈值二值化，返回 0/255 的 mask。

    - 默认用 color 自带的多段色相区间；红色这种环绕 H=0 的颜色就是两段取并集。
    - 传入 lower/upper（三元组）时改用显式阈值，供 tune 滑块微调使用。
    """
    spec = get_color(color)

    if lower is not None and upper is not None:
        # 显式阈值只有一段区间
        segments = [(int(lower[0]), int(upper[0]))]
        s_min, s_max = int(lower[1]), int(upper[1])
        v_min, v_max = int(lower[2]), int(upper[2])
    else:
        segments = list(spec.ranges)
        s_min, s_max, v_min, v_max = spec.s_min, spec.s_max, spec.v_min, spec.v_max

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_lo, h_hi in segments:
        if h_lo > h_hi:                    # 允许写成 (170, 10) 这种环绕形式
            h_lo, h_hi = h_hi, h_lo
        mask |= cv2.inRange(hsv,
                            np.array([h_lo, s_min, v_min], dtype=np.uint8),
                            np.array([h_hi, s_max, v_max], dtype=np.uint8))
    return mask


def refine_bounds(hsv: np.ndarray, coarse_mask: np.ndarray, spec: ColorSpec):
    """用粗筛结果的 H/S 中位数把阈值收紧，返回 (lower, upper)。

    细化结果绝不会超出固定阈值：下界只会上移、上界只会下移；粗筛像素太少时
    直接返回固定阈值，保证结果可复现。V（明度）保持固定阈值不变，避免阴影 /
    高光处漏检。
    """
    if int(np.count_nonzero(coarse_mask)) < MIN_SAMPLES:
        return spec.lower, spec.upper

    selected = coarse_mask > 0
    h = hsv[..., 0][selected].astype(np.int32)
    s = hsv[..., 1][selected].astype(np.int32)
    h_med, s_med = int(np.median(h)), int(np.median(s))

    h_lo, h_hi = spec.ranges[0][0], spec.ranges[-1][1]
    return (
        np.array([np.clip(h_med - spec.h_tolerance, h_lo, h_hi),
                  np.clip(s_med - spec.s_tolerance, spec.s_min, spec.s_max),
                  spec.v_min], dtype=np.uint8),
        np.array([np.clip(h_med + spec.h_tolerance, h_lo, h_hi),
                  np.clip(s_med + spec.s_tolerance, spec.s_min, spec.s_max),
                  spec.v_max], dtype=np.uint8),
    )


def clean_mask(
    mask: np.ndarray,
    morph_ksize: int = MORPH_KSIZE,
    fill_holes: bool = True,
) -> np.ndarray:
    """形态学后处理：开运算去掉零散噪点，闭运算填补色块内部空洞。"""
    cleaned = mask
    if morph_ksize and morph_ksize >= 3:
        ksize = morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        if fill_holes:
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned


def find_regions(mask: np.ndarray, min_area: int = MIN_AREA) -> list[dict]:
    """在 mask 上找外轮廓，过滤过小的噪点，按面积从大到小返回。

    每项为 dict: contour / area / bbox=(x, y, w, h)。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        regions.append({
            "contour": contour,
            "area": float(area),
            "bbox": cv2.boundingRect(contour),
        })
    regions.sort(key=lambda item: item["area"], reverse=True)
    return regions
