"""颜色识别的核心流程：把一张图变成“某种颜色的掩膜 + 区域列表 + 统计信息”。

这里只做识别，不做显示、不读命令行、不写文件；
掩膜的生成与后处理细节在 mask_postprocess.py，颜色阈值在 color_specs.py。
"""

from __future__ import annotations

import cv2
import numpy as np

from color_specs import ColorSpec, get_color
from mask_postprocess import (
    MEDIAN_BLUR_KSIZE,
    MIN_AREA,
    MORPH_KSIZE,
    clean_mask,
    find_regions,
    refine_bounds,
    threshold_hsv,
    to_hsv,
)


def detect_color(
    image: np.ndarray,
    color: ColorSpec | str = "blue",
    adaptive: bool | None = None,
    blur_ksize: int = MEDIAN_BLUR_KSIZE,
    morph_ksize: int = MORPH_KSIZE,
    min_area: int = MIN_AREA,
    fill_holes: bool = True,
    keep_largest_only: bool = False,
) -> dict:
    """识别一种颜色。

    adaptive=None 时按颜色自身的设定决定是否做自适应细化；
    显式传 True/False 可强制开 / 关（命令行 --no-adaptive 走的就是 False）。

    返回 dict：
        color   使用的 ColorSpec
        mask    0/255 的二值掩膜
        result  只保留该颜色的原图
        regions 区域列表（contour / area / bbox）
        lower / upper  实际使用的阈值
        stats   像素数、占比、区域数
    """
    spec = get_color(color)
    hsv = to_hsv(image, blur_ksize)

    # 第一步：用固定阈值粗筛
    coarse = threshold_hsv(hsv, spec)

    use_adaptive = spec.adaptive_allowed if adaptive is None else adaptive
    used_lower, used_upper = spec.lower, spec.upper
    if use_adaptive:
        used_lower, used_upper = refine_bounds(hsv, coarse, spec)
        # 阈值收紧后重新二值化，进一步减少背景干扰
        mask = threshold_hsv(hsv, spec, used_lower, used_upper)
    else:
        mask = coarse

    mask = clean_mask(mask, morph_ksize, fill_holes, keep_largest_only)
    regions = find_regions(mask, min_area)
    result = cv2.bitwise_and(image, image, mask=mask)

    pixels = int(np.count_nonzero(mask))
    total = int(mask.size)
    return {
        "color": spec,
        "mask": mask,
        "result": result,
        "regions": regions,
        "lower": used_lower,
        "upper": used_upper,
        "stats": {
            "pixels": pixels,
            "total_pixels": total,
            "ratio": pixels / float(total) if total else 0.0,
            "region_count": len(regions),
        },
    }
