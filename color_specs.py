"""颜色定义（ColorSpec）与待识别颜色的注册表。

本模块只负责“什么颜色算黄 / 蓝 / 绿 / 红”，也就是 HSV 阈值和可视化颜色；
真正的识别流程在 color_detection.py，后处理在 mask_postprocess.py。
想识别新颜色，只需在下面的 COLORS 里加一条，其它模块无需改动。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ============================================================
# HSV 取值范围（OpenCV 约定）
# ------------------------------------------------------------
# H ∈ [0, 179]，S ∈ [0, 255]，V ∈ [0, 255]
# ============================================================
HUE_MIN, HUE_MAX = 0, 179
SAT_MIN, SAT_MAX = 0, 255
VAL_MIN, VAL_MAX = 0, 255

# 各颜色通用的细化容差：围绕中位色相收紧多少
H_TOLERANCE = 15

# 一个颜色最多允许的色相跨度；超过则跳过自适应细化（红色环绕 H=0，跨了 179）
MAX_REFINE_SPAN = 70


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """把 '#RRGGBB' 转成 OpenCV 的 BGR 三元组（仅用于可视化配色）。"""
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


@dataclass(frozen=True)
class ColorSpec:
    """一种待识别颜色的完整定义。"""

    name: str                                    # 英文名（命令行用）
    label: str                                   # 中文名（打印/标注用）
    ranges: tuple[tuple[int, int], ...]          # 色相区间，可多段（红色环绕 H=0）
    s_min: int
    v_min: int
    s_max: int = SAT_MAX
    v_max: int = VAL_MAX
    bgr: tuple[int, int, int] = (255, 255, 255)  # 可视化颜色
    h_tolerance: int = H_TOLERANCE               # 细化时围绕中位色相的容差
    s_tolerance: int = 60                        # 细化时围绕中位饱和度的容差

    @property
    def lower(self) -> np.ndarray:
        """主色相区间的下限（多段时取第一段）。"""
        return np.array([self.ranges[0][0], self.s_min, self.v_min], dtype=np.uint8)

    @property
    def upper(self) -> np.ndarray:
        """主色相区间的上限（多段时取第一段）。"""
        return np.array([self.ranges[0][1], self.s_max, self.v_max], dtype=np.uint8)

    @property
    def adaptive_allowed(self) -> bool:
        """是否允许自适应细化：色相跨度太大的颜色（如红色）跳过。"""
        return (self.ranges[-1][1] - self.ranges[0][0]) <= MAX_REFINE_SPAN

    def describe(self) -> str:
        """一行描述阈值，用于打印。"""
        ranges = " U ".join(f"H[{lo},{hi}]" for lo, hi in self.ranges)
        return f"{ranges} S[{self.s_min},{self.s_max}] V[{self.v_min},{self.v_max}]"


# ============================================================
# 颜色注册表
# ------------------------------------------------------------
#   blue   : H≈100~128；偏青的浅蓝在 90 附近，所以下限放宽到 85；
#            再往上是紫色（样例紫色花 H≈130~150），必须排除。
#   yellow : H≈18~35；低于 18 开始偏橙，高于 35 开始偏黄绿。
#   green  : H≈40~85。
#   red    : 围绕 H=0 环绕，拆成 [0,10] 和 [170,179] 两段取并集。
# S 下限用于甩掉白色桌面 / 灰色背景，V 下限用于甩掉过暗的像素。
# ============================================================
COLORS: dict[str, ColorSpec] = {
    "blue": ColorSpec(
        name="blue", label="蓝色",
        ranges=((85, 128),), s_min=40, v_min=50,
        bgr=_hex_to_bgr("#3C78FF"),
    ),
    "yellow": ColorSpec(
        name="yellow", label="黄色",
        ranges=((18, 35),), s_min=60, v_min=80,
        bgr=_hex_to_bgr("#FFD400"),
    ),
    "green": ColorSpec(
        name="green", label="绿色",
        ranges=((40, 85),), s_min=50, v_min=50,
        bgr=_hex_to_bgr("#32CD32"),
    ),
    "red": ColorSpec(
        name="red", label="红色",
        ranges=((0, 10), (170, HUE_MAX)), s_min=70, v_min=70,
        bgr=_hex_to_bgr("#FF3B30"),
    ),
}

SUPPORTED_COLORS = tuple(COLORS)


def get_color(color: ColorSpec | str) -> ColorSpec:
    """接受颜色名或 ColorSpec，统一返回 ColorSpec。"""
    if isinstance(color, ColorSpec):
        return color
    key = str(color).strip().lower()
    if key not in COLORS:
        raise KeyError(f"不支持的颜色 '{color}'，可选：{', '.join(SUPPORTED_COLORS)}")
    return COLORS[key]


def resolve_color(raw: str | None, default: str = "yellow") -> str:
    """解析命令行 --color：只接受单个颜色名，未指定时用默认颜色。"""
    name = (raw or default).strip().lower()
    if name not in COLORS:
        raise ValueError(f"不支持的颜色 '{name}'；可选 {', '.join(SUPPORTED_COLORS)}")
    return name
