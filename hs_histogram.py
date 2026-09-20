"""H-S 二维直方图：看色相 / 饱和度的联合分布，用来验证颜色阈值选得是否合理。

图上会用白色虚线标出某个颜色的 H 区间（默认 yellow，可以在命令行第二个参数里改）。
图片路径同样取 .env 的 IMG_PATH（也可以用命令行第一个参数覆盖）。
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt

from color_specs import COLORS, get_color
from image_io import get_image_path, load_image


def plot_hs_histogram(image, color_name: str = "yellow", title: str = "") -> None:
    """画出 H-S 二维直方图，并标出目标颜色的 H 区间。

    颜色越亮表示该 (H, S) 组合的像素越多；H 分成 180 份，S 分成 256 份。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    plt.figure()
    plt.title(title or "2D Histogram (Hue vs Saturation)")
    plt.xlabel("Saturation")
    plt.ylabel("Hue")
    plt.imshow(histogram, interpolation="nearest", aspect="auto", cmap="jet")

    spec = get_color(color_name)
    for h_value in (spec.ranges[0][0], spec.ranges[-1][1]):
        plt.axhline(h_value, color="w", linestyle="--", linewidth=1)
    plt.text(5, spec.ranges[0][0] + 2,
             f"{spec.name} H range {spec.ranges[0][0]}~{spec.ranges[-1][1]}",
             color="w", fontsize=8)

    plt.colorbar()
    plt.show()


def main() -> int:
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    color_name = sys.argv[2] if len(sys.argv) > 2 else "yellow"
    if color_name not in COLORS:
        sys.exit(f"错误：不支持的颜色 {color_name}；可选 {', '.join(COLORS)}")

    path = get_image_path(cli_path)
    image = load_image(path)
    if image is None:
        sys.exit(f"错误：无法读取图像，请检查路径是否正确 -> {path}")

    plot_hs_histogram(image, color_name, f"{os.path.basename(path)}  (Hue vs Saturation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
