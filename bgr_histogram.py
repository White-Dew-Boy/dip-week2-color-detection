"""BGR 三通道直方图：看一张图里各通道的灰度分布，方便决定阈值区间。

图片路径同样取 .env 的 IMG_PATH（也可以用命令行第一个参数覆盖）。
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt

from image_io import get_image_path, load_image


def plot_bgr_histogram(image, title: str = "BGR Color Histogram") -> None:
    """画出 B / G / R 三个通道的直方图（OpenCV 的通道顺序是 BGR）。"""
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")

    for index, channel_color in enumerate(("b", "g", "r")):
        histogram = cv2.calcHist([image], [index], None, [256], [0, 256])
        plt.plot(histogram, color=channel_color)
        plt.xlim([0, 256])

    plt.show()


def main() -> int:
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None
    path = get_image_path(cli_path)
    image = load_image(path)
    if image is None:
        sys.exit(f"错误：无法读取图像，请检查路径是否正确 -> {path}")

    plot_bgr_histogram(image, os.path.basename(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
