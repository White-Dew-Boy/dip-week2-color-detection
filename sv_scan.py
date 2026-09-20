"""S-V 检测扫描图：指定一个色相 H，扫描 S × V 平面，看哪些 (S, V) 能被检测到。

用法：
    python sv_scan.py --hue 28                  # H=28（黄色区）扫 S×V
    python sv_scan.py --hue 110                 # H=110（蓝色区）
    python sv_scan.py --hue 28 --save out/      # 存成 PNG（out/sv_scan_h28.png）
    python sv_scan.py --hue 28 --no-show        # 只打印统计 / 存图，不弹窗

输出内容：
    一张 S-V 色卡（横轴 Saturation 0~255，纵轴 V 255~0），
    显示每个 (S, V) 组合在该 H 下的**真实颜色**，检测不到的格子留黑；
    右侧图例列出每种已注册颜色能检测到的占比。

判定与识别逻辑完全一致：直接复用 mask_postprocess.threshold_hsv()，
所以图上显示"能检测到"的格子，程序在图上就一定能识别到。
注意 H 轴在 OpenCV 里是 0~179；红色跨 H=0 环绕，它的两段区间都会被正确判定。
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

import visualization as vis
from color_specs import COLORS, HUE_MAX, SAT_MAX, VAL_MAX, HUE_MIN, get_color
from mask_postprocess import threshold_hsv

CELL = 4                       # 每格放大倍数
BLANK_COLOR = (0, 0, 0)        # 检测不到的格子：留空（纯黑）
TITLE_H = 34
AXIS_BOTTOM = 40
AXIS_LEFT = 56
PANEL_BG = (28, 28, 28)
AXIS_COLOR = (210, 210, 210)
LEGEND_W = 300                 # 右侧图例宽度


# ============================================================
# 1. 扫描：固定 H，让 S 和 V 各自变化
# ============================================================
def build_hsv_grid(h: int) -> np.ndarray:
    """生成 S-V 网格（固定 H）：横轴 S 0→255，纵轴 V 255→0。"""
    sat = np.arange(SAT_MAX + 1, dtype=np.uint8)[None, :]           # (1, 256)
    val = np.arange(VAL_MAX, -1, -1, dtype=np.uint8)[:, None]       # (256, 1) 上亮下暗
    hsv = np.zeros((VAL_MAX + 1, SAT_MAX + 1, 3), dtype=np.uint8)
    hsv[..., 0] = h
    hsv[..., 1] = sat
    hsv[..., 2] = val
    return hsv


def scan(h: int):
    """在固定 H 的 S-V 平面上扫描所有已注册颜色，返回 (真实色域, 掩膜字典)。"""
    hsv = build_hsv_grid(h)
    raw = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    masks = {name: threshold_hsv(hsv, spec) > 0 for name, spec in COLORS.items()}
    return raw, masks


# ============================================================
# 2. 坐标轴、色卡与图例
# ============================================================
def draw_axis(canvas: np.ndarray, left: int, top: int, width: int, height: int) -> None:
    """横轴 S（每 64 一格）、纵轴 V（每 64 一格），轴标题放在刻度数字之外。"""
    bottom = top + height
    cv2.rectangle(canvas, (left, top), (left + width, bottom), AXIS_COLOR, 1)

    ticks = list(range(0, SAT_MAX + 1, 64))
    if ticks[-1] != SAT_MAX:
        ticks.append(SAT_MAX)

    # 纵轴标题放最左侧，刻度数字整体右移，避免与标题重叠
    vis.draw_label(canvas, "V", (6, top - 6), AXIS_COLOR, 0.45)
    for v in ticks:
        y = top + (VAL_MAX - v) * CELL
        cv2.line(canvas, (left - 5, y), (left, y), AXIS_COLOR, 1)
        vis.draw_label(canvas, str(v), (20, y + 4), AXIS_COLOR, 0.45)

    for s in ticks:
        x = left + s * CELL
        cv2.line(canvas, (x, bottom), (x, bottom + 5), AXIS_COLOR, 1)
        vis.draw_label(canvas, str(s), (x - 10, bottom + 24), AXIS_COLOR, 0.45)
    vis.draw_label(canvas, "Saturation (S)", (left + width // 2 - 45, bottom + 38),
                   AXIS_COLOR, 0.45)


def draw_boundaries(canvas: np.ndarray, name: str, left: int, top: int) -> None:
    """把该颜色的 S 下限、V 下限画成虚线（一眼看出"门槛"在哪）。"""
    spec = get_color(name)
    x = left + spec.s_min * CELL
    for y in range(top, top + (VAL_MAX + 1) * CELL, 12):
        cv2.line(canvas, (x, y), (x, min(y + 6, top + (VAL_MAX + 1) * CELL)), spec.bgr, 1)
    y = top + (VAL_MAX - spec.v_min) * CELL
    for x0 in range(left, left + (SAT_MAX + 1) * CELL, 12):
        cv2.line(canvas, (x0, y), (min(x0 + 6, left + (SAT_MAX + 1) * CELL), y), spec.bgr, 1)


def make_panel(image: np.ndarray, title: str) -> np.ndarray:
    """给色卡加上标题条与坐标轴。"""
    height, width = image.shape[:2]
    panel_h, panel_w = height * CELL, width * CELL
    canvas = np.full((TITLE_H + panel_h + AXIS_BOTTOM, AXIS_LEFT + panel_w + 16, 3),
                     PANEL_BG, dtype=np.uint8)
    scaled = cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    top, left = TITLE_H, AXIS_LEFT
    canvas[top:top + panel_h, left:left + panel_w] = scaled
    vis.draw_label(canvas, title, (10, 24), (255, 255, 255), 0.6)
    draw_axis(canvas, left, top, panel_w, panel_h)
    return canvas


def draw_legend(panel: np.ndarray, left: int, top: int, masks: dict, h: int) -> None:
    """右侧列出每种颜色在该 H 下的可检测占比。"""
    total = (SAT_MAX + 1) * (VAL_MAX + 1)
    y = top + 26
    for name, spec in COLORS.items():
        hits = int(np.count_nonzero(masks[name]))
        x = left + 14
        cv2.rectangle(panel, (x, y - 13), (x + 18, y + 5), spec.bgr, -1)
        vis.draw_label(panel, name, (x + 26, y), (255, 255, 255), 0.5)
        vis.draw_label(panel, f"{hits * 100.0 / total:.2f}% of S-V plane",
                       (x + 26, y + 21), (165, 165, 165), 0.42)
        in_range = any(lo <= h <= hi for lo, hi in spec.ranges)
        note = f"H={h} in {spec.ranges[0][0]}~{spec.ranges[-1][1]}" if in_range \
            else f"H={h} outside this color"
        vis.draw_label(panel, note, (x + 26, y + 40),
                       spec.bgr if in_range else (150, 150, 150), 0.4)
        y += 68

    x = left + 14
    cv2.rectangle(panel, (x, y - 13), (x + 18, y + 5), BLANK_COLOR, -1)
    cv2.rectangle(panel, (x, y - 13), (x + 18, y + 5), (120, 120, 120), 1)
    vis.draw_label(panel, "not shown (blank)", (x + 26, y), (255, 255, 255), 0.5)
    claimed = np.zeros(next(iter(masks.values())).shape, dtype=bool)
    for hit in masks.values():
        claimed |= hit
    vis.draw_label(panel, f"{100 - 100.0 * int(np.count_nonzero(claimed)) / total:.2f}% of S-V plane",
                   (x + 26, y + 21), (165, 165, 165), 0.42)


# ============================================================
# 3. 文字统计
# ============================================================
def print_summary(h: int, masks: dict) -> None:
    """打印该 H 下每个颜色能检测到的 S-V 门槛。"""
    total = (SAT_MAX + 1) * (VAL_MAX + 1)
    print(f"扫描切片    : H={h}，S 0~{SAT_MAX} × V 0~{VAL_MAX}，共 {total} 格")
    print(f"参与判定    : {', '.join(masks)}")
    print()
    print(f"{'颜色':<8}{'H 是否在该色区间':<18}{'阈值':<40}{'可检测格子':>10}{'占比':>9}")
    print("-" * 87)
    for name, hit in masks.items():
        spec = get_color(name)
        in_range = "是" if any(lo <= h <= hi for lo, hi in spec.ranges) else "否（必为 0）"
        hits = int(np.count_nonzero(hit))
        print(f"{name:<8}{in_range:<18}{spec.describe():<40}{hits:>10}{hits * 100.0 / total:>8.2f}%")

    covered = np.zeros(next(iter(masks.values())).shape, dtype=bool)
    for hit in masks.values():
        covered |= hit
    covered = int(np.count_nonzero(covered))
    print(f"{'合计':<8}{'(至少一种颜色能检测到)':<18}{'':<40}{covered:>10}"
          f"{covered * 100.0 / total:>8.2f}%")

    # 只要该 H 落在某个颜色的区间内，就能直接从阈值推出门槛
    print()
    for name, hit in masks.items():
        spec = get_color(name)
        if not any(lo <= h <= hi for lo, hi in spec.ranges):
            continue
        s_ok = int(hit.any(axis=0).sum())
        v_ok = int(hit.any(axis=1).sum())
        print(f"[{name}] H={h} 落在区间内：S ≥ {spec.s_min}（满足的 S 取值共 {s_ok} 个），"
              f"V ≥ {spec.v_min}（满足的 V 取值共 {v_ok} 个）")


# ============================================================
# 4. 命令行
# ============================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="指定色相 H，扫描 S-V 平面看哪些 (S,V) 能被检测到")
    parser.add_argument("--hue", type=int, default=28, metavar=f"0-{HUE_MAX}",
                        help=f"要扫描的色相 H（默认 28，黄色区），范围 {HUE_MIN}~{HUE_MAX}")
    parser.add_argument("--save", "-s", default=None, metavar="DIR",
                        help="保存到该目录（文件名含 H 值）")
    parser.add_argument("--no-show", action="store_true", help="不弹窗，只打印统计 / 存图")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not HUE_MIN <= args.hue <= HUE_MAX:
        sys.exit(f"错误：--hue 必须在 {HUE_MIN}~{HUE_MAX} 之间（OpenCV 的 H 范围），"
                 f"收到 {args.hue}")

    raw, masks = scan(args.hue)
    print_summary(args.hue, masks)

    # 色卡：显示每个 (S, V) 的真实颜色，检测不到的格子留空
    claimed = np.zeros(raw.shape[:2], dtype=bool)
    for hit in masks.values():
        claimed |= hit
    detection = np.where(claimed[..., None], raw, BLANK_COLOR).astype(np.uint8)

    panel = make_panel(detection, f"detectable colors at H={args.hue}")
    wide = np.full((panel.shape[0], panel.shape[1] + LEGEND_W, 3), PANEL_BG, dtype=np.uint8)
    wide[:, :panel.shape[1]] = panel
    draw_legend(wide, panel.shape[1], TITLE_H, masks, args.hue)
    for name in masks:
        draw_boundaries(wide, name, AXIS_LEFT, TITLE_H)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        out = os.path.join(args.save, f"sv_scan_h{args.hue}.png")
        cv2.imwrite(out, wide)
        print()
        print(f"扫描图已保存到: {out}")

    if not args.no_show:
        cv2.imshow(f"S-V scan H={args.hue} - any key", vis.fit_for_display(wide, 1800))
        print()
        print("已弹出扫描图窗口：先点一下窗口，再按任意键退出。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
