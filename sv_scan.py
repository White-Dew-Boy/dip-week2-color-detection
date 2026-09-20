"""S-V 检测扫描图：指定一个色相 H，扫描 S × V 平面，看哪些 (S, V) 能被检测到。

用法：
    python sv_scan.py --hue 28                  # H=28（黄色区）：实际能扫到的 S-V 色卡
    python sv_scan.py --hue 110                 # H=110（蓝色区）
    python sv_scan.py --hue 0                   # H=0（红色，色相跨 0 环绕）
    python sv_scan.py --hue 28 --no-optimize    # 只看固定阈值的理论范围
    python sv_scan.py --hue 28 --compare        # 理论 vs 实际，并排两张
    python sv_scan.py --hue 28 --save out/      # 存成 PNG
    python sv_scan.py --hue 28 --no-show        # 只打印统计 / 存图，不弹窗

输出内容：
    一张 S-V 色卡（横轴 Saturation 0~255，纵轴 V 255~0），只显示**实际能检测到**
    的格子的真实颜色，检测不到的留黑；右侧图例给出每种颜色优化前后的占比。

"实际"= 把 S-V 扫描图当成普通图片，走一遍与 main.py 完全相同的识别流程：
    中值滤波 -> 固定阈值 -> 自适应阈值细化 -> 形态学开/闭运算 -> 区域过滤
其中影响最大的是**自适应阈值细化**：它会按粗筛像素的 S 中位数把范围收紧，
所以实际色卡会比"固定阈值"的理论色卡小一圈。用 --no-optimize 可切回理论色卡。

判定与识别共用同一套代码（color_detection.detect_color），所以色卡上显示
"能检测到"的格子，程序在该颜色下就一定能识别到。红色因色相跨 0 环绕、区间分两段，
不做自适应细化，因此它的理论值与实际值一致。
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

import visualization as vis
from color_detection import detect_color
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
    """在固定 H 的 S-V 平面上扫描所有已注册颜色，返回 (真实色域, 粗筛掩膜字典)。

    粗筛掩膜 = 只用固定阈值的理论范围，供 --no-optimize / --compare 使用。
    """
    hsv = build_hsv_grid(h)
    raw = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    masks = {name: threshold_hsv(hsv, spec) > 0 for name, spec in COLORS.items()}
    return raw, masks


def scan_optimized(raw: np.ndarray) -> dict[str, dict]:
    """把 S-V 扫描图当成普通图片，交给识别主流程跑一遍。

    用的是与 main.py 完全相同的优化流程（medianBlur -> 固定阈值 -> 自适应细化
    -> 形态学 -> 区域过滤），所以得到的才是"实际能扫到"的范围。
    """
    return {name: detect_color(raw, name) for name in COLORS}


def build_card(raw: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    """色卡：能检测到的格子显示该 (S,V) 的真实颜色，检测不到的留黑。"""
    claimed = np.zeros(raw.shape[:2], dtype=bool)
    for hit in masks.values():
        claimed |= hit
    return np.where(claimed[..., None], raw, BLANK_COLOR).astype(np.uint8)


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


def draw_legend(panel: np.ndarray, left: int, top: int, cards: dict, h: int,
                optimized: bool = True) -> None:
    """右侧图例：每种颜色在该 H 下的可检测占比（优化模式下同时给出理论占比）。"""
    total = (SAT_MAX + 1) * (VAL_MAX + 1)
    y = top + 26
    for name, spec in COLORS.items():
        actual = int(cards[name]["actual"].sum())
        theory = int(cards[name]["theory"].sum())
        x = left + 14
        cv2.rectangle(panel, (x, y - 13), (x + 18, y + 5), spec.bgr, -1)
        vis.draw_label(panel, name, (x + 26, y), (255, 255, 255), 0.5)

        if optimized:
            vis.draw_label(panel, f"actual {actual * 100.0 / total:.2f}% of S-V plane",
                           (x + 26, y + 21), (255, 255, 255), 0.42)
            if actual != theory:
                vis.draw_label(panel, f"was {theory * 100.0 / total:.2f}% before optimize",
                               (x + 26, y + 40), (165, 165, 165), 0.42)
            else:
                vis.draw_label(panel, "(no change: not optimizable)",
                               (x + 26, y + 40), (165, 165, 165), 0.42)
        else:
            vis.draw_label(panel, f"threshold only, {actual * 100.0 / total:.2f}%",
                           (x + 26, y + 21), (255, 255, 255), 0.42)

        in_range = any(lo <= h <= hi for lo, hi in spec.ranges)
        note = f"H={h} in {spec.ranges[0][0]}~{spec.ranges[-1][1]}" if in_range \
            else f"H={h} outside this color"
        vis.draw_label(panel, note, (x + 26, y + 59),
                       spec.bgr if in_range else (150, 150, 150), 0.4)
        y += 88

    x = left + 14
    cv2.rectangle(panel, (x, y - 13), (x + 18, y + 5), BLANK_COLOR, -1)
    cv2.rectangle(panel, (x, y - 13), (x + 18, y + 5), (120, 120, 120), 1)
    vis.draw_label(panel, "not detectable (blank)", (x + 26, y), (255, 255, 255), 0.5)
    claimed = np.zeros(next(iter(cards.values()))["actual"].shape, dtype=bool)
    for item in cards.values():
        claimed |= item["actual"]
    vis.draw_label(panel, f"{100 - 100.0 * int(claimed.sum()) / total:.2f}% of S-V plane",
                   (x + 26, y + 21), (165, 165, 165), 0.42)


# ============================================================
# 3. 文字统计
# ============================================================
def print_summary(h: int, cards: dict, show_actual: bool = True) -> None:
    """打印该 H 下每种颜色的理论 / 实际可检测范围与生效阈值。"""
    total = (SAT_MAX + 1) * (VAL_MAX + 1)
    print(f"扫描切片    : H={h}，S 0~{SAT_MAX} × V 0~{VAL_MAX}，共 {total} 格")
    print(f"参与判定    : {', '.join(cards)}")
    print()
    print(f"{'颜色':<8}{'理论(固定阈值)':>15}{'实际(加优化)':>14}{'削减':>9}"
          f"   实际生效阈值")
    print("-" * 96)
    for name, item in cards.items():
        theory = int(item["theory"].sum())
        actual = int(item["actual"].sum())
        used = f"S[{int(item['lower'][1])},{int(item['upper'][1])}]"
        if len(item["color"].ranges) > 1:          # 红色这种多段色相不做细化
            used = "同固定阈值"
        print(f"{name:<8}{theory * 100.0 / total:>14.2f}%{actual * 100.0 / total:>13.2f}%"
              f"{(actual - theory) * 100.0 / total:>8.2f}%   {used}")

    covered = np.zeros(next(iter(cards.values()))["actual"].shape, dtype=bool)
    for item in cards.values():
        covered |= item["actual"]
    print(f"{'合计':<8}{'':>15}{int(covered.sum()) * 100.0 / total:>13.2f}%"
          f"   (至少一种颜色能检测到)")

    # 实际能检测到的 S / V 取值范围
    print()
    for name, item in cards.items():
        spec = item["color"]
        if not any(lo <= h <= hi for lo, hi in spec.ranges):
            continue
        mask = item["actual"] if show_actual else item["theory"]
        label = "实际" if show_actual else "理论"
        s_ok = np.where(mask.any(axis=0))[0]
        v_ok = np.where(mask.any(axis=1))[0]
        if not len(s_ok):
            print(f"[{name}] H={h} 在色相区间内，但{label}上一格都没有")
            continue
        print(f"[{name}] H={h}：理论 S ≥ {spec.s_min}、V ≥ {spec.v_min}；"
              f"{label} S {int(s_ok.min())}~{int(s_ok.max())}、"
              f"V {VAL_MAX - int(v_ok.max())}~{VAL_MAX - int(v_ok.min())}")


# ============================================================
# 4. 命令行
# ============================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="指定色相 H，扫描 S-V 平面看哪些 (S,V) 能被检测到")
    parser.add_argument("--hue", type=int, default=28, metavar=f"0-{HUE_MAX}",
                        help=f"要扫描的色相 H（默认 28，黄色区），范围 {HUE_MIN}~{HUE_MAX}")
    parser.add_argument("--no-optimize", action="store_true",
                        help="只看固定阈值的理论范围（不加自适应细化与形态学）")
    parser.add_argument("--compare", action="store_true",
                        help="并排输出两张色卡：左=固定阈值，右=加优化后的实际范围")
    parser.add_argument("--save", "-s", default=None, metavar="DIR",
                        help="保存到该目录（文件名含 H 值）")
    parser.add_argument("--no-show", action="store_true", help="不弹窗，只打印统计 / 存图")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not HUE_MIN <= args.hue <= HUE_MAX:
        sys.exit(f"错误：--hue 必须在 {HUE_MIN}~{HUE_MAX} 之间（OpenCV 的 H 范围），"
                 f"收到 {args.hue}")

    raw, theory_masks = scan(args.hue)                  # 真实色域 + 固定阈值掩膜
    optimized = scan_optimized(raw)                     # 完整流程（与 main.py 相同）
    cards = {
        name: {
            "color": det["color"],
            "theory": theory_masks[name],
            "actual": det["mask"] > 0,
            "lower": det["lower"],
            "upper": det["upper"],
        }
        for name, det in optimized.items()
    }
    print_summary(args.hue, cards, show_actual=not args.no_optimize)

    def one_card(mask_key: str, title: str, subtitle: bool) -> np.ndarray:
        """生成一张"色卡 + 图例"的面板。"""
        panel = make_panel(build_card(raw, {n: c[mask_key] for n, c in cards.items()}), title)
        wide = np.full((panel.shape[0], panel.shape[1] + LEGEND_W, 3), PANEL_BG, dtype=np.uint8)
        wide[:, :panel.shape[1]] = panel
        show = {n: {"actual": c[mask_key], "theory": c["theory"]} for n, c in cards.items()}
        draw_legend(wide, panel.shape[1], TITLE_H, show, args.hue, optimized=subtitle)
        for name in cards:
            draw_boundaries(wide, name, AXIS_LEFT, TITLE_H)
        return wide

    title = f"detectable colors at H={args.hue}"
    if args.compare:
        left = one_card("theory", f"threshold only (H={args.hue})", subtitle=False)
        right = one_card("actual", f"after optimize (H={args.hue})", subtitle=True)
        sheet = vis.tile_images([left, right], gap=8, background=PANEL_BG)
    elif args.no_optimize:
        sheet = one_card("theory", title, subtitle=False)
    else:
        sheet = one_card("actual", title, subtitle=True)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        suffix = "_compare" if args.compare else ("_theory" if args.no_optimize else "")
        out = os.path.join(args.save, f"sv_scan_h{args.hue}{suffix}.png")
        cv2.imwrite(out, sheet)
        print()
        print(f"扫描图已保存到: {out}")

    if not args.no_show:
        cv2.imshow(f"S-V scan H={args.hue} - any key", vis.fit_for_display(sheet, 1800))
        print()
        print("已弹出扫描图窗口：先点一下窗口，再按任意键退出。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
