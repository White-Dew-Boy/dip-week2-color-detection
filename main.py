"""WEEK2 - 固定颜色识别（默认黄色）：命令行入口 + 结果输出。

项目各模块分工：
    image_io.py         读图片、解析 .env 里的 IMG_PATH
    color_specs.py      颜色定义与 HSV 阈值注册表（想加颜色改这里）
    mask_postprocess.py 二值化、阈值细化、形态学去噪、连通域
    color_detection.py  识别主流程（掩膜 + 区域 + 统计）
    visualization.py    画轮廓、彩色掩膜、对比拼图
    tuning_app.py       可选的滑块微调窗口
    main.py             本文件：命令行参数、打印统计、保存 / 显示结果

用法：
    python main.py                          # 无参数启动：从图片识别默认颜色并显示结果
    python main.py --camera                 # 改用 0 号摄像头实时识别（按 q / ESC 退出）
    python main.py --camera 1                # 用 1 号摄像头
    python main.py --camera --save out/     # 摄像头模式下按 s 把当前帧存到 out/
    python main.py --color blue             # 改识别蓝色（一次只识别一种颜色）
    python main.py --no-show                # 只识别并打印统计，不弹窗（服务器 / 批处理用）
    python main.py --save out/              # 把 mask / 结果 / 对比图保存到 out/
    python main.py --image assets/yuanshen.png
    python main.py --largest                # 只保留最大色块（识别单个目标时更干净）
    python main.py --mode tune --color yellow   # 微调阈值，按 s 保存（仅图片模式）

图片模式：只弹出一张 4 联拼接图（original | mask | masked | detected），
          **按原始分辨率显示**，按任意键关闭。
摄像头模式：输入源换成摄像头，识别流程与图片模式完全相同；
          窗口按 q / ESC 退出，按 s 把当前帧和识别结果存图，Ctrl+C 也能结束。

可在 .env 或系统环境变量里配置的设置项（系统环境变量优先）：
    IMG_PATH             默认读哪张图
    DEFAULT_COLOR        默认识别的颜色，默认 yellow（可写 blue / green / red）
    MAX_LISTED_REGIONS   统计信息里最多列出几个区域，默认 5
    DISPLAY_MAX_WIDTH    显示窗口最大宽度（px）。默认 0 = 不限制，拼接图按原始
                         分辨率显示、画质不变；设成 1600 之类的正数则会缩小
命令行参数（--color 等）会覆盖这里的默认值。
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
from dotenv import load_dotenv
import visualization as vis
from color_detection import detect_color
from color_specs import COLORS, SUPPORTED_COLORS, resolve_color
from image_io import get_image_path, load_image
from tuning_app import run_tuner

# ============================================================
# 可通过环境变量配置的设置项（写在 .env 或直接设为系统环境变量；
# 系统环境变量的优先级高于 .env，两者都没有时用括号里的默认值）
# ============================================================
ENV_DEFAULT_COLOR = "DEFAULT_COLOR"                # 默认识别的颜色（yellow）
ENV_MAX_LISTED_REGIONS = "MAX_LISTED_REGIONS"      # 统计里最多列出几个区域（5）
ENV_DISPLAY_MAX_WIDTH = "DISPLAY_MAX_WIDTH"        # 显示时窗口最大宽度 px（1600）

# 摄像头模式（--camera）的固定设置
CAMERA_FRAME_WIDTH = 640        # 请求的采集宽度（摄像头不支持时自动忽略）
CAMERA_FRAME_HEIGHT = 480       # 请求的采集高度（摄像头不支持时自动忽略）
CAMERA_OUTPUT_EVERY = 30        # 无窗口模式（--no-show）下每隔多少帧打印一次统计
CAMERA_PRINT_INTERVAL = 60      # 有窗口时每隔多少帧打印一次运行状态
CAMERA_WARMUP_FRAMES = 5        # 打开后先丢弃几帧，等自动曝光稳定


def env_int(name: str, default: int) -> int:
    """读整数环境变量；未设置或写得不是整数时用默认值。"""
    raw = os.getenv(name, "").strip()
    try:
        return int(raw)
    except ValueError:
        if raw:
            print(f"提示：环境变量 {name}='{raw}' 不是整数，改用默认值 {default}")
        return default


def env_str(name: str, default: str) -> str:
    """读字符串环境变量；未设置时用默认值。"""
    return os.getenv(name, "").strip() or default


# ============================================================
# 结果图片的组装与保存
# ============================================================
def build_visuals(image, det: dict) -> dict:
    """把识别结果整理成待保存 / 待显示的各张图。"""
    name = det["color"].name
    spec = det["color"]
    overlay = vis.draw_regions(image, det["regions"], bgr=spec.bgr, prefix=f"{name} ")
    mask_bgr = vis.colorize_mask(det["mask"], spec)

    return {
        "name": name,
        "mask": det["mask"],
        "mask_bgr": mask_bgr,
        "result": det["result"],
        "detected": overlay,
        "compare": vis.tile_images(
            [image, mask_bgr, det["result"], overlay],
            titles=["original", f"{name} mask", f"{name} masked", f"{name} detected"]),
    }


def save_visuals(save_dir: str, image_path: str, visuals: dict) -> None:
    """把结果图片写到磁盘，文件名带图片名和颜色名。"""
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    name = visuals["name"]
    for kind in ("mask", "result", "detected"):
        cv2.imwrite(os.path.join(save_dir, f"{stem}_{name}_{kind}.png"), visuals[kind])
    cv2.imwrite(os.path.join(save_dir, f"{stem}_compare.png"), visuals["compare"])
    print(f"结果已保存到  : {os.path.join(save_dir, stem)}_*.png")


def show_visuals(visuals: dict, max_width: int | None) -> None:
    """弹窗显示结果：只显示一张 4 联拼接图（原图 | 掩膜 | 结果 | 标注）。

    默认 max_width 为 None，即按**原始分辨率**显示，画质不损失
    （DISPLAY_MAX_WIDTH 设为正数时才缩小，0 或不设表示不限制）。
    退出方式：先点一下窗口（让焦点在 OpenCV 窗口上，而不是终端），再按任意键关闭。
    """
    hint = " (any key to quit)"
    cv2.imshow(f"compare{hint}", vis.fit_for_display(visuals["compare"], max_width))

    print("已弹出 4 联拼接图（original | mask | masked | detected）："
          "点一下窗口后按任意键退出。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# 统计信息打印
# ============================================================
def print_report(image_path: str, det: dict, max_regions: int) -> None:
    """打印一种颜色的识别统计，方便不同方法 / 不同图片之间对比。"""
    spec, stats = det["color"], det["stats"]
    print(f"颜色        : {spec.name} ({spec.label})")
    print(f"识别阈值    : {spec.describe()}")

    # 自适应细化只动 H/S 区间（多段色相如红色不细化），变了才多打一行
    if len(spec.ranges) == 1:
        h_lo, h_hi = int(det["lower"][0]), int(det["upper"][0])
        s_lo, s_hi = int(det["lower"][1]), int(det["upper"][1])
        if (h_lo, h_hi, s_lo, s_hi) != (*spec.ranges[0], spec.s_min, spec.s_max):
            print(f"细化后阈值  : H[{h_lo},{h_hi}] S[{s_lo},{s_hi}] V[{spec.v_min},{spec.v_max}]")

    print(f"{spec.label}像素    : {stats['pixels']} / {stats['total_pixels']}"
          f"  ({stats['ratio'] * 100:.2f}%)")
    print(f"{spec.label}区域数  : {stats['region_count']}")
    for index, region in enumerate(det["regions"][:max_regions], start=1):
        x, y, w, h = region["bbox"]
        print(f"  #{index}: 面积={region['area']:.0f}px  包围盒=(x={x}, y={y}, w={w}, h={h})")
    if stats["region_count"] > max_regions:
        print(f"  ... 其余 {stats['region_count'] - max_regions} 个区域从略")


def print_frame_stats(frame_index: int, det: dict) -> None:
    """摄像头模式下的单行统计（每帧都刷屏没法看，所以只印关键信息）。"""
    stats = det["stats"]
    print(f"第 {frame_index:5d} 帧  {det['color'].name}: "
          f"{stats['region_count']}区/{stats['ratio'] * 100:.1f}%")


def run_camera(camera_index: int, color_name: str, args, max_regions: int,
               max_width: int) -> int:
    """摄像头实时识别：逐帧调用与图片模式完全相同的 detect_color()。

    检测逻辑（阈值、形态学、区域统计）全部复用现有模块，本函数只负责
    开摄像头、逐帧送进去、显示/保存、以及按键退出。
    """
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        print(f"错误：打不开摄像头 {camera_index}。请检查是否被其它程序占用，"
              f"或用 --camera 1 换一个设备号。")
        return 1

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    for _ in range(CAMERA_WARMUP_FRAMES):        # 丢掉预热帧，等自动曝光稳定
        capture.read()

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头 {camera_index} 已打开：{width}x{height}，识别颜色 {color_name}")
    if args.no_show:
        print(f"无窗口模式：每 {CAMERA_OUTPUT_EVERY} 帧打印一次统计，按 Ctrl+C 结束。")
    else:
        print("窗口中按 q / ESC 退出；按 s 保存当前帧的识别结果。")

    hint = " (q/ESC quit, s save)"
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("错误：读取摄像头画面失败，已停止。")
                break
            frame_index += 1

            det = detect_color(
                frame, color_name,
                adaptive=False if args.no_adaptive else None,
                keep_largest_only=args.largest,
            )
            overlay = vis.draw_regions(frame, det["regions"],
                                       bgr=det["color"].bgr, prefix=f"{color_name} ")

            if args.no_show:
                if frame_index % CAMERA_OUTPUT_EVERY == 0:
                    print_frame_stats(frame_index, det)
            else:
                cv2.imshow(f"camera{hint}", vis.fit_for_display(frame, max_width))
                cv2.imshow(f"detected{hint}", vis.fit_for_display(overlay, max_width))
                cv2.imshow(f"mask{hint}", vis.fit_for_display(det["mask"], max_width))

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    save_dir = args.save or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "out")
                    os.makedirs(save_dir, exist_ok=True)
                    stem = os.path.join(save_dir, f"camera_frame{frame_index:05d}")
                    cv2.imwrite(f"{stem}.png", frame)
                    cv2.imwrite(f"{stem}_detected.png", overlay)
                    print(f"已保存当前帧: {stem}.png / {stem}_detected.png")

                if frame_index % CAMERA_PRINT_INTERVAL == 0:
                    print_frame_stats(frame_index, det)
    except KeyboardInterrupt:
        print()                                  # Ctrl+C 正常退出
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"摄像头已关闭，共处理 {frame_index} 帧。")
    return 0


# ============================================================
# 命令行
# ============================================================
def parse_args(argv=None, default_color: str = "yellow"):
    parser = argparse.ArgumentParser(
        description=f"固定颜色识别（HSV 阈值 + 形态学去噪）；"
                    f"无参数启动时识别 {default_color} 并显示结果图片")
    parser.add_argument("--image", "-i", default=None,
                        help="待处理图片路径（默认取 .env 的 IMG_PATH）；不给 --camera 时用它")
    parser.add_argument("--color", "-c", default=None,
                        help=f"要识别的颜色：{', '.join(SUPPORTED_COLORS)}"
                             f"（默认 {default_color}，由环境变量 {ENV_DEFAULT_COLOR} 决定）")
    parser.add_argument("--camera", type=int, nargs="?", const=0, default=None, metavar="N",
                        help="从摄像头实时读取画面（不写这个参数就从图片读取）。"
                             "只写 --camera 用 0 号摄像头，--camera 1 用 1 号")
    parser.add_argument("--mode", "-m", choices=["fixed", "tune"], default="fixed",
                        help="fixed=固定阈值识别（默认）；tune=滑块微调模式（仅图片模式）")
    parser.add_argument("--save", "-s", default=None, metavar="DIR",
                        help="把 mask / 结果 / 对比图保存到该目录；可与 --no-show 一起用于批处理")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹窗，只识别并打印统计（无图形界面环境用）")
    parser.add_argument("--largest", action="store_true",
                        help="只保留最大色块（识别单个目标时更干净）")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="关闭自适应阈值细化，严格只用固定阈值")
    parser.add_argument("--save-bounds", default=None, metavar="FILE",
                        help="tune 模式下按 s 保存阈值 JSON 的文件名")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv()                      # .env 里的设置项（含下方几个）先读进来
    default_color = env_str(ENV_DEFAULT_COLOR, "yellow")
    args = parse_args(argv, default_color)

    try:
        color_name = resolve_color(args.color, default_color)
    except (KeyError, ValueError) as error:
        sys.exit(f"错误：{error}")

    max_regions = env_int(ENV_MAX_LISTED_REGIONS, 5)
    # 拼接图默认按原始分辨率显示（画质不变）；DISPLAY_MAX_WIDTH 设为正数才限制宽度
    compare_max_width = env_int(ENV_DISPLAY_MAX_WIDTH, 0)
    # 摄像头每帧都缩放，成本较高，仍用固定上限
    max_width = env_int(ENV_DISPLAY_MAX_WIDTH, 1600) or 1600

    # ---------- 摄像头模式：给了 --camera 就走这里 ----------
    if args.camera is not None:
        if args.mode == "tune":
            sys.exit("错误：微调模式需要一张静态图片（用于拖动滑块），"
                     "请不要和 --camera 一起使用。")
        if args.image:
            print(f"提示：同时给了 --image 和 --camera，本次以摄像头 {args.camera} 为准。")
        return run_camera(args.camera, color_name, args, max_regions, max_width)

    # ---------- 图片模式（默认）----------
    try:
        image_path = get_image_path(args.image)
    except FileNotFoundError as error:
        sys.exit(f"错误：{error}")

    image = load_image(image_path)
    if image is None:
        sys.exit(f"错误：无法读取图像，请检查路径是否正确 -> {image_path}")

    if args.mode == "tune":
        spec = COLORS[color_name]
        save_path = args.save_bounds or os.path.join(os.path.dirname(image_path), f"{spec.name}_bounds.json")
        if not os.path.isabs(save_path):
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)
        run_tuner(image, spec, save_path)
        return 0

    det = detect_color(
        image, color_name,
        adaptive=False if args.no_adaptive else None,
        keep_largest_only=args.largest,
    )

    print_report(image_path, det, max_regions=max_regions)

    visuals = build_visuals(image, det)
    if args.save:
        save_visuals(args.save, image_path, visuals)
    # 默认显示拼接图（原分辨率）；只有显式加 --no-show 才跳过
    if not args.no_show:
        show_visuals(visuals, max_width=compare_max_width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
