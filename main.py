"""WEEK2 - 固定颜色识别（默认黄色）：命令行入口 + 结果输出。

项目各模块分工：
    image_io.py         读图片、解析 .env 里的 IMG_PATH
    color_specs.py      颜色定义与 HSV 阈值注册表（想加颜色改这里）
    mask_postprocess.py 二值化、阈值细化、形态学去噪、连通域
    color_detection.py  识别主流程（掩膜 + 区域 + 统计）
    visualization.py    画轮廓、彩色掩膜、对比拼图
    main.py             本文件：命令行参数、打印统计、保存 / 显示结果

用法：
    python main.py                          # 无参数启动：从图片识别默认颜色并显示结果
    python main.py --color blue             # 改识别蓝色（一次只识别一种颜色）
    python main.py --no-show                # 只识别并打印统计，不弹窗（服务器 / 批处理用）
    python main.py --save out/              # 把 mask / 结果 / 对比图保存到 out/
    python main.py --no-blur                # 消融：关闭中值滤波去噪
    python main.py --no-adaptive            # 消融：关闭自适应阈值细化（只用固定阈值）
    python main.py --no-morph               # 消融：关闭形态学开/闭运算
    python main.py --no-blur --no-adaptive --no-morph   # 三者全关 = 纯固定阈值基线

只弹出一张 4 联拼接图（original | mask | masked | detected），
**按原始分辨率显示**，按任意键关闭。

三个消融开关 --no-blur / --no-adaptive / --no-morph 分别关闭中值滤波去噪、自适应阈值
细化与形态学后处理，任意组合可复现 2^3 = 8 种配置（全关即"只用固定阈值"的基线）。
它们只提供"关闭"方向：对红色这类跨 H=0、区间分两段的颜色，强制开启细化会把两段压成
一段，ColorSpec.adaptive_allowed 正是为此设置的保护，命令行不应绕过。

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
from color_specs import SUPPORTED_COLORS, resolve_color
from image_io import get_image_path, load_image
from mask_postprocess import MEDIAN_BLUR_KSIZE, MORPH_KSIZE

# ============================================================
# 可通过环境变量配置的设置项（写在 .env 或直接设为系统环境变量；
# 系统环境变量的优先级高于 .env，两者都没有时用括号里的默认值）
# ============================================================
ENV_DEFAULT_COLOR = "DEFAULT_COLOR"                # 默认识别的颜色（yellow）
ENV_MAX_LISTED_REGIONS = "MAX_LISTED_REGIONS"      # 统计里最多列出几个区域（5）
ENV_DISPLAY_MAX_WIDTH = "DISPLAY_MAX_WIDTH"        # 显示时窗口最大宽度 px（1600）


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
# 三个待考察模块的开关（逐因子消融实验用）
# ------------------------------------------------------------
# 中值滤波、自适应阈值细化、形态学后处理可各自单独关闭，关闭方式分别是：
#     中值滤波    blur_ksize=1    —— to_hsv()     直接跳过 medianBlur
#     自适应细化  adaptive=False  —— detect_color() 不再调用 refine_bounds()
#     形态学      morph_ksize=1   —— clean_mask()  跳过开/闭运算（含补洞）
# 三个开关相互独立，组合起来正好是 2^3 = 8 种配置，覆盖"基线 -> 完整流程"的每一步。
# 这里只提供"关闭"开关，不提供"强制开启"：自适应细化对色相跨度大的颜色（如红色
# 跨 H=0、区间分两段）开启后会把两段压成一段，ColorSpec.adaptive_allowed 正是为
# 此设置的保护，命令行不应绕过它。
# ============================================================
def factor_switches(args) -> dict:
    """把三个命令行开关翻译成 detect_color() 的关键字参数。"""
    return {
        "adaptive": False if args.no_adaptive else None,
        "blur_ksize": 1 if args.no_blur else MEDIAN_BLUR_KSIZE,
        "morph_ksize": 1 if args.no_morph else MORPH_KSIZE,
    }


def describe_modules(args) -> str:
    """一行说明本次启用了哪些模块，让消融实验的日志能自证配置。"""
    return "中值滤波={} 自适应细化={} 形态学={}".format(
        "关" if args.no_blur else "开",
        "关" if args.no_adaptive else "自动",
        "关" if args.no_morph else "开")



# ============================================================
# 结果图片的组装与保存
# ============================================================
def build_visuals(image, det: dict) -> dict:
    """把识别结果整理成待保存 / 待显示的各张图。

    mask 用原始二值掩膜（黑底白块），result 是只保留目标颜色的原图，
    与 example/main.py 里的 mask / res 两张图一致。
    """
    name = det["color"].name
    spec = det["color"]
    overlay = vis.draw_regions(image, det["regions"], bgr=spec.bgr, prefix=f"{name} ")

    return {
        "name": name,
        "mask": det["mask"],
        "result": det["result"],
        "detected": overlay,
        "compare": vis.tile_images(
            [image, det["mask"], det["result"], overlay],
            titles=["original", f"{name} mask", f"{name} res", f"{name} detected"]),
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
def print_report(image_path: str, det: dict, max_regions: int, modules: str = "") -> None:
    """打印一种颜色的识别统计，方便不同方法 / 不同图片之间对比。

    modules 是本次启用的模块说明（见 describe_modules），打印出来是为了让
    消融实验的日志能自证配置，便于事后核对每一组结果对应哪三个开关。
    """
    spec, stats = det["color"], det["stats"]
    print(f"颜色        : {spec.name} ({spec.label})")
    if modules:
        print(f"启用模块    : {modules}")
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


# ============================================================
# 命令行
# ============================================================
def parse_args(argv=None, default_color: str = "yellow"):
    parser = argparse.ArgumentParser(
        description=f"固定颜色识别（HSV 阈值 + 形态学去噪）；"
                    f"无参数启动时识别 {default_color} 并显示结果图片")
    parser.add_argument("--image", "-i", default=None,
                        help="待处理图片路径（默认取 .env 的 IMG_PATH）")
    parser.add_argument("--color", "-c", default=None,
                        help=f"要识别的颜色：{', '.join(SUPPORTED_COLORS)}"
                             f"（默认 {default_color}，由环境变量 {ENV_DEFAULT_COLOR} 决定）")
    parser.add_argument("--save", "-s", default=None, metavar="DIR",
                        help="把 mask / 结果 / 对比图保存到该目录；可与 --no-show 一起用于批处理")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹窗，只识别并打印统计（无图形界面环境用）")
    # 三个"关闭型"开关：分别对应中值滤波、自适应阈值细化、形态学后处理，
    # 任意组合即可复现逐因子消融的 8 种配置
    parser.add_argument("--no-blur", action="store_true",
                        help="关闭中值滤波去噪（直接用原始 HSV 分割），消融实验用")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="关闭自适应阈值细化，严格只用固定阈值")
    parser.add_argument("--no-morph", action="store_true",
                        help="关闭形态学开/闭运算（不去噪也不补洞），消融实验用")
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
    # ---------- 图片模式（默认）----------
    try:
        image_path = get_image_path(args.image)
    except FileNotFoundError as error:
        sys.exit(f"错误：{error}")

    image = load_image(image_path)
    if image is None:
        sys.exit(f"错误：无法读取图像，请检查路径是否正确 -> {image_path}")

    det = detect_color(image, color_name, **factor_switches(args))

    print_report(image_path, det, max_regions=max_regions,
                 modules=describe_modules(args))

    visuals = build_visuals(image, det)
    if args.save:
        save_visuals(args.save, image_path, visuals)
    # 默认显示拼接图（原分辨率）；只有显式加 --no-show 才跳过
    if not args.no_show:
        show_visuals(visuals, max_width=compare_max_width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
