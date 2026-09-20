# WEEK2 — 固定颜色识别（OpenCV / HSV 阈值）

用固定的 HSV 阈值识别图片中的指定颜色（默认**黄色**），并输出掩膜、结果图和对比图。
最初只能识别蓝色，现在颜色是参数化的：想加颜色只改 `color_specs.py` 一个文件。

## 项目结构

```
main.py              命令行入口：参数解析、打印统计、保存 / 显示结果
image_io.py          读图片（支持中文路径）、解析 .env 里的 IMG_PATH
color_specs.py       颜色定义与 HSV 阈值注册表（想加颜色改这里）
mask_postprocess.py  二值化、自适应阈值细化、形态学去噪、连通域与轮廓
color_detection.py   识别主流程：掩膜 + 区域列表 + 统计信息
visualization.py     画轮廓标注、彩色掩膜、横向对比拼图
tuning_app.py        可选的滑块微调窗口（按 s 保存阈值）
sv_scan.py           指定 H，扫描 S×V 平面看哪些 (S,V) 能检测到
bgr_histogram.py     BGR 三通道直方图
hs_histogram.py      H-S 二维直方图（标出目标颜色的 H 区间）
assets/              测试图片
materials/           实验课 Assignment
```

数据流：`image_io` 读图 → `color_detection` 调 `mask_postprocess` 做二值化与去噪
（阈值来自 `color_specs`）→ `visualization` 画图 → `main` 打印 / 保存 / 显示。

## 安装与运行

```bash
pip install -r requirements.txt
python main.py                          # 无参数启动：识别黄色并弹窗显示结果
```

## 设置项（环境变量 / .env）

`main.py` 的默认值都从环境变量读取，写在项目根目录的 `.env` 或设为系统环境变量均可。
优先级：**命令行参数 > 系统环境变量 > .env > 代码内置默认值**。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `IMG_PATH` | `assets/cap.png` | 默认读哪张图（相对项目根目录）。**后缀可省略**，见下方说明 |
| `DEFAULT_COLOR` | `yellow` | 默认识别的颜色，可写 `blue` / `yellow` / `green` / `red`（一次只识别一种） |
| `MAX_LISTED_REGIONS` | `5` | 统计信息里最多列出几个区域，超出只显示数量 |
| `DISPLAY_MAX_WIDTH` | `0`（不限制） | 显示窗口的最大宽度（px）。`0` = 按**原始分辨率**显示，画质不变；设成正数（如 `1600`）则自动缩小 |

> 图片模式只弹出**一张 4 联拼接图**（original | mask | masked | detected），
> 默认按原始分辨率显示，**画质与保存的 PNG 完全一致**。
> 拼接图总宽 ≈ 单格宽 × 4，所以大图（如 910px 宽的 `night.jpg`）窗口会达到 2744px；
> 若超出屏幕，把 `DISPLAY_MAX_WIDTH` 设成 `1600` 让它自动缩小（代价是画质下降）。

例：想让默认识别蓝色、并只看前 3 个区域，把 `.env` 改成

```ini
DEFAULT_COLOR = "blue"
MAX_LISTED_REGIONS = 3
```

注意：`.env` 需保存为 **UTF-8 无 BOM**，否则 python-dotenv 会报解码错误。
整数变量写错（如 `MAX_LISTED_REGIONS = abc`）会提示一行并回退默认值，不会崩溃。

### IMG_PATH 可以省略后缀

图片有的是 `.jpg`、有的是 `.png`，写 `IMG_PATH` 时不必记住后缀，程序会自动找到：

```ini
IMG_PATH = "assets/lina"      # → assets/lina.jpg
IMG_PATH = "flowers"          # → assets/flowers.PNG
IMG_PATH = "cap"              # → assets/cap.png
IMG_PATH = "assets/cap.jpg"   # 后缀写错也能找到 assets/cap.png
```

匹配规则（按顺序，命中即用，所以**写全路径时行为与以前完全一致**）：

1. 原样路径存在 → 直接用
2. 依次补上 `.png` `.jpg` `.jpeg` `.bmp` `.webp` `.tif` `.tiff`
3. 在所在目录（含子目录）按文件名匹配，**忽略大小写**（覆盖 `.JPG` / `.PNG` 这类写法）
4. 在整个项目目录里搜一遍（所以只写文件名、不写 `assets/` 也能找到）

找不到时会在报错里列出项目里所有可用图片，方便直接复制名字。
命令行参数 `--image` / `-i` 同样支持省略后缀。

## 常用参数

| 命令 | 作用 |
| --- | --- |
| `python main.py` | 从图片识别默认颜色，弹出 4 联拼接图（原分辨率，按任意键关闭） |
| `python main.py --camera` | **改用 0 号摄像头实时识别**（`q`/`ESC` 退出，`s` 存当前帧） |
| `python main.py --camera 1` | 用 1 号摄像头 |
| `python main.py --camera --save out/` | 摄像头模式下按 `s` 把帧和结果存到 `out/` |
| `python main.py --color blue` | 改识别蓝色（覆盖 `DEFAULT_COLOR`） |
| `python main.py -i assets/yuanshen.png` | 指定图片（默认取 `.env` 的 `IMG_PATH`） |
| `python main.py --no-show` | 不弹窗，只打印统计（无图形界面环境） |
| `python main.py --save out/` | 保存 mask / 结果 / 对比图到 `out/` |
| `python main.py --largest` | 只保留最大色块（识别单个目标时更干净） |
| `python main.py --no-adaptive` | 关闭自适应细化，严格只用固定阈值 |
| `python main.py --mode tune -c yellow` | 滑块微调阈值，按 `s` 保存 JSON、`q` 退出（仅图片模式） |

> **一次只识别一种颜色**：`--color` 接受单个颜色名（`blue` / `yellow` / `green` / `red`），
> 不支持逗号分隔或 `all`。想换颜色再跑一次即可。

### 摄像头模式（`--camera [N]`）

**不写 `--camera` 时行为与以前完全一致**（从图片读取）；加了它则输入源换成摄像头，
检测流程（阈值、形态学、区域统计）**一行代码都没变**，只是把"一张图"换成"每一帧"。

```powershell
python main.py --camera                       # 0 号摄像头，识别 .env 的 DEFAULT_COLOR
python main.py --camera 1 --color blue        # 1 号摄像头，识别蓝色
python main.py --camera --no-show             # 无窗口：每 30 帧打印一次统计，Ctrl+C 结束
python main.py --camera --save out/           # 按 s 时把帧和识别结果存到 out/
```

| 按键 | 作用 |
| --- | --- |
| `q` / `ESC` | 退出 |
| `s` | 保存当前帧：`camera_frame00003.png` + `camera_frame00003_detected.png` |
| `Ctrl+C` | 也能正常结束（会释放摄像头） |

其他说明：采集分辨率请求 640×480（摄像头不支持时自动忽略）；打开后丢弃 5 帧预热，
等自动曝光稳定；打不开摄像头会提示检查占用或换设备号；`--camera` 与 `--mode tune`
不能同时用（微调需要一张静态图片）。


直方图脚本同样支持指定图片，`hs_histogram.py` 还能指定要标注的颜色：

```bash
python bgr_histogram.py
python hs_histogram.py assets/yuanshen.png blue
```

## 查看"当前能识别哪些颜色"

`main.py` 用 `--color` 每次识别一种颜色；想知道**每种颜色的阈值覆盖了色空间的哪一片**，
用 `sv_scan.py` —— 固定一个色相，扫描 S×V 平面看哪些 (S, V) 组合能被检测到：

```bash
python sv_scan.py --hue 28                            # 弹窗看色卡
python sv_scan.py --hue 28 --save out/                # 存 out/sv_scan_h28.png
python sv_scan.py --hue 28 --no-show                  # 只打印统计 / 存图
```

`sv_scan.py` 输出一张 S-V 色卡（横轴 Saturation 0~255，纵轴 V 255→0），
显示每个 (S, V) 组合在该 H 下的**真实颜色**（右上最艳最亮，往左下逐渐发灰变暗），
检测不到的格子留黑；右侧图例列出四种已注册颜色在该 H 下的占比，色卡上还用本色虚线
画出各自的 S 下限、V 下限。该 H 下的 S/V 门槛也会打印在控制台，
例如 H=28 时：黄色 S ≥ 60、V ≥ 80。
判定复用 `threshold_hsv()`，所以色卡显示的范围和程序实际识别范围完全一致。

```bash
python sv_scan.py --hue 28              # H=28（黄色区）
python sv_scan.py --hue 110             # H=110（蓝色区）
python sv_scan.py --hue 0               # H=0（红色，验证跨 0 环绕）
python sv_scan.py --hue 85              # 边界 H=85：绿的上限 = 蓝的下限，两者同时命中
python sv_scan.py --hue 37              # 空隙 H，整张全黑（谁都检测不到）
python sv_scan.py --hue 28 --save out/  # 存 out/sv_scan_h28.png
python sv_scan.py --hue 28 --no-show    # 只打印统计 / 存图，不弹窗
```

## 各颜色的固定阈值

HSV 取值范围：H ∈ [0, 179]，S ∈ [0, 255]，V ∈ [0, 255]。

| 颜色 | H 范围 | S | V | 说明 |
| --- | --- | --- | --- | --- |
| yellow | 18 ~ 35 | ≥ 60 | ≥ 80 | 低于 18 偏橙、高于 35 偏黄绿，均排除 |
| blue | 85 ~ 128 | ≥ 40 | ≥ 50 | 覆盖青蓝（≈90）到纯蓝（≈120），排除紫色（≥130） |
| green | 40 ~ 85 | ≥ 50 | ≥ 50 | — |
| red | 0 ~ 10 ∪ 170 ~ 179 | ≥ 70 | ≥ 70 | 色相在 H=0 处环绕，两段取并集 |

阈值是扫描样例图统计后定下来的：S 下限用于甩掉白色桌面 / 灰色背景，
V 下限用于甩掉过暗像素。识别时还会用粗筛结果的 H/S 中位数把阈值**收紧**
（只会更严格，不会更宽松），从而在不同光照下更稳定；`--no-adaptive` 可关掉这一步。

## 加一种新颜色

在 `color_specs.py` 的 `COLORS` 里加一条即可，其余模块无需改动：

```python
"orange": ColorSpec(
    name="orange", label="橙色",
    ranges=((11, 22),), s_min=100, v_min=100,
    bgr=_hex_to_bgr("#FF8C00"),
),
```

## 已知样例结果（assets/flowers.png）

| 颜色 | 区域数 | 像素占比 |
| --- | --- | --- |
| yellow | 6 | 7.57% |
| blue | 5 | 6.06% |
| red | 5 | 7.45% |
| green | 0 | 0.03% |
