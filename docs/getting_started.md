# AutoBot 新手快速入门

> AutoBot 是一个基于**计算机视觉 + A\* 路径规划 + Arduino 硬件**的 CS:GO/CS2 自动游戏机器人系统。
> 它通过实时截屏识别游戏画面，自动规划路径导航到 A 点，并通过 Arduino 模拟键盘鼠标操作游戏。
>
> ⚠️ **合规提示**：本项目属于游戏自动化工具，使用它可能违反游戏用户协议并导致账号封禁。请仅将代码用于学习计算机视觉、路径规划等技术目的，风险自负。

---

## 目录

1. [项目简介与架构](#1-项目简介与架构)
2. [目录结构导览](#2-目录结构导览)
3. [环境准备](#3-环境准备)
4. [快速开始（三步走）](#4-快速开始三步走)
5. [核心工作原理](#5-核心工作原理)
6. [配置与标定](#6-配置与标定)
7. [常见问题 FAQ](#7-常见问题-faq)
8. [进阶指引](#8-进阶指引)

---

## 1. 项目简介与架构

AutoBot 不依赖游戏 API，而是像人一样"看屏幕、动键鼠"：

- **看**：用 `mss` 实时截屏，用 YOLO 模型识别人物阵营（CT/T）和场景地标，用 HSV 颜色检测在左上角小地图（雷达）上定位自己；
- **想**：用 A\* 算法在二值化的 Mirage 地图上规划从当前位置到 A 点的路线；
- **动**：通过串口向 Arduino Leonardo 发送命令，由 Arduino 以 USB HID 设备身份模拟真实的键盘/鼠标操作。

整体数据流：

```
截屏(mss) → 小地图HSV定位玩家 → YOLO识别阵营(CT/T)
    → 启动运动校准 → A*规划路径 → 沿waypoint移动
    → 遇敌：暂停导航 → 瞄准 → 开火（队友防误伤）
    → 到达A点：守点巡逻
    → 卡死检测与恢复
输出：GameController →（dry-run 只打印日志 | Arduino 串口模拟键鼠）
```

主要功能一览：

- 自动阵营识别（YOLO 场景识别 → 出生点推断 → 人物检测，三级兜底）
- 小地图定位（HSV 颜色阈值识别玩家标记，含跳点过滤与朝向估计）
- A\* 路径规划导航，实时避障
- 自动战斗（检测敌人 → 瞄准 → 开火，队友挡线时禁止开火）
- A 点守点模式（左右巡逻 + 扫视）
- 卡死检测与恢复、启动时运动校准
- PyQt5 图形界面控制台

---

## 2. 目录结构导览

克隆仓库后，项目根目录结构如下：

```
autobot/                             ← 仓库根目录（你的工作目录）
├── main.py                       # 主程序入口（约 1200 行），主循环调度
├── requirements.txt              # Python 依赖清单
├── configs\minimap.json          # 小地图 ROI 与标定点配置（唯一的 JSON 配置）
├── core\                         # 8 个核心模块（见下表）
├── gui\main_window.py            # PyQt5 图形界面
├── arduino\auto_bot_controller\auto_bot_controller.ino  # Arduino 固件（C++）
├── maps\                         # Mirage 导航地图（二值图、节点坐标、参考图）
├── paths\                        # 预生成路径：ct_to_a.json / t_to_a.json
├── models\                       # YOLO 权重：person.pt、place.pt
├── tools\                        # 标定/调试工具脚本（6 个）
├── assets\                       # 演示视频与效果截图
└── docs\                         # 文档（本文件、MRD）
```

> 运行时程序会在项目根目录自动创建 `runs\debug\` 输出调试截图。

`core/` 各模块职责：

| 模块 | 职责 |
|---|---|
| `auto_navigation.py` | A\* 路径规划引擎，读取 `maps/mirage_binary.png`（白=可走）和 `mirage_nodes.json`（CT/T 路线点名） |
| `navigator.py` | 实时 A\* 导航 + waypoint 跟随（每帧根据当前位置重新规划） |
| `player_locator.py` | 在小地图 ROI 内用 HSV 阈值找玩家色点（CT 蓝/青、T 黄），含位置稳定器和朝向估计 |
| `minimap_matcher.py` | ORB 特征匹配定位（HSV 失效时的备用方案） |
| `place_detector.py` | YOLO 场景识别（`place.pt`），根据地标推断所在区域/阵营 |
| `role_detector.py` | YOLO 人物检测（`person.pt`），识别 ct_body/ct_head/t_body/t_head |
| `arduino_control.py` | Arduino 串口封装：移动、转向、朝向估计、开火、运动校准 |
| `anti_stuck.py` | 卡死检测与恢复动作（后退/转向/跳跃后重新规划） |

---

## 3. 环境准备

### 3.1 硬件与系统要求

- **操作系统**：Windows 10/11（代码使用了 Windows 专用 API）
- **显示器**：1920×1080 分辨率（配置默认按此标定，其他分辨率需重新标定小地图）
- **Arduino**：Leonardo / Micro / Pro Micro（必须是支持 USB HID 的型号，普通 UNO 不行）
- **Python**：3.10 或更高版本（代码使用了 3.10 语法）

### 3.2 安装 Python 依赖

建议先创建虚拟环境：

```bash
cd autobot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

依赖包括：`opencv-python`、`numpy`、`ultralytics`（YOLO）、`PyQt5`、`pyserial`、`mss`、`Pillow`。

### 3.3 验证模型文件

确认以下文件存在，**缺失时程序会启动报错**：

- `models/person.pt` — 人物检测模型
- `models/place.pt` — 场景识别模型

### 3.4 上传 Arduino 固件（仅正式运行需要）

新手第一次运行可以跳过这一步，直接用 Dry-run 模式（见下节）。

1. 安装 [Arduino IDE](https://www.arduino.cc/en/software)
2. 打开 `arduino\auto_bot_controller\auto_bot_controller.ino`
3. 工具 → 开发板 → 选择 **Arduino Leonardo**（或 Micro / Pro Micro）
4. 选择正确的 COM 端口，点击"上传"
5. 用 USB 线连接电脑，在设备管理器中确认 COM 口号（默认配置为 **COM6**）

固件通过 `Keyboard.h` / `Mouse.h` 库把串口收到的命令（`forward`、`turn:pixels`、`aim:x,y`、`fire`、`jump` 等）转成真实键鼠操作。串口波特率默认 **9600**。

---

## 4. 快速开始（三步走）

### 第 ① 步：Dry-run 模式（新手必做）

不接 Arduino 也能跑，程序只做视觉识别和日志输出，不操作游戏：

```bash
python main.py --verbose
```

观察日志：能否定位到小地图上的玩家、能否识别阵营、规划的路径是否合理。调试截图会输出到 `runs\debug\`。

### 第 ② 步：正式运行（连接 Arduino）

进游戏（ Mirage 地图，CT 或 T 出生点），然后：

```bash
# 标准运行方式
python main.py --arduino --verbose --enable-recovery

# 串口号不是 COM6 时指定
python main.py --arduino --arduino-port COM8 --verbose
```

程序会先做运动校准（自动测量移动速度和初始朝向），然后开始导航到 A 点。

### 第 ③ 步：图形界面方式（可选）

```bash
python -m gui.main_window
```

GUI 内可以启动/停止机器人、选择运行模式（Arduino / Dry-run）、选择阵营、配置串口号等，并实时回显日志。

### 退出方式

按键盘 **ESC** 或 **Ctrl+C** 终止程序。

### 常用参数速查

完整参数见 `python main.py --help`，以下是新手最常用的：

| 参数 | 说明 |
|---|---|
| `--arduino` | 启用 Arduino 控制（不加即 Dry-run） |
| `--arduino-port COM6` | 串口号（默认 COM6） |
| `--role {auto,ct,t}` | 阵营，默认 auto 自动识别 |
| `--disable-combat` | 只导航不开火 |
| `--disable-site-guard` | 到 A 点后不守点 |
| `--slow-test` | 慢速短脉冲移动，便于观察调试 |
| `--verbose` | 输出详细调试日志 |
| `--invert-turn` | 左右转向反了时启用 |
| `--heading-offset 度数` | 朝向角度偏移校准 |
| `--conf 0.25` | YOLO 置信度阈值 |
| `--max-frames N` | 最多运行 N 帧后停止（0=无限，调试好用） |

---

## 5. 核心工作原理

`main.py` 的主循环每帧截屏一次，每 `--detect-interval-frames`（默认 2）帧做一次识别/导航更新：

1. **输入**：`ScreenCapture`（mss 全屏截图）。
2. **定位**：`PlayerLocator` 在小地图 ROI 内用 HSV 颜色阈值找到玩家标记色点，`PositionStabilizer` 过滤单帧大跳变并估计移动朝向。
3. **阵营识别**（`--role auto` 时）：`PlaceDetector`（场景识别）优先 → 出生点距离推断兜底 → `RoleDetector`（人物检测）再兜底，经连续多帧确认后锁定，不再变化。
4. **启动校准**：`MovementCalibrator` 发出短移动脉冲，测量实际移动速度和初始朝向。
5. **导航**：`Navigator` 从当前小地图位置实时 A\* 规划到 A 点（基于 `maps/mirage_binary.png`，白=可走）；位置落在黑区时先恢复到最近白区再规划。
6. **战斗与守点**：
   - `CombatController`：全屏检测到敌人 → 暂停导航 → 鼠标微调瞄准 → 准星进入死区才开火；队友检测框挡住弹道时禁止开火。
   - `SiteGuard`：到达 A 点附近后左右小范围巡逻 + 扫视，遇敌开火。
7. **输出**：所有动作统一经 `GameController` 发出 —— Dry-run 时只打印日志，`--arduino` 时经 pyserial 发送串口命令给 Arduino 固件执行。
8. **防卡死**：`AntiStuck` 检测位置持续不动 → 执行后退/转向/跳跃等恢复动作 → 重新规划路径。

想深入阅读代码，建议按 `main.py` 主循环 → `core/navigator.py` → `core/player_locator.py` → `core/arduino_control.py` 的顺序看。

---

## 6. 配置与标定

项目**没有环境变量、没有 API key**，配置只有两处：

### 6.1 命令行参数

约 60 个 argparse 参数（见第 4 节速查表和 `--help`），覆盖串口、校准、战斗、导航等所有方面。

### 6.2 `configs/minimap.json` —— 新手最大的坑

此文件定义小地图（雷达）在屏幕截图中的区域（ROI）以及"小地图坐标 → 路径地图坐标"的标定点映射：

```json
{
  "roi_left": 26,          // 小地图左上角 X
  "roi_top": 59,           // 小地图左上角 Y
  "roi_width": 275,        // 小地图宽度
  "roi_height": 272,       // 小地图高度
  "map_x_min": 0, "map_y_min": 0, "map_x_max": 1000, "map_y_max": 1000,
  "calibration_points": [  // 标定点：小地图坐标 → 1000×1000 路径地图坐标
    { "name": "ct_spwan", "minimap": [107, 150], "map": [291, 676] }
  ]
}
```

**这些数值与游戏分辨率、HUD 缩放比例强相关。** 换电脑、换分辨率、改 HUD 设置后必须重新标定，否则"找不到小地图"或定位错乱 —— 这是新手最常遇到的问题（见 FAQ Q1）。标定工具见第 8 节。

### 6.3 地图与路径数据

- `maps/mirage_binary.png`：二值化 Mirage 导航地图（白=可通行，黑=障碍），A\* 的依据。
- `maps/mirage_nodes.json`：CT/T 阵营路线的关键点坐标（1000×1000 坐标系）。
- `paths/ct_to_a.json`、`paths/t_to_a.json`：预生成的参考路径。

---

## 7. 常见问题 FAQ

**Q1：日志提示找不到小地图 / 定位完全不对？**
A：`configs/minimap.json` 中的 `roi_*` 值与你的分辨率/HUD 缩放不匹配。用 `tools/calibrate_minimap.py` 重新标定（见第 8 节），或用命令行参数 `--minimap-left/--minimap-top/--minimap-width/--minimap-height` 临时覆盖。

**Q2：Arduino 连接不上？**
A：① 确认固件已上传且开发板是 Leonardo/Micro/Pro Micro（支持 USB HID）；② 在设备管理器确认实际 COM 口号，用 `--arduino-port` 指定；③ 波特率保持默认 9600。

**Q3：YOLO 模型加载失败？**
A：确认 `models/person.pt` 和 `models/place.pt` 存在且完整。这两个文件是必需的，缺失时程序直接报错退出。

**Q4：移动方向不对（前后左右反了）？**
A：用 `--invert-turn` 反转转向，或用 `--heading-offset` 校准朝向偏移。

**Q5：准星瞄不准？**
A：根据游戏内鼠标灵敏度调整 `--mouse-pixels-per-degree`（默认 2.25），以及 `--combat-aim-gain`、`--combat-fire-deadzone`。

**Q6：角色经常卡住不动？**
A：确认已加 `--enable-recovery`；可调 `--stable-frames`（提高定位稳定性）和 `--max-position-jump`（位置跳变容差）。

**Q7：报错 `ImportError: Missing dependency: pip install mss`？**
A：按提示 `pip install mss`，或检查虚拟环境是否已激活。

**Q8：想先看看程序"在想什么"但不想动游戏？**
A：不加 `--arduino` 就是 Dry-run 模式，只输出日志；调试图像输出在 `runs\debug\`。

---

## 8. 进阶指引

### 8.1 标定与调试工具（`tools/`）

全部支持 `python tools/xxx.py --help` 查看用法：

| 脚本 | 用途 |
|---|---|
| `calibrate_minimap.py` | 交互式小地图 ROI 标定（最常用） |
| `calibrate_minimap_points.py` | 选择小地图 → 路径地图的标定点 |
| `validate_minimap_mapping.py` | 验证标定后坐标映射是否准确 |
| `save_minimap_ref.py` | 保存小地图参考图像 |
| `click_get_coord.py` | 点击屏幕获取坐标 |
| `binarize_map.py` | 把地图原图二值化成导航图 |

### 8.2 YOLO 模型训练

仓库内的 `models/` 只包含训练完成的推理权重（`person.pt`、`place.pt`）。训练工程（标注数据集约 10 万帧游戏画面、训练脚本与超参配置）体积较大，未随仓库分发，此处说明其组织方式供参考：

- 场景识别模型为 YOLOv8，共 10 个地标类（t_wall_A、t_walk、t_flower、t_door 等）
- 人物检测模型类别：ct_body / ct_head / t_body / t_head
- 标注格式为 YOLO txt；训练使用 ultralytics 官方工作流

### 8.3 调试输出

运行时的定位调试图、路径可视化 PNG 会写入 `runs\debug\`，排查定位/导航问题时先看这里。

### 8.4 深入阅读路线

1. 根目录 `README.md` — 项目全貌与架构说明
2. `main.py` — 主循环，所有模块的装配与调度
3. `core/auto_navigation.py` — A\* 引擎与路线点定义
4. `arduino/auto_bot_controller/auto_bot_controller.ino` — 串口命令协议与 HID 实现
