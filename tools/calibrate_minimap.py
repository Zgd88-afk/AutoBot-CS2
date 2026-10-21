import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# 计算项目根目录，保证配置和调试截图保存到固定位置。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs" / "minimap.json"
DEBUG_SCREENSHOT = PROJECT_ROOT / "runs" / "debug" / "minimap_calibrate_screen.png"

# OpenCV 窗口名称。
WINDOW_NAME = "calibrate minimap"

# 保存鼠标拖拽状态。
dragging = False
start_point: tuple[int, int] | None = None
end_point: tuple[int, int] | None = None
frame: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="框选游戏小地图区域。")
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="截图前等待秒数，方便切回游戏，默认 5 秒。",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="要截图的显示器编号，默认 1。",
    )
    return parser.parse_args()


def capture_screen(monitor_index: int = 1) -> np.ndarray:
    """截取指定显示器画面。"""
    try:
        import mss
    except ImportError as error:
        raise ImportError("Missing dependency: pip install mss") from error

    with mss.mss() as capture:
        monitors = capture.monitors
        if monitor_index >= len(monitors):
            raise ValueError(f"monitor 不存在：{monitor_index}，可用范围 1-{len(monitors) - 1}")

        image = np.array(capture.grab(monitors[monitor_index]))

    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def capture_with_delay(delay: int, monitor_index: int) -> np.ndarray:
    """倒计时后截图，给用户时间切回游戏窗口。"""
    for second in range(delay, 0, -1):
        print(f"capture in {second}s, switch to game...")
        time.sleep(1)

    screenshot = capture_screen(monitor_index)
    DEBUG_SCREENSHOT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_SCREENSHOT), screenshot)
    print(f"screenshot saved: {DEBUG_SCREENSHOT.relative_to(PROJECT_ROOT)}")
    return screenshot


def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
    """处理鼠标拖拽框选小地图区域。"""
    global dragging, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_point = (x, y)
        end_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        end_point = (x, y)


def draw_preview() -> np.ndarray:
    """绘制当前框选预览。"""
    assert frame is not None
    preview = frame.copy()

    if start_point is not None and end_point is not None:
        cv2.rectangle(preview, start_point, end_point, (0, 0, 255), 2)

    cv2.putText(
        preview,
        "Drag minimap | s save | c capture again | r reset | q quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def save_config() -> bool:
    """保存小地图 ROI 配置。"""
    if start_point is None or end_point is None:
        print("select minimap first")
        return False

    x1, y1 = start_point
    x2, y2 = end_point
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    if width < 20 or height < 20:
        print("selection too small")
        return False

    config = {
        "roi_left": left,
        "roi_top": top,
        "roi_width": width,
        "roi_height": height,
        "map_x_min": 0,
        "map_y_min": 0,
        "map_x_max": 1000,
        "map_y_max": 1000,
    }

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"saved: {CONFIG_FILE.relative_to(PROJECT_ROOT)}")
    print(f"roi: left={left}, top={top}, width={width}, height={height}")
    return True


def main() -> None:
    """程序入口。"""
    global frame, start_point, end_point

    args = parse_args()
    frame = capture_with_delay(args.delay, args.monitor)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print("drag minimap, press s to save, c to capture again, q to quit")

    while True:
        cv2.imshow(WINDOW_NAME, draw_preview())
        key = cv2.waitKey(30) & 0xFF

        if key == ord("s"):
            if save_config():
                break
        elif key == ord("c"):
            start_point = None
            end_point = None
            frame = capture_with_delay(args.delay, args.monitor)
        elif key == ord("r"):
            start_point = None
            end_point = None
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
