import json
from pathlib import Path

import cv2
import numpy as np


# 项目路径和数据文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs" / "minimap.json"
NODES_FILE = PROJECT_ROOT / "maps" / "mirage_nodes.json"
SCREENSHOT_FILE = PROJECT_ROOT / "runs" / "debug" / "minimap_calibrate_screen.png"

# 放大显示，避免蓝色玩家点太小点不中。
WINDOW_NAME = "calibrate minimap points"
ZOOM = 3

# 当前点击的小地图原始坐标。
clicked_point: tuple[int, int] | None = None


def load_config() -> dict:
    """读取小地图 ROI 配置。"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError("请先运行 tools/calibrate_minimap.py 框选小地图。")
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_nodes() -> dict[str, list[int]]:
    """读取地图节点，并兼容 spawn/a_site 别名。"""
    with NODES_FILE.open("r", encoding="utf-8") as file:
        raw_nodes = json.load(file)

    nodes: dict[str, list[int]] = {}
    for group_nodes in raw_nodes.values():
        for name, point in group_nodes.items():
            nodes[name] = point

    aliases = {
        "ct_spawn": "ct_spwan",
        "t_spawn": "t_spwan",
        "a_site": "A_site",
    }
    for alias, real_name in aliases.items():
        if real_name in nodes:
            nodes[alias] = nodes[real_name]

    return nodes


def load_minimap(config: dict) -> np.ndarray:
    """读取最近截图并裁出小地图 ROI。"""
    image = cv2.imread(str(SCREENSHOT_FILE))
    if image is None:
        raise FileNotFoundError("找不到截图，请先运行 tools/calibrate_minimap.py。")

    left = int(config["roi_left"])
    top = int(config["roi_top"])
    width = int(config["roi_width"])
    height = int(config["roi_height"])
    return image[top : top + height, left : left + width]


def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
    """记录鼠标点击位置，并从放大坐标换算回小地图原始坐标。"""
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x // ZOOM, y // ZOOM)
        print(f"clicked: {clicked_point}")


def draw_points(minimap: np.ndarray, points: list[dict]) -> np.ndarray:
    """绘制标定点并放大显示。"""
    preview = minimap.copy()

    for point in points:
        x, y = point["minimap"]
        cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            preview,
            point["name"],
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        preview,
        "click blue player / A, type node in console, s save, q quit",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    return cv2.resize(preview, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)


def main() -> None:
    """程序入口。"""
    global clicked_point

    config = load_config()
    nodes = load_nodes()
    minimap = load_minimap(config)
    calibration_points: list[dict] = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("请标定两个点：")
    print("1. 点击中间蓝色/青色玩家点，输入 ct_spwan 或 t_spwan")
    print("2. 点击 A 点位置，输入 A_site")
    print("注意：不是点下面的 CT起始点文字，也不是点红圈；要点蓝色玩家点本身。")
    print("按 s 保存，按 q 退出。")

    while True:
        cv2.imshow(WINDOW_NAME, draw_points(minimap, calibration_points))
        key = cv2.waitKey(30) & 0xFF

        if clicked_point is not None:
            name = input("node name: ").strip()
            if name in nodes:
                calibration_points.append(
                    {
                        "name": name,
                        "minimap": [clicked_point[0], clicked_point[1]],
                        "map": nodes[name],
                    }
                )
                print(f"added: {name} -> {clicked_point} -> {nodes[name]}")
            else:
                print(f"unknown node: {name}")
            clicked_point = None

        if key == ord("s"):
            config["calibration_points"] = calibration_points
            with CONFIG_FILE.open("w", encoding="utf-8") as file:
                json.dump(config, file, ensure_ascii=False, indent=2)
                file.write("\n")
            print(f"saved: {CONFIG_FILE.relative_to(PROJECT_ROOT)}")
            break
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
