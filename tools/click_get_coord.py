import json
from pathlib import Path
from typing import Any

import cv2


# 计算项目根目录，保证从任意目录运行脚本都能找到 maps 目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 定义地图图片和节点文件路径。
MAP_IMAGE = PROJECT_ROOT / "maps" / "mirage_binary.png"
NODES_FILE = PROJECT_ROOT / "maps" / "mirage_nodes.json"

# 定义 OpenCV 窗口名称。
WINDOW_NAME = "mirage coord picker"

# 定义模型类别名称，点击标注时建议使用这些名字。
CLASS_NAMES = {
    0: "t_wall_A",
    1: "t_walk",
    2: "t_flower",
    3: "t_door",
    4: "t_steer",
    5: "t_box",
    6: "t_box_A",
    7: "ct_board",
    8: "ct_pavilion",
    9: "ct_pipe",
    10: "t_spwan",
    11: "ct_spwan",
    12: "A_site"
}

# 提取建议点名列表，控制台会显示这些名字。
SUGGESTED_NAMES = list(CLASS_NAMES.values())

# 保存程序运行时的节点数据。
nodes: dict[str, dict[str, list[int]]] = {
    "ct": {},
    "t": {},
    "landmarks": {},
}

# 保存原始地图和当前显示图。
base_image: Any = None
display_image: Any = None


def load_nodes() -> dict[str, dict[str, list[int]]]:
    """读取已有节点文件，如果不存在就返回空结构。"""
    if not NODES_FILE.exists():
        return {"ct": {}, "t": {}, "landmarks": {}}

    with NODES_FILE.open("r", encoding="utf-8") as file:
        loaded_nodes = json.load(file)

    # 补齐必要分组，避免旧文件缺少某个键时报错。
    for group_name in ("ct", "t", "landmarks"):
        loaded_nodes.setdefault(group_name, {})

    return loaded_nodes


def save_nodes() -> None:
    """保存节点数据到 JSON 文件。"""
    NODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NODES_FILE.open("w", encoding="utf-8") as file:
        json.dump(nodes, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"已保存：{NODES_FILE.relative_to(PROJECT_ROOT)}")


def set_node(name: str, x: int, y: int) -> None:
    """按照类别名前缀把坐标写入对应分组。"""
    coordinate = [x, y]

    # t_ 开头的类别保存到 t 分组。
    if name.startswith("t_"):
        nodes["t"][name] = coordinate

    # ct_ 开头的类别保存到 ct 分组。
    if name.startswith("ct_"):
        nodes["ct"][name] = coordinate

    # 所有标注点都同步保存到 landmarks，方便统一读取。
    nodes["landmarks"][name] = coordinate


def draw_nodes() -> None:
    """在地图上绘制已经标注过的点。"""
    global display_image

    # 每次重绘都从原始地图复制一份，避免文字和圆点反复叠加。
    display_image = base_image.copy()

    # 合并所有分组里的点，同名点只画一次。
    merged_nodes: dict[str, list[int]] = {}
    for group_nodes in nodes.values():
        merged_nodes.update(group_nodes)

    # 逐个绘制点和点名。
    for name, coordinate in merged_nodes.items():
        x, y = int(coordinate[0]), int(coordinate[1])
        cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            display_image,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
    """处理鼠标点击事件。"""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    print(f"坐标：{x}, {y}")
    name = input("类别名：").strip()
    if not name:
        print("已跳过")
        return

    set_node(name, x, y)
    draw_nodes()
    cv2.imshow(WINDOW_NAME, display_image)
    save_nodes()
    print(f"已标注：{name} = [{x}, {y}]")


def main() -> None:
    """程序入口。"""
    global nodes, base_image

    # 读取二值地图图片。
    loaded_image = cv2.imread(str(MAP_IMAGE), cv2.IMREAD_COLOR)
    if loaded_image is None:
        raise FileNotFoundError(f"找不到地图图片：{MAP_IMAGE}")

    # 读取已有节点并初始化显示图。
    nodes = load_nodes()
    base_image = loaded_image
    draw_nodes()

    print("左键点击标点，输入类别名。按 s 保存，按 q 退出。")
    print("建议类别：" + ", ".join(SUGGESTED_NAMES))

    # 创建窗口并绑定鼠标回调。
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # 主循环负责刷新窗口和处理键盘操作。
    while True:
        cv2.imshow(WINDOW_NAME, display_image)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("s"), ord("S")):
            save_nodes()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
