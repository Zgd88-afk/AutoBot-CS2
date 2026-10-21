import json
from pathlib import Path

import cv2


# 项目路径和文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "configs" / "minimap.json"
SCREENSHOT_FILE = PROJECT_ROOT / "runs" / "debug" / "minimap_calibrate_screen.png"
OUTPUT_FILE = PROJECT_ROOT / "maps" / "mirage_minimap_ref.png"


def main() -> None:
    """从最近一次校准截图中裁出参考小地图。"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError("请先运行 tools/calibrate_minimap.py 保存小地图 ROI。")

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = json.load(file)

    image = cv2.imread(str(SCREENSHOT_FILE))
    if image is None:
        raise FileNotFoundError("找不到校准截图，请先运行 tools/calibrate_minimap.py。")

    left = int(config["roi_left"])
    top = int(config["roi_top"])
    width = int(config["roi_width"])
    height = int(config["roi_height"])
    minimap = image[top : top + height, left : left + width]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_FILE), minimap)
    print(f"saved: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
