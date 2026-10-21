import argparse
from pathlib import Path

import cv2
import numpy as np


# 计算项目根目录，保证无论从哪个目录运行脚本都能找到 maps 目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 定义输入地图、输出二值图和调试图路径。
INPUT_IMAGE = PROJECT_ROOT / "maps" / "mirage_map.jpg"
OUTPUT_IMAGE = PROJECT_ROOT / "maps" / "mirage_binary.png"
DEBUG_IMAGE = PROJECT_ROOT / "runs" / "debug" / "mirage_binary_debug.png"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="将地图图片转换为黑白二值图。")
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="二值化阈值，范围 0-255，默认 127。",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="反转黑白结果：白色变黑色，黑色变白色。",
    )
    return parser.parse_args()


def validate_threshold(threshold: int) -> None:
    """检查阈值是否在 OpenCV 支持的范围内。"""
    if not 0 <= threshold <= 255:
        raise ValueError("阈值必须在 0 到 255 之间。")


def binarize_map(threshold: int, invert: bool) -> np.ndarray:
    """读取地图图片，转换灰度图，并根据阈值生成二值图。"""
    # 读取原始地图图片。
    image = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"找不到输入图片：{INPUT_IMAGE}")

    # 转换为灰度图，为阈值二值化做准备。
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 默认规则：高于阈值的像素变白色可走区域，低于阈值的像素变黑色障碍。
    threshold_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary_image = cv2.threshold(gray_image, threshold, 255, threshold_mode)

    return binary_image


def save_images(binary_image: np.ndarray) -> None:
    """保存正式输出图和调试图。"""
    # 确保输出目录存在。
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_IMAGE.parent.mkdir(parents=True, exist_ok=True)

    # 保存二值化结果，白色表示可走区域，黑色表示障碍。
    if not cv2.imwrite(str(OUTPUT_IMAGE), binary_image):
        raise OSError(f"保存失败：{OUTPUT_IMAGE}")

    # 调试图目前保存同一份二值化结果，方便后续查看处理效果。
    if not cv2.imwrite(str(DEBUG_IMAGE), binary_image):
        raise OSError(f"保存失败：{DEBUG_IMAGE}")


def main() -> None:
    """程序入口。"""
    args = parse_args()
    validate_threshold(args.threshold)

    binary_image = binarize_map(args.threshold, args.invert)
    save_images(binary_image)

    # 控制台只输出必要信息，避免干扰后续脚本调用。
    print(f"完成：{OUTPUT_IMAGE.relative_to(PROJECT_ROOT)}")
    print(f"调试图：{DEBUG_IMAGE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
