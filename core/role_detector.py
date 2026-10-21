import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


# 计算项目根目录，保证从任意目录运行脚本都能找到模型文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 默认 YOLO 模型路径。
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "person.pt"

# 根据模型类别名判断阵营。
ROLE_CLASS_MAP = {
    "ct_body": "ct",
    "ct_head": "ct",
    "t_body": "t",
    "t_head": "t",
}


@dataclass
class RoleDetection:
    """保存一次角色检测结果。"""

    role: str
    position: tuple[int, int]
    box: tuple[int, int, int, int]
    confidence: float
    class_name: str

    @property
    def aim_position(self) -> tuple[int, int]:
        """Return a good mouse aim point inside the detection box."""
        x1, y1, x2, y2 = self.box
        center_x = (x1 + x2) // 2
        if self.class_name.endswith("_head"):
            center_y = (y1 + y2) // 2
        else:
            center_y = round(y1 + (y2 - y1) * 0.35)
        return center_x, center_y


class RoleDetector:
    """使用 YOLO 检测当前画面中的游戏角色位置和阵营。"""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.verbose = verbose

        # 加载已经训练好的 YOLO 模型。
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件：{self.model_path}")
        self.model = YOLO(str(self.model_path))

    def detect_all(self, frame: Any) -> list[RoleDetection]:
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )[0]
        # 没有检测框时无法判断阵营。
        if result.boxes is None or len(result.boxes) == 0:
            return []
        detections: list[RoleDetection] = []
        # 遍历所有检测框，只采用 ct_body、ct_head、t_body、t_head。
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(result.names.get(class_id, ""))
            role = ROLE_CLASS_MAP.get(class_name)
            if role is None:
                continue

            box_confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            detection = RoleDetection(
                role=role,
                position=(center_x, center_y),
                box=(x1, y1, x2, y2),
                confidence=box_confidence,
                class_name=class_name,
            )
            detections.append(detection)
        return detections

    def detect(self, frame: Any) -> RoleDetection | None:
        """检测单帧图像，返回置信度最高的角色。"""
        detections = self.detect_all(frame)
        if not detections:
            return None

        best_detection = max(detections, key=lambda detection: detection.confidence)

        # 默认不刷日志，只有 verbose=True 时输出一行简洁结果。
        if self.verbose and best_detection is not None:
            print(f"阵营：{best_detection.role}，位置：{best_detection.position}")

        return best_detection

    def detect_enemy(self, frame: Any, own_role: str) -> RoleDetection | None:
        """检测敌方角色；own_role 为 ct 时找 t，own_role 为 t 时找 ct。"""
        enemy_role = "t" if own_role == "ct" else "ct"
        enemies = [detection for detection in self.detect_all(frame) if detection.role == enemy_role]
        if not enemies:
            return None

        # 优先头部框，其次置信度。
        return max(
            enemies,
            key=lambda detection: (
                1 if detection.class_name.endswith("_head") else 0,
                detection.confidence,
            ),
        )

    def detect_role(self, frame: Any) -> str | None:
        """检测单帧图像，并只返回 ct、t 或 None。"""
        detection = self.detect(frame)
        if detection is None:
            return None
        return detection.role


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便使用单张图片测试。"""
    parser = argparse.ArgumentParser(description="使用 YOLO 检测游戏角色阵营。")
    parser.add_argument("--image", required=True, help="用于测试的图片路径。")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="YOLO 模型路径，默认 models/person.pt。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="检测置信度阈值，默认 0.25。",
    )
    return parser.parse_args()


def main() -> None:
    """命令行测试入口。"""
    args = parse_args()

    # 读取测试图片。
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"找不到图片：{args.image}")

    # 检测阵营并输出简洁结果。
    detector = RoleDetector(Path(args.model), confidence=args.conf)
    role = detector.detect_role(image)
    print(role if role is not None else "None")


if __name__ == "__main__":
    main()
