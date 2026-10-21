import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "place.pt"


@dataclass
class PlaceDetection:
    """保存一次环境点检测结果。"""

    role: str
    place_name: str
    box: tuple[int, int, int, int]
    confidence: float


class PlaceDetector:
    """使用环境识别模型判断当前更像 CT 路线还是 T 路线。"""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.verbose = verbose

        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到环境模型文件：{self.model_path}")
        self.model = YOLO(str(self.model_path))

    def detect(self, frame: Any) -> PlaceDetection | None:
        """检测单帧环境，并返回最可信的 CT/T 环境点。"""
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        best_detection: PlaceDetection | None = None
        for box in result.boxes:
            class_id = int(box.cls[0])
            place_name = str(result.names.get(class_id, ""))
            role = role_from_place(place_name)
            if role is None:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detection = PlaceDetection(
                role=role,
                place_name=place_name,
                box=(x1, y1, x2, y2),
                confidence=confidence,
            )

            if best_detection is None or detection.confidence > best_detection.confidence:
                best_detection = detection

        if self.verbose and best_detection is not None:
            print(
                f"环境：{best_detection.place_name}，"
                f"阵营：{best_detection.role}，"
                f"置信度：{best_detection.confidence:.2f}"
            )

        return best_detection

    def detect_role(self, frame: Any) -> str | None:
        """检测单帧环境，并只返回 ct、t 或 None。"""
        detection = self.detect(frame)
        if detection is None:
            return None
        return detection.role


def role_from_place(place_name: str) -> str | None:
    """按环境类别名前缀映射阵营。"""
    if place_name.startswith("ct_"):
        return "ct"
    if place_name.startswith("t_"):
        return "t"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 YOLO 检测环境点并推断阵营。")
    parser.add_argument("--image", required=True, help="用于测试的图片路径。")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="YOLO 环境模型路径，默认 models/place.pt。",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"找不到图片：{args.image}")

    detector = PlaceDetector(Path(args.model), confidence=args.conf, verbose=True)
    role = detector.detect_role(image)
    print(role if role is not None else "None")


if __name__ == "__main__":
    main()
