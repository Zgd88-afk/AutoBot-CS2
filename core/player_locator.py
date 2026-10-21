import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 计算项目根目录，方便保存定位调试图。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_IMAGE = PROJECT_ROOT / "runs" / "debug" / "player_locator_debug.png"

# 像素坐标点，格式为 (x, y)。
Point = tuple[int, int]


@dataclass
class MinimapConfig:
    """小地图裁剪和坐标映射配置。"""

    # 小地图在游戏截图中的位置和大小。
    roi_left: int = 0
    roi_top: int = 0
    roi_width: int = 320
    roi_height: int = 320

    # 小地图 ROI 映射到路径地图坐标的范围。
    map_x_min: int = 0
    map_y_min: int = 0
    map_x_max: int = 1000
    map_y_max: int = 1000

    # 玩家标记颜色阈值：优先识别蓝色/青色玩家点。
    blue_hue_min: int = 45
    blue_hue_max: int = 110
    blue_saturation_min: int = 80
    blue_value_min: int = 120

    # T 阵营玩家点是黄色时使用这组阈值；仍然只识别小圆点，不识别白色亮点。
    yellow_hue_min: int = 18
    yellow_hue_max: int = 40
    yellow_saturation_min: int = 80
    yellow_value_min: int = 120

    # 黄色朝向箭头阈值，用于识别人物面朝方向。
    arrow_hue_min: int = 18
    arrow_hue_max: int = 40
    arrow_saturation_min: int = 80
    arrow_value_min: int = 120
    arrow_search_radius: int = 45

    # 备用阈值：找不到蓝点时，再识别低饱和、高亮度的白色/浅色标记。
    marker_saturation_max: int = 80
    marker_value_min: int = 180
    min_area: int = 4
    max_area: int = 500

    # 过滤贴边亮点，减少小地图边框、字母和 HUD 的干扰。
    edge_margin_ratio: float = 0.08

    # 必须先确认小地图圆形雷达存在；防止桌面、终端、IDE 被误当成游戏画面。
    require_minimap_circle: bool = True

    # 玩家点应该是小而紧凑的色块，过滤桌面文字和 UI 图标。
    min_marker_width: int = 3
    min_marker_height: int = 3
    max_marker_width: int = 24
    max_marker_height: int = 24
    max_marker_aspect_ratio: float = 2.4
    min_marker_fill_ratio: float = 0.18

    # 单帧最大允许跳变距离，超过则沿用上一帧位置。
    max_jump_distance: int = 80

    # 手动标定点，格式为 [{"minimap": [x, y], "map": [x, y]}]。
    calibration_points: list[dict[str, Any]] | None = None


@dataclass
class PlayerLocation:
    """保存小地图定位结果。"""

    position: Point
    minimap_position: Point
    confidence: float
    heading_degrees: float | None = None


class PlayerLocator:
    """不使用 YOLO，通过小地图识别玩家坐标。"""

    def __init__(self, config: MinimapConfig | None = None, verbose: bool = False) -> None:
        # 保存配置、日志开关和上一帧候选点。
        self.config = config or MinimapConfig()
        self.verbose = verbose
        self.last_marker: Point | None = None
        self.active_bounds: tuple[int, int, int, int] | None = None
        self.active_circle: tuple[int, int, int] | None = None
        self.last_active_bounds: tuple[int, int, int, int] | None = None

    def locate(self, frame: np.ndarray, role: str | None = None) -> PlayerLocation | None:
        """从整帧截图中定位玩家，并返回地图坐标。"""
        minimap = self._crop_minimap(frame)
        self.active_bounds = self._detect_active_bounds(minimap)
        if self.active_bounds is None:
            self.last_marker = None
            if self.verbose:
                self._save_debug_image(minimap, [], None)
            return None
        self.last_active_bounds = self.active_bounds

        marker, heading_degrees, candidates = self._find_player_marker(minimap, role)
        if marker is None:
            if self.verbose:
                self._save_debug_image(minimap, candidates, None)
            return None

        # 小地图现在固定方向，不再使用 ORB/SIFT 做旋转匹配。
        # 直接把玩家点按手动标定的缩放和平移关系映射到大地图。
        map_position = self._minimap_to_map(marker)

        if self.verbose:
            print(f"pos: {map_position}")
            self._save_debug_image(minimap, candidates, marker)

        return PlayerLocation(
            position=map_position,
            minimap_position=marker,
            confidence=1.0,
            heading_degrees=heading_degrees,
        )

    def _to_reference_marker(self, minimap: np.ndarray, marker: Point) -> Point:
        """把当前小地图玩家点转换到参考小地图坐标。"""
        return marker

    def _crop_minimap(self, frame: np.ndarray) -> np.ndarray:
        """根据配置裁剪小地图区域。"""
        height, width = frame.shape[:2]
        left = clamp(self.config.roi_left, 0, width - 1)
        top = clamp(self.config.roi_top, 0, height - 1)
        right = clamp(left + self.config.roi_width, left + 1, width)
        bottom = clamp(top + self.config.roi_height, top + 1, height)
        return frame[top:bottom, left:right]

    def _detect_active_bounds(self, minimap: np.ndarray) -> tuple[int, int, int, int] | None:
        """自动识别小地图圆形边框，返回真正参与坐标映射的矩形区域。"""
        self.active_circle = None
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # 用霍夫圆检测雷达圆形边框，避免 ROI 框到文字和外部背景导致坐标拉伸。
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(minimap.shape[:2]) // 2,
            param1=80,
            param2=25,
            minRadius=min(minimap.shape[:2]) // 4,
            maxRadius=min(minimap.shape[:2]) // 2,
        )

        if circles is None:
            if self.config.require_minimap_circle and self.last_active_bounds is not None:
                return self.last_active_bounds
            if self.config.require_minimap_circle:
                return None
            return 0, 0, minimap.shape[1], minimap.shape[0]

        circle = np.round(circles[0][0]).astype(int)
        center_x, center_y, radius = int(circle[0]), int(circle[1]), int(circle[2])
        self.active_circle = (center_x, center_y, radius)
        left = clamp(center_x - radius, 0, minimap.shape[1] - 1)
        top = clamp(center_y - radius, 0, minimap.shape[0] - 1)
        right = clamp(center_x + radius, left + 1, minimap.shape[1])
        bottom = clamp(center_y + radius, top + 1, minimap.shape[0])
        return left, top, right - left, bottom - top

    def _find_player_marker(
        self,
        minimap: np.ndarray,
        role: str | None = None,
    ) -> tuple[Point | None, float | None, list[tuple[float, Point, np.ndarray]]]:
        """在小地图中寻找玩家自己的亮色标记。"""
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        # 第一优先级：识别小地图中间的蓝色/青色玩家标记。
        blue_lower = np.array(
            [self.config.blue_hue_min, self.config.blue_saturation_min, self.config.blue_value_min],
            dtype=np.uint8,
        )
        blue_upper = np.array([self.config.blue_hue_max, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # 去掉细碎噪点。
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        candidates = self._collect_candidates(minimap, mask) if role in (None, "auto", "ct") else []

        # 找不到蓝色玩家点时，才退回白色亮点识别。
        if not candidates and role in (None, "auto", "t"):
            yellow_lower = np.array(
                [
                    self.config.yellow_hue_min,
                    self.config.yellow_saturation_min,
                    self.config.yellow_value_min,
                ],
                dtype=np.uint8,
            )
            yellow_upper = np.array([self.config.yellow_hue_max, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            candidates = self._collect_candidates(minimap, mask)

        if not candidates:
            return None, None, candidates

        # 蓝色/青色小圆点就是当前位置；朝向由 main.py 用移动方向计算。
        marker = self._select_candidate(candidates, prefer_center=True)
        return marker, None, candidates

    def _estimate_heading_from_arrow(self, minimap: np.ndarray, marker: Point) -> float | None:
        """识别玩家点附近的黄色箭头，并计算人物朝向。"""
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [
                self.config.arrow_hue_min,
                self.config.arrow_saturation_min,
                self.config.arrow_value_min,
            ],
            dtype=np.uint8,
        )
        upper = np.array([self.config.arrow_hue_max, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 只保留玩家点附近区域，避免 A/B 字母被识别成方向箭头。
        local_mask = np.zeros_like(mask)
        cv2.circle(local_mask, marker, self.config.arrow_search_radius, 255, -1)
        mask = cv2.bitwise_and(mask, local_mask)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 选择面积最大的黄色箭头片段。
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 3:
            return None

        points = contour.reshape(-1, 2)
        center = np.array(marker, dtype=np.float64)
        distances = np.sum((points - center) ** 2, axis=1)
        tip = points[int(np.argmax(distances))]

        minimap_angle = float(np.degrees(np.arctan2(center[1] - tip[1], tip[0] - center[0])) % 360.0)
        return self._minimap_heading_to_map_heading(minimap_angle)

    def _collect_candidates(self, minimap: np.ndarray, mask: np.ndarray) -> list[tuple[float, Point, np.ndarray]]:
        """从颜色 mask 中提取候选点。"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple[float, Point, np.ndarray]] = []
        active_left, active_top, active_width, active_height = self.active_bounds or (
            0,
            0,
            minimap.shape[1],
            minimap.shape[0],
        )
        active_right = active_left + active_width
        active_bottom = active_top + active_height
        margin_x = round(active_width * self.config.edge_margin_ratio)
        margin_y = round(active_height * self.config.edge_margin_ratio)
        circle_center: Point | None = None
        circle_radius = 0
        if self.active_circle is not None:
            center_x, center_y, radius = self.active_circle
            circle_center = (center_x, center_y)
            circle_radius = max(1, radius - max(margin_x, margin_y))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_area or area > self.config.max_area:
                continue

            _rect_x, _rect_y, rect_width, rect_height = cv2.boundingRect(contour)
            if rect_width < self.config.min_marker_width or rect_height < self.config.min_marker_height:
                continue
            if rect_width > self.config.max_marker_width or rect_height > self.config.max_marker_height:
                continue
            aspect_ratio = max(rect_width, rect_height) / max(1, min(rect_width, rect_height))
            if aspect_ratio > self.config.max_marker_aspect_ratio:
                continue
            fill_ratio = area / max(1, rect_width * rect_height)
            if fill_ratio < self.config.min_marker_fill_ratio:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            x = int(moments["m10"] / moments["m00"])
            y = int(moments["m01"] / moments["m00"])

            # 过滤靠边候选，避免识别到小地图边框、外部 HUD 和文字。
            if x < active_left + margin_x or y < active_top + margin_y:
                continue
            if x > active_right - margin_x or y > active_bottom - margin_y:
                continue

            if circle_center is not None and squared_distance((x, y), circle_center) > circle_radius ** 2:
                continue

            candidates.append((area, (x, y), contour))

        return candidates

    def _select_candidate(self, candidates: list[tuple[float, Point, np.ndarray]], prefer_center: bool = False) -> Point:
        """根据上一帧位置选择最稳定的候选点。"""
        if prefer_center:
            active_left, active_top, active_width, active_height = self.active_bounds or (
                0,
                0,
                self.config.roi_width,
                self.config.roi_height,
            )
            center = (active_left + active_width // 2, active_top + active_height // 2)
            candidates.sort(
                key=lambda item: (
                    squared_distance(item[1], center),
                    -item[0],
                )
            )
            self.last_marker = candidates[0][1]
            return self.last_marker

        if self.last_marker is None:
            # 第一帧没有历史位置，选面积最大的亮点。
            candidates.sort(key=lambda item: item[0], reverse=True)
            self.last_marker = candidates[0][1]
            return self.last_marker

        # 有历史位置时，优先选择距离上一帧最近的候选。
        candidates.sort(
            key=lambda item: (
                squared_distance(item[1], self.last_marker),
                -item[0],
            )
        )
        best_marker = candidates[0][1]

        # 如果跳变太大，认为这一帧识别到了别的亮点，沿用上一帧。
        if squared_distance(best_marker, self.last_marker) > self.config.max_jump_distance ** 2:
            return self.last_marker

        self.last_marker = best_marker
        return best_marker

    def _estimate_heading_from_candidate(
        self,
        marker: Point,
        candidates: list[tuple[float, Point, np.ndarray]],
    ) -> float | None:
        """从玩家箭头轮廓估计朝向角度，0=右，90=上。"""
        selected_contour: np.ndarray | None = None
        for _area, point, contour in candidates:
            if point == marker:
                selected_contour = contour
                break

        if selected_contour is None or len(selected_contour) < 3:
            return None

        points = selected_contour.reshape(-1, 2)
        center = np.array(marker, dtype=np.float64)
        distances = np.sum((points - center) ** 2, axis=1)
        tip = points[int(np.argmax(distances))]
        minimap_angle = float(np.degrees(np.arctan2(center[1] - tip[1], tip[0] - center[0])) % 360.0)
        return self._minimap_heading_to_map_heading(minimap_angle)

    def _minimap_heading_to_map_heading(self, minimap_angle: float) -> float:
        """把小地图上的朝向角转换到路径地图坐标系。"""
        # 雷达固定方向后，小地图角度和路径地图角度一致。
        # 不再按标定点额外旋转，避免转向被错误补偿。
        return minimap_angle

    def _minimap_to_map(self, minimap_position: Point) -> Point:
        """把小地图 ROI 内坐标映射到路径地图坐标。"""
        calibrated_position = self._minimap_to_map_by_points(minimap_position)
        if calibrated_position is not None:
            return calibrated_position

        x, y = minimap_position
        active_left, active_top, active_width, active_height = self.active_bounds or (
            0,
            0,
            self.config.roi_width,
            self.config.roi_height,
        )
        x_ratio = (x - active_left) / max(1, active_width - 1)
        y_ratio = (y - active_top) / max(1, active_height - 1)
        x_ratio = max(0.0, min(1.0, x_ratio))
        y_ratio = max(0.0, min(1.0, y_ratio))

        map_x = round(self.config.map_x_min + x_ratio * (self.config.map_x_max - self.config.map_x_min))
        map_y = round(self.config.map_y_min + y_ratio * (self.config.map_y_max - self.config.map_y_min))
        return int(map_x), int(map_y)

    def _minimap_to_map_by_points(self, minimap_position: Point) -> Point | None:
        """优先使用手动标定点进行相似变换坐标映射。"""
        points = self.config.calibration_points or []
        if len(points) < 2:
            return None

        # 雷达固定方向后，只做 x/y 缩放和平移拟合，不再做旋转或仿射变换。
        # 标定点越分散越稳定，建议至少点 3 个：出生点、A 点、B 点附近各一个。
        src = np.array([point["minimap"] for point in points], dtype=np.float64)
        dst = np.array([point["map"] for point in points], dtype=np.float64)
        source_x = np.column_stack([src[:, 0], np.ones(len(src), dtype=np.float64)])
        source_y = np.column_stack([src[:, 1], np.ones(len(src), dtype=np.float64)])
        scale_x, offset_x = np.linalg.lstsq(source_x, dst[:, 0], rcond=None)[0]
        scale_y, offset_y = np.linalg.lstsq(source_y, dst[:, 1], rcond=None)[0]

        map_x = int(round(scale_x * minimap_position[0] + offset_x))
        map_y = int(round(scale_y * minimap_position[1] + offset_y))
        map_x = max(0, min(1000, map_x))
        map_y = max(0, min(1000, map_y))
        return map_x, map_y

    def _minimap_to_map_by_affine(self, minimap_position: Point, points: list[dict[str, Any]]) -> Point | None:
        """三个以上标定点时，使用仿射变换处理不同比例和拉伸。"""
        src = np.array([point["minimap"] for point in points], dtype=np.float32)
        dst = np.array([point["map"] for point in points], dtype=np.float32)
        matrix, _inliers = cv2.estimateAffine2D(src, dst)
        if matrix is None:
            return None

        source_point = np.array([minimap_position[0], minimap_position[1], 1.0], dtype=np.float32)
        mapped = matrix.dot(source_point)
        map_x = int(round(float(mapped[0])))
        map_y = int(round(float(mapped[1])))
        map_x = max(0, min(1000, map_x))
        map_y = max(0, min(1000, map_y))
        return map_x, map_y

    def _save_debug_image(
        self,
        minimap: np.ndarray,
        candidates: list[tuple[float, Point, np.ndarray]],
        marker: Point | None,
    ) -> None:
        """保存小地图识别调试图。"""
        debug = minimap.copy()
        if self.active_bounds is not None:
            left, top, width, height = self.active_bounds
            cv2.rectangle(debug, (left, top), (left + width, top + height), (255, 0, 0), 1)

        # 黄色圈出所有候选点。
        for _area, point, _contour in candidates:
            cv2.circle(debug, point, 4, (0, 255, 255), 1)

        # 红色圈出最终选中的玩家点。
        if marker is not None:
            cv2.circle(debug, marker, 8, (0, 0, 255), 2)

        DEBUG_IMAGE.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG_IMAGE), debug)


def clamp(value: int, min_value: int, max_value: int) -> int:
    """把数值限制到指定范围内。"""
    return max(min_value, min(value, max_value))


def squared_distance(a: Point, b: Point) -> int:
    """计算两个点之间的平方距离。"""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便单张截图测试定位。"""
    parser = argparse.ArgumentParser(description="从小地图识别玩家地图坐标。")
    parser.add_argument("--image", required=True, help="游戏截图路径。")
    parser.add_argument("--roi-left", type=int, default=0)
    parser.add_argument("--roi-top", type=int, default=0)
    parser.add_argument("--roi-width", type=int, default=320)
    parser.add_argument("--roi-height", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    """命令行测试入口。"""
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"找不到图片：{args.image}")

    config = MinimapConfig(
        roi_left=args.roi_left,
        roi_top=args.roi_top,
        roi_width=args.roi_width,
        roi_height=args.roi_height,
    )
    locator = PlayerLocator(config, verbose=True)
    location = locator.locate(image)
    print(location.position if location is not None else "None")


if __name__ == "__main__":
    main()
