import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# 计算项目根目录，保证从任意目录运行脚本时路径都正确。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 定义地图、节点、路径输出和调试图输出路径。
MAP_IMAGE = PROJECT_ROOT / "maps" / "mirage_binary.png"
NODES_FILE = PROJECT_ROOT / "maps" / "mirage_nodes.json"
PATHS_DIR = PROJECT_ROOT / "paths"
DEBUG_DIR = PROJECT_ROOT / "runs" / "debug"

# 白色像素表示可走区域，灰度大于等于该阈值视为可走。
WALKABLE_THRESHOLD = 127

# 路线中的标准点名；代码会自动兼容标注文件里的 spwan 和 A_site。
CT_ROUTE = ["ct_spawn", "ct_board", "ct_pavilion", "ct_pipe", "a_site"]
T_ROUTE = [
    "t_spawn",
    "t_wall_A",
    "t_flower",
    "t_door",
    "t_steer",
    "t_box",
    "t_box_A",
    "t_walk",
    "a_site",
]

# 标注时可能写错或大小写不同的点名，在这里统一成标准点名。
NODE_ALIASES = {
    "ct_spwan": "ct_spawn",
    "t_spwan": "t_spawn",
    "A_site": "a_site",
}


Point = tuple[int, int]


@dataclass
class PathPlan:
    """保存一条路径的核心信息。"""

    name: str
    role: str
    full_path: list[Point]
    waypoints: list[Point]


class RateLimitedLogger:
    """限制状态日志频率，避免控制台刷屏。"""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.last_log_time = 0.0

    def log(self, message: str) -> None:
        """每隔指定秒数最多输出一次日志。"""
        now = time.time()
        if now - self.last_log_time >= self.interval_seconds:
            print(message)
            self.last_log_time = now


class AutoNavigator:
    """负责读取地图和节点，并用 A* 计算导航路径。"""

    def __init__(
        self,
        map_path: Path = MAP_IMAGE,
        nodes_path: Path = NODES_FILE,
        simplify_step: int = 25,
        safety_margin_pixels: int = 2,
        preferred_clearance_pixels: float = 10.0,
        clearance_cost_weight: float = 2.0,
    ) -> None:
        self.map_path = map_path
        self.nodes_path = nodes_path
        self.simplify_step = simplify_step
        self.safety_margin_pixels = max(0, safety_margin_pixels)
        self.preferred_clearance_pixels = max(0.0, preferred_clearance_pixels)
        self.clearance_cost_weight = max(0.0, clearance_cost_weight)
        self.min_segment_clearance_pixels = max(
            0.0,
            min(self.preferred_clearance_pixels * 0.45, self.safety_margin_pixels + 2.0),
        )
        self.gray_map = self._load_gray_map()
        self.raw_walkable_map = self.gray_map >= WALKABLE_THRESHOLD
        self.clearance_map = cv2.distanceTransform(
            self.raw_walkable_map.astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        self.walkable_map = self._build_safe_walkable_map(self.raw_walkable_map)
        self.walkable_points = np.column_stack(np.where(self.walkable_map))
        self.nodes = self._load_nodes()

    def _build_safe_walkable_map(self, walkable_map: np.ndarray) -> np.ndarray:
        """Shrink white areas by a small margin so planned paths avoid black edges."""
        if self.safety_margin_pixels <= 0:
            return walkable_map

        kernel_size = self.safety_margin_pixels * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(walkable_map.astype(np.uint8), kernel, iterations=1)
        return eroded.astype(bool)

    def _load_gray_map(self) -> np.ndarray:
        """读取二值地图，并转换为灰度图。"""
        gray_map = cv2.imread(str(self.map_path), cv2.IMREAD_GRAYSCALE)
        if gray_map is None:
            raise FileNotFoundError(f"找不到地图文件：{self.map_path}")
        if gray_map.ndim == 3 and gray_map.shape[2] in (3, 4):
            gray_map = cv2.cvtColor(gray_map, cv2.COLOR_BGR2GRAY)
        elif gray_map.ndim == 3 and gray_map.shape[2] == 1:
            gray_map = gray_map[:, :, 0]
        return gray_map

    def _load_nodes(self) -> dict[str, Point]:
        """读取节点 JSON，并把不同分组中的节点合并为统一字典。"""
        if not self.nodes_path.exists():
            raise FileNotFoundError(f"找不到节点文件：{self.nodes_path}")

        with self.nodes_path.open("r", encoding="utf-8") as file:
            raw_nodes = json.load(file)

        merged_nodes: dict[str, Point] = {}
        for group_nodes in raw_nodes.values():
            for name, coordinate in group_nodes.items():
                standard_name = NODE_ALIASES.get(name, name)
                merged_nodes[standard_name] = (int(coordinate[0]), int(coordinate[1]))

        return merged_nodes

    def get_node(self, name: str) -> Point:
        """按标准点名读取节点坐标。"""
        standard_name = NODE_ALIASES.get(name, name)
        if standard_name not in self.nodes:
            raise KeyError(f"节点不存在：{name}")
        return self.nodes[standard_name]

    def is_walkable(self, point: Point) -> bool:
        """判断像素点是否在可走区域。"""
        x, y = point
        height, width = self.walkable_map.shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return bool(self.walkable_map[y, x])

    def nearest_walkable(self, point: Point, max_radius: int = 40) -> Point:
        """如果节点不在白色区域，就向外搜索最近的可走点。"""
        if self.is_walkable(point):
            return point

        x, y = point
        height, width = self.walkable_map.shape[:2]
        best_point: Point | None = None
        best_distance = float("inf")

        # 逐圈向外找白色像素，找到最近点后立即返回。
        for radius in range(1, max_radius + 1):
            x_min = max(0, x - radius)
            x_max = min(width - 1, x + radius)
            y_min = max(0, y - radius)
            y_max = min(height - 1, y + radius)

            for yy in range(y_min, y_max + 1):
                for xx in range(x_min, x_max + 1):
                    if not self.walkable_map[yy, xx]:
                        continue
                    distance = math.hypot(xx - x, yy - y)
                    if distance < best_distance:
                        best_distance = distance
                        best_point = (xx, yy)

            if best_point is not None:
                return best_point

        return self.nearest_walkable_global(point)

    def nearest_walkable_global(self, point: Point) -> Point:
        """在整张地图中寻找距离当前点最近的白色可走点。"""
        if self.walkable_points.size == 0:
            raise ValueError("地图中没有任何可走区域。")

        x, y = point
        ys = self.walkable_points[:, 0]
        xs = self.walkable_points[:, 1]
        distances = (xs - x) ** 2 + (ys - y) ** 2
        best_index = int(np.argmin(distances))
        return int(xs[best_index]), int(ys[best_index])

    def recovery_target_toward(
        self,
        point: Point,
        goal: Point,
        max_radius: int = 180,
    ) -> Point:
        """Pick a nearby walkable recovery point that also points toward the route goal."""
        if self.is_walkable(point):
            return point

        radius = max(1, int(max_radius))
        x, y = point
        goal = self.nearest_walkable(goal, max_radius=radius)
        height, width = self.walkable_map.shape[:2]
        x_min = max(0, x - radius)
        x_max = min(width - 1, x + radius)
        y_min = max(0, y - radius)
        y_max = min(height - 1, y + radius)
        if x_min > x_max or y_min > y_max:
            return self.nearest_walkable_global(point)

        local_walkable = self.walkable_map[y_min : y_max + 1, x_min : x_max + 1]
        local_ys, local_xs = np.nonzero(local_walkable)
        if local_xs.size == 0:
            return self.nearest_walkable(point, max_radius=radius)

        xs = local_xs + x_min
        ys = local_ys + y_min
        dx = xs - x
        dy = ys - y
        distances_squared = dx * dx + dy * dy
        radius_squared = radius * radius
        inside_radius = distances_squared <= radius_squared
        if not np.any(inside_radius):
            return self.nearest_walkable(point, max_radius=radius)

        xs = xs[inside_radius]
        ys = ys[inside_radius]
        dx = dx[inside_radius]
        dy = dy[inside_radius]
        distances = np.sqrt(distances_squared[inside_radius].astype(np.float64))

        goal_dx = goal[0] - x
        goal_dy = goal[1] - y
        goal_length = math.hypot(goal_dx, goal_dy)
        target_distances = np.hypot(xs - goal[0], ys - goal[1])
        clearances = self.clearance_map[ys, xs].astype(np.float64)

        if goal_length > 0.0:
            candidate_lengths = np.maximum(distances, 1e-6)
            alignment = (dx * goal_dx + dy * goal_dy) / (candidate_lengths * goal_length)
            alignment = np.clip(alignment, -1.0, 1.0)
            alignment_penalty = (1.0 - alignment) * 30.0
        else:
            alignment_penalty = 0.0

        desired_distance = min(float(radius), max(24.0, self.preferred_clearance_pixels * 3.0))
        max_recovery_step = min(float(radius), max(48.0, self.preferred_clearance_pixels * 5.0))
        distance_score = np.abs(distances - desired_distance) * 0.45 + distances * 0.75
        far_penalty = np.maximum(0.0, distances - max_recovery_step) * 1.5
        clearance_deficit = np.maximum(0.0, self.preferred_clearance_pixels - clearances)
        too_close_penalty = np.where(distances < 8.0, 50.0, 0.0)
        scores = (
            target_distances * 0.18
            + distance_score
            + far_penalty
            + alignment_penalty
            + clearance_deficit * 6.0
            + too_close_penalty
            - clearances * 0.4
        )

        best_index = int(np.argmin(scores))
        return int(xs[best_index]), int(ys[best_index])

    def get_current_position(
        self,
        role: str,
        detected_position: Point | None = None,
    ) -> Point:
        """获取角色当前位置；没有截图检测结果时，默认使用对应出生点。"""
        if detected_position is not None:
            return self.nearest_walkable(detected_position)

        spawn_name = "ct_spawn" if role == "ct" else "t_spawn"
        return self.nearest_walkable(self.get_node(spawn_name))

    def find_spawn(self, role: str) -> Point:
        """获取角色阵营对应的起点。"""
        spawn_name = "ct_spawn" if role == "ct" else "t_spawn"
        return self.nearest_walkable(self.get_node(spawn_name))

    def astar(self, start: Point, goal: Point) -> list[Point]:
        """使用 8 邻域 A* 计算从 start 到 goal 的最短路径。"""
        start = self.nearest_walkable(start)
        goal = self.nearest_walkable(goal)

        open_heap: list[tuple[float, int, Point]] = []
        came_from: dict[Point, Point] = {}
        g_score: dict[Point, float] = {start: 0.0}
        closed_set: set[Point] = set()
        push_index = 0

        heapq.heappush(open_heap, (self._heuristic(start, goal), push_index, start))

        while open_heap:
            _f_score, _index, current = heapq.heappop(open_heap)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            for neighbor, move_cost in self._neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + move_cost
                if tentative_g_score >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                push_index += 1
                f_score = tentative_g_score + self._heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, push_index, neighbor))

        raise RuntimeError(f"A* 寻路失败：{start} -> {goal}")

    def _neighbors(self, point: Point) -> list[tuple[Point, float]]:
        """生成 8 邻域可走邻居，斜向移动时避免穿过障碍角。"""
        x, y = point
        neighbors: list[tuple[Point, float]] = []

        for dx, dy in (
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ):
            next_point = (x + dx, y + dy)
            if not self.is_walkable(next_point):
                continue

            # 斜向移动时，两侧正交格子也必须可走，避免路径穿过黑色角落。
            if dx != 0 and dy != 0:
                if not self.is_walkable((x + dx, y)) or not self.is_walkable((x, y + dy)):
                    continue
                neighbors.append((next_point, self._movement_cost(next_point, math.sqrt(2))))
            else:
                neighbors.append((next_point, self._movement_cost(next_point, 1.0)))

        return neighbors

    def _movement_cost(self, point: Point, base_cost: float) -> float:
        """Prefer pixels farther away from black areas without closing narrow routes."""
        if self.preferred_clearance_pixels <= 0.0 or self.clearance_cost_weight <= 0.0:
            return base_cost

        x, y = point
        clearance = float(self.clearance_map[y, x])
        if clearance >= self.preferred_clearance_pixels:
            return base_cost

        penalty_ratio = (self.preferred_clearance_pixels - clearance) / self.preferred_clearance_pixels
        return base_cost * (1.0 + self.clearance_cost_weight * penalty_ratio)

    def _heuristic(self, start: Point, goal: Point) -> float:
        """使用八方向移动常用的 octile 距离作为启发函数。"""
        dx = abs(goal[0] - start[0])
        dy = abs(goal[1] - start[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def _reconstruct_path(self, came_from: dict[Point, Point], current: Point) -> list[Point]:
        """从 A* 的父节点字典回溯完整路径。"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        self.validate_path(path)
        return path

    def simplify_path(self, path: list[Point]) -> list[Point]:
        """Keep sparse waypoints while ensuring each straight segment stays walkable."""
        if not path:
            return []

        step = max(1, self.simplify_step)
        waypoints = [path[0]]
        anchor_index = 0

        while anchor_index < len(path) - 1:
            preferred_index = min(anchor_index + step, len(path) - 1)
            next_index = preferred_index

            while next_index > anchor_index + 1 and not self.line_is_safe(
                path[anchor_index],
                path[next_index],
            ):
                next_index -= 1

            if next_index == anchor_index:
                next_index = anchor_index + 1

            waypoints.append(path[next_index])
            anchor_index = next_index

        self.validate_path(waypoints, require_line_of_sight=True)
        return waypoints

    def line_is_walkable(self, start: Point, end: Point) -> bool:
        """Check every sampled pixel on a straight segment against the safe white map."""
        for point in self._line_points(start, end):
            if not self.is_walkable(point):
                return False
        return True

    def line_is_safe(self, start: Point, end: Point) -> bool:
        """Check line of sight and avoid simplifying through very low-clearance edges."""
        if not self.line_is_walkable(start, end):
            return False
        if self.min_segment_clearance_pixels <= 0.0:
            return True
        return self.line_min_clearance(start, end) >= self.min_segment_clearance_pixels

    def line_min_clearance(self, start: Point, end: Point) -> float:
        """Return the minimum raw walkable clearance along a segment."""
        min_clearance = float("inf")
        for x, y in self._line_points(start, end):
            min_clearance = min(min_clearance, float(self.clearance_map[y, x]))
        return min_clearance

    def _line_points(self, start: Point, end: Point) -> list[Point]:
        """Sample integer pixels on a segment using Bresenham's algorithm."""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx - dy
        points: list[Point] = []

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                return points

            error2 = error * 2
            if error2 > -dy:
                error -= dy
                x0 += step_x
            if error2 < dx:
                error += dx
                y0 += step_y

    def validate_path(self, path: list[Point], require_line_of_sight: bool = False) -> None:
        """Fail fast if a generated path or waypoint segment touches black space."""
        for point in path:
            if not self.is_walkable(point):
                raise RuntimeError(f"路径包含不可走黑区点：{point}")

        if not require_line_of_sight:
            return

        for start, end in zip(path, path[1:]):
            if not self.line_is_walkable(start, end):
                raise RuntimeError(f"路径段穿过黑区：{start} -> {end}")

    def plan_route(self, role: str, target_name: str = "a_site") -> PathPlan:
        """按照预设路线依次连接关键节点，并生成完整路径和简化 waypoint。"""
        route = CT_ROUTE if role == "ct" else T_ROUTE
        route = [target_name if name == "a_site" else name for name in route]

        full_path: list[Point] = []
        for start_name, end_name in zip(route, route[1:]):
            start = self.nearest_walkable(self.get_node(start_name))
            end = self.nearest_walkable(self.get_node(end_name))
            segment_path = self.astar(start, end)

            # 拼接多段路径时跳过重复的段首点。
            if full_path:
                full_path.extend(segment_path[1:])
            else:
                full_path.extend(segment_path)

        plan_name = f"{role}_to_a" if target_name == "a_site" else f"{role}_to_{target_name}"
        return PathPlan(
            name=plan_name,
            role=role,
            full_path=full_path,
            waypoints=self.simplify_path(full_path),
        )

    def plan_to_spawn_if_needed(
        self,
        role: str,
        current_position: Point,
        arrive_distance: float = 8.0,
    ) -> list[Point]:
        """如果当前角色不在出生点附近，就规划一条回到起点的路径。"""
        spawn = self.find_spawn(role)
        if distance(current_position, spawn) <= arrive_distance:
            return []
        return self.astar(current_position, spawn)

    def export_plan(self, plan: PathPlan) -> None:
        """导出路径 JSON 和调试可视化图片。"""
        PATHS_DIR.mkdir(parents=True, exist_ok=True)
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        path_file = PATHS_DIR / f"{plan.name}.json"
        debug_file = DEBUG_DIR / f"{plan.name}_path.png"

        # JSON 只导出简化后的 waypoints，格式符合需求。
        data = {
            "name": plan.name,
            "role": plan.role,
            "waypoints": [[x, y] for x, y in plan.waypoints],
        }
        with path_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")

        # 调试图中蓝线表示完整路径，红点表示简化后的 waypoints。
        debug_image = cv2.cvtColor(self.gray_map, cv2.COLOR_GRAY2BGR)
        for start, end in zip(plan.full_path, plan.full_path[1:]):
            cv2.line(debug_image, start, end, (255, 0, 0), 1)

        for index, point in enumerate(plan.waypoints):
            cv2.circle(debug_image, point, 4, (0, 0, 255), -1)
            if index in (0, len(plan.waypoints) - 1):
                cv2.putText(
                    debug_image,
                    str(index),
                    (point[0] + 6, point[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imwrite(str(debug_file), debug_image)


class PathFollower:
    """根据实时位置跟随 waypoint，并在偏离时重新规划。"""

    def __init__(
        self,
        navigator: AutoNavigator,
        waypoints: list[Point],
        arrive_distance: float = 8.0,
        deviation_distance: float = 35.0,
        movement_callback: Callable[[Point], None] | None = None,
    ) -> None:
        self.navigator = navigator
        self.waypoints = waypoints
        self.arrive_distance = arrive_distance
        self.deviation_distance = deviation_distance
        self.movement_callback = movement_callback
        self.current_index = 0
        self.logger = RateLimitedLogger()

    def update(self, current_position: Point) -> Point | None:
        """根据当前位置返回下一个目标点；到达终点后返回 None。"""
        if self.current_index >= len(self.waypoints):
            return None

        target = self.waypoints[self.current_index]
        if distance(current_position, target) <= self.arrive_distance:
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.logger.log("已到达最终目标")
                return None
            target = self.waypoints[self.current_index]

        # 如果当前位置偏离当前 waypoint 太远，就从当前位置重新规划到当前目标。
        if distance(current_position, target) > self.deviation_distance:
            corrected_path = self.navigator.astar(current_position, target)
            corrected_waypoints = self.navigator.simplify_path(corrected_path)
            self.waypoints = corrected_waypoints + self.waypoints[self.current_index + 1 :]
            self.current_index = 0
            target = self.waypoints[self.current_index]
            self.logger.log("已重新规划路径")

        # movement_callback 由后续键鼠控制模块接入，这里只负责给出目标点。
        if self.movement_callback is not None:
            self.movement_callback(target)

        self.logger.log(f"当前目标：{target}")
        return target


def distance(a: Point, b: Point) -> float:
    """计算两个像素坐标之间的欧氏距离。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成 Mirage 自动导航路径。")
    parser.add_argument(
        "--role",
        choices=["ct", "t", "both"],
        default="both",
        help="生成哪一方路线，默认 both。",
    )
    parser.add_argument(
        "--target",
        default="a_site",
        help="目标点名称，默认 a_site。",
    )
    parser.add_argument(
        "--simplify-step",
        type=int,
        default=25,
        help="每隔多少个路径点保留一个 waypoint，默认 25。",
    )
    parser.add_argument("--current-x", type=int, default=None, help="当前角色 x 坐标。")
    parser.add_argument("--current-y", type=int, default=None, help="当前角色 y 坐标。")
    return parser.parse_args()


def main() -> None:
    """程序入口：生成路径 JSON 和可视化调试图。"""
    args = parse_args()
    navigator = AutoNavigator(simplify_step=args.simplify_step)
    roles = ["ct", "t"] if args.role == "both" else [args.role]

    current_position = None
    if args.current_x is not None and args.current_y is not None:
        current_position = (args.current_x, args.current_y)

    for role in roles:
        # 如果给了当前位置，先规划回出生点的辅助路径。
        if current_position is not None:
            spawn_path = navigator.plan_to_spawn_if_needed(role, current_position)
            if spawn_path:
                spawn_plan = PathPlan(
                    name=f"{role}_to_spawn",
                    role=role,
                    full_path=spawn_path,
                    waypoints=navigator.simplify_path(spawn_path),
                )
                navigator.export_plan(spawn_plan)

        # 从出生点开始生成到目标点的正式路线。
        plan = navigator.plan_route(role, args.target)
        navigator.export_plan(plan)
        print(f"完成：{plan.name}，waypoints={len(plan.waypoints)}")


if __name__ == "__main__":
    main()
