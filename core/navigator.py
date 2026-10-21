import argparse
import time
from dataclasses import dataclass
from typing import Callable

try:
    from core.auto_navigation import AutoNavigator, Point, distance
except ModuleNotFoundError:
    from auto_navigation import AutoNavigator, Point, distance


@dataclass
class NavigationState:
    """保存当前导航状态。"""

    role: str
    path_name: str
    current_index: int
    target: Point | None
    finished: bool
    stuck: bool


class RateLimitedLogger:
    """限制控制台日志频率，避免每帧刷屏。"""

    def __init__(self, interval_seconds: float = 1.0, enabled: bool = False) -> None:
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.last_log_time = 0.0

    def log(self, message: str) -> None:
        """每秒最多输出一次状态日志。"""
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_log_time >= self.interval_seconds:
            print(message)
            self.last_log_time = now


class Navigator:
    """从实时当前位置直接用 A* 规划到目标点，并负责 waypoint 跟随。"""

    def __init__(
        self,
        role: str,
        target_name: str = "a_site",
        arrive_distance: float = 8.0,
        deviation_distance: float = 45.0,
        stuck_seconds: float = 2.0,
        stuck_move_distance: float = 3.0,
        simplify_step: int = 12,
        safety_margin_pixels: int = 2,
        preferred_clearance_pixels: float = 10.0,
        clearance_cost_weight: float = 2.0,
        movement_callback: Callable[[Point, Point | None], None] | None = None,
        recovery_callback: Callable[[Point], None] | None = None,
        enable_stuck_recovery: bool = True,
        verbose: bool = False,
    ) -> None:
        if role not in ("ct", "t"):
            raise ValueError("role 必须是 ct 或 t。")

        # 保存导航参数。
        self.role = role
        self.target_name = target_name
        self.arrive_distance = arrive_distance
        self.deviation_distance = deviation_distance
        self.stuck_seconds = stuck_seconds
        self.stuck_move_distance = stuck_move_distance
        self.movement_callback = movement_callback
        self.recovery_callback = recovery_callback
        self.enable_stuck_recovery = enable_stuck_recovery
        self.logger = RateLimitedLogger(enabled=verbose)

        # 复用 AutoNavigator 中的地图读取和 A* 寻路能力。
        self.astar_navigator = AutoNavigator(
            simplify_step=simplify_step,
            safety_margin_pixels=safety_margin_pixels,
            preferred_clearance_pixels=preferred_clearance_pixels,
            clearance_cost_weight=clearance_cost_weight,
        )
        self.target_point = self.astar_navigator.nearest_walkable(
            self.astar_navigator.get_node(target_name)
        )

        # 路线在首次拿到当前位置后实时生成，不再接入预生成 ct/t 路线。
        self.path_name = f"{role}_dynamic_to_{target_name}"
        self.waypoints: list[Point] = []
        self.full_path: list[Point] = []
        self.current_index = 0
        self.route_joined = False
        self.progress_lookahead = 6
        self.pass_distance = 18.0

        # 防卡死状态记录。
        self.last_position: Point | None = None
        self.last_move_time = time.time()
        self.stuck = False

    def update(self, current_position: Point) -> NavigationState:
        """根据实时角色位置更新导航，并返回当前目标点。"""
        if not self.astar_navigator.is_walkable(current_position):
            # 黑区恢复由 main.py 负责；导航器只接受真实白区坐标。
            raise ValueError(f"当前位置不在白色可走区域：{current_position}")

        if distance(current_position, self.target_point) <= self.arrive_distance:
            return self._make_state(target=None, finished=True)

        if not self.route_joined or not self.waypoints:
            self._plan_from_current(current_position, reason="start")

        if self.enable_stuck_recovery:
            self._update_stuck_state(current_position)
        else:
            self.stuck = False

        if self.current_index >= len(self.waypoints):
            return self._make_state(target=None, finished=True)

        self._advance_progress(current_position)
        if self.current_index >= len(self.waypoints):
            return self._make_state(target=None, finished=True)

        target = self.waypoints[self.current_index]
        if not self.astar_navigator.is_walkable(target):
            target = self._repair_path(current_position)

        # 到达当前 waypoint 后切换到下一个 waypoint。
        if distance(current_position, target) <= self.arrive_distance:
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.logger.log("已到达目标")
                return self._make_state(target=None, finished=True)
            target = self.waypoints[self.current_index]
            if not self.astar_navigator.is_walkable(target):
                target = self._repair_path(current_position)

        # 如果偏离当前目标太远，直接从当前位置重新 A* 到最终目标。
        if distance(current_position, target) > self.deviation_distance:
            target = self._repair_path(current_position)

        # 如果停滞超过指定时间，触发恢复操作并重新规划路径。
        if self.enable_stuck_recovery and self.stuck:
            target = self._recover_from_stuck(current_position)

        # movement_callback 由后续键鼠控制模块接入，这里只把目标点交出去。
        original_target = target
        target = self._safe_movement_target(current_position, target)
        next_target = original_target if target != original_target else self._next_waypoint()

        if self.movement_callback is not None and target is not None:
            self.movement_callback(target, next_target)

        self.logger.log(f"目标点：{target}")
        return self._make_state(target=target, finished=False)

    def _plan_from_current(self, current_position: Point, reason: str) -> None:
        """从当前位置直接规划到最终目标点。"""
        if not self.astar_navigator.is_walkable(current_position):
            raise ValueError(f"当前位置不在白色可走区域：{current_position}")

        self.full_path = self.astar_navigator.astar(current_position, self.target_point)
        self.waypoints = self.astar_navigator.simplify_path(self.full_path)
        self.current_index = 0
        self.route_joined = True
        self.logger.log(
            f"A* {reason}: {current_position} -> {self.target_point}, waypoints={len(self.waypoints)}"
        )

    def replace_route(self, waypoints: list[Point], reason: str = "replace") -> None:
        """用外部恢复模块生成的新路线替换当前路线。"""
        if not waypoints:
            self.waypoints = []
            self.full_path = []
            self.current_index = 0
            self.route_joined = False
            return

        self.astar_navigator.validate_path(waypoints, require_line_of_sight=True)
        self.waypoints = waypoints
        self.full_path = waypoints
        self.current_index = 0
        self.route_joined = True
        self.logger.log(f"A* {reason}: waypoints={len(waypoints)}")

    def _safe_movement_target(self, current_position: Point, target: Point) -> Point:
        """如果直线会穿过黑色障碍，就只走 A* 路径上的近距离白色小目标。"""
        if not self.astar_navigator.is_walkable(current_position):
            raise ValueError(f"当前位置不在白色可走区域：{current_position}")

        if not self.astar_navigator.is_walkable(target):
            target = self.astar_navigator.nearest_walkable(target, max_radius=150)

        if self._line_is_walkable(current_position, target):
            return target

        path = self.astar_navigator.astar(current_position, target)
        if len(path) <= 1:
            return current_position

        safe_index = 1
        max_index = min(10, len(path) - 1)
        for index in range(max_index, 0, -1):
            candidate = path[index]
            if self._line_is_walkable(current_position, candidate):
                safe_index = index
                break

        safe_target = path[safe_index]
        self.logger.log(f"safe target: {safe_target}")
        return safe_target

    def _line_is_walkable(self, start: Point, end: Point) -> bool:
        """用 Bresenham 采样检查两点直线是否都在白色可走区域。"""
        return self.astar_navigator.line_is_walkable(start, end)

    def _line_is_safe(self, start: Point, end: Point) -> bool:
        """检查直线可走，并尽量避开黑区边缘。"""
        return self.astar_navigator.line_is_safe(start, end)

    def _next_waypoint(self) -> Point | None:
        """Return the waypoint after the current target so control can prepare corners."""
        next_index = self.current_index + 1
        if 0 <= next_index < len(self.waypoints):
            return self.waypoints[next_index]
        return None

    def recovery_goal(self) -> Point:
        """Pick the route point black-area recovery should move toward."""
        if self.route_joined and self.waypoints:
            index = min(self.current_index + 1, len(self.waypoints) - 1)
            return self.waypoints[index]
        return self.target_point

    def recovery_target_from_black(self, current_position: Point, max_radius: int) -> Point:
        """Choose a white recovery point that continues toward the route."""
        return self.astar_navigator.recovery_target_toward(
            current_position,
            self.recovery_goal(),
            max_radius=max_radius,
        )

    def _advance_progress(self, current_position: Point) -> None:
        """Advance waypoint index monotonically so passed points are not targeted again."""
        while self.current_index < len(self.waypoints):
            target = self.waypoints[self.current_index]
            if distance(current_position, target) <= self.arrive_distance:
                self.current_index += 1
                continue

            if self.current_index > 0 and self._passed_waypoint(
                current_position,
                self.waypoints[self.current_index - 1],
                target,
            ):
                self.logger.log(f"skip passed waypoint: {self.current_index}")
                self.current_index += 1
                continue

            break

        if self.current_index >= len(self.waypoints):
            return

        search_end = min(len(self.waypoints), self.current_index + self.progress_lookahead)
        nearest_index = min(
            range(self.current_index, search_end),
            key=lambda index: distance(current_position, self.waypoints[index]),
        )
        if nearest_index > self.current_index:
            current_target_distance = distance(current_position, self.waypoints[self.current_index])
            nearest_distance = distance(current_position, self.waypoints[nearest_index])
            if nearest_distance + self.pass_distance < current_target_distance:
                self.logger.log(f"advance route index: {self.current_index} -> {nearest_index}")
                self.current_index = nearest_index

    def _passed_waypoint(self, current_position: Point, previous: Point, target: Point) -> bool:
        """Return True once current_position has moved beyond target along previous->target."""
        segment_x = target[0] - previous[0]
        segment_y = target[1] - previous[1]
        length_squared = segment_x * segment_x + segment_y * segment_y
        if length_squared == 0:
            return False

        position_x = current_position[0] - previous[0]
        position_y = current_position[1] - previous[1]
        projection = (position_x * segment_x + position_y * segment_y) / length_squared
        return projection > 1.0 and distance(current_position, target) <= self.deviation_distance

    def _update_stuck_state(self, current_position: Point) -> None:
        """检测角色是否长时间停在同一位置。"""
        now = time.time()
        if self.last_position is None:
            self.last_position = current_position
            self.last_move_time = now
            self.stuck = False
            return

        # 角色移动距离足够大时，刷新最后移动时间。
        if distance(current_position, self.last_position) > self.stuck_move_distance:
            self.last_position = current_position
            self.last_move_time = now
            self.stuck = False
            return

        # 超过 2 秒基本不动，认为可能卡住。
        self.stuck = now - self.last_move_time >= self.stuck_seconds

    def _repair_path(self, current_position: Point) -> Point:
        """角色偏离路径时，重新规划到最终目标。"""
        self._plan_from_current(current_position, reason="repair")
        return self.waypoints[self.current_index]

    def _recover_from_stuck(self, current_position: Point) -> Point:
        """角色卡住时，重新规划到最终目标并触发恢复回调。"""
        self._plan_from_current(current_position, reason="stuck")
        self.stuck = False
        self.last_move_time = time.time()
        self.last_position = current_position

        target = self.waypoints[self.current_index]
        if self.recovery_callback is not None:
            self.recovery_callback(target)

        self.logger.log("已触发防卡死恢复")
        return target

    def _make_state(self, target: Point | None, finished: bool) -> NavigationState:
        """构造统一的导航状态返回值。"""
        return NavigationState(
            role=self.role,
            path_name=self.path_name,
            current_index=self.current_index,
            target=target,
            finished=finished,
            stuck=self.stuck,
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便做简单测试。"""
    parser = argparse.ArgumentParser(description="从当前位置 A* 规划到 A 点。")
    parser.add_argument("--role", choices=["ct", "t"], required=True, help="角色阵营。")
    parser.add_argument("--x", type=int, required=True, help="当前角色 x 坐标。")
    parser.add_argument("--y", type=int, required=True, help="当前角色 y 坐标。")
    return parser.parse_args()


def main() -> None:
    """命令行测试入口，只输出当前应该前往的目标点。"""
    args = parse_args()
    navigator = Navigator(args.role)
    state = navigator.update((args.x, args.y))
    print(state.target if state.target is not None else "done")


if __name__ == "__main__":
    main()
