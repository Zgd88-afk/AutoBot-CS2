import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from core.auto_navigation import AutoNavigator, Point, distance
except ModuleNotFoundError:
    from auto_navigation import AutoNavigator, Point, distance


# 计算项目根目录，保证从任意目录运行脚本时路径都正确。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RecoveryResult:
    """保存一次卡死恢复后的结果。"""

    stuck: bool
    target: Point
    path: list[Point]
    waypoints: list[Point]


class RateLimitedLogger:
    """限制控制台日志频率，避免每帧刷屏。"""

    def __init__(self, interval_seconds: float = 1.0, enabled: bool = False) -> None:
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.last_log_time = 0.0

    def log(self, message: str) -> None:
        """每秒最多输出一次日志。"""
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_log_time >= self.interval_seconds:
            print(message)
            self.last_log_time = now


class AntiStuck:
    """检测角色是否卡住，并在卡住后执行恢复和路径重规划。"""

    def __init__(
        self,
        stuck_seconds: float = 2.0,
        min_move_distance: float = 5.0,
        history_seconds: float = 3.0,
        simplify_step: int = 25,
        stop_callback: Callable[[], None] | None = None,
        backward_callback: Callable[[float], None] | None = None,
        turn_callback: Callable[[float], None] | None = None,
        jump_callback: Callable[[], None] | None = None,
        forward_callback: Callable[[], None] | None = None,
        move_to_callback: Callable[[Point], None] | None = None,
        verbose: bool = False,
    ) -> None:
        # 保存卡死判断参数。
        self.stuck_seconds = stuck_seconds
        self.min_move_distance = min_move_distance
        self.history_seconds = history_seconds
        self.position_history: deque[tuple[float, Point]] = deque()

        # 复用 AutoNavigator 中的 A* 路径规划能力。
        self.navigator = AutoNavigator(simplify_step=simplify_step)

        # 保存恢复动作回调，后续可以接入键鼠控制。
        self.stop_callback = stop_callback
        self.backward_callback = backward_callback
        self.turn_callback = turn_callback
        self.jump_callback = jump_callback
        self.forward_callback = forward_callback
        self.move_to_callback = move_to_callback

        # 默认不输出状态；需要调试时传 verbose=True。
        self.logger = RateLimitedLogger(enabled=verbose)

    def update(self, current_position: Point) -> bool:
        """更新当前位置历史，并返回当前是否卡死。"""
        now = time.time()
        current_position = self.navigator.nearest_walkable(current_position)
        self.position_history.append((now, current_position))

        # 删除超出历史窗口的旧位置。
        while self.position_history and now - self.position_history[0][0] > self.history_seconds:
            self.position_history.popleft()

        stuck = self.is_stuck()
        if stuck:
            self.logger.log("检测到卡死")
        return stuck

    def is_stuck(self) -> bool:
        """判断最近 2 秒内移动距离是否小于阈值。"""
        if len(self.position_history) < 2:
            return False

        now = time.time()
        recent_position: Point | None = None
        latest_position = self.position_history[-1][1]

        # 找到至少 stuck_seconds 之前的最近记录。
        for timestamp, position in self.position_history:
            if now - timestamp >= self.stuck_seconds:
                recent_position = position
                break

        if recent_position is None:
            return False

        return distance(recent_position, latest_position) < self.min_move_distance

    def recover_and_replan(self, current_position: Point, target: Point) -> RecoveryResult:
        """执行卡死恢复动作，然后用 A* 重新规划到目标点。"""
        current_position = self.navigator.nearest_walkable(current_position)
        target = self.navigator.nearest_walkable(target)

        recovery_target = self.navigator.recovery_target_toward(
            current_position,
            target,
            max_radius=90,
        )

        # 先执行恢复动作序列。
        self.recover(recovery_target)

        # 恢复后从当前位置重新规划到目标点。
        path = self.navigator.astar(current_position, target)
        waypoints = self.navigator.simplify_path(path)

        # 清空历史，避免恢复后立即再次判定卡死。
        self.reset(current_position)
        self.logger.log("已完成恢复并重规划")

        return RecoveryResult(
            stuck=True,
            target=target,
            path=path,
            waypoints=waypoints,
        )

    def recover(self, recovery_target: Point | None = None) -> None:
        """执行恢复操作：停止、后退、朝目标恢复点重新对准。"""
        # 停止所有移动。
        if self.stop_callback is not None:
            self.stop_callback()

        # 向后移动一小段时间或距离，具体实现由外部控制模块决定。
        if self.backward_callback is not None:
            self.backward_callback(0.35)

        if recovery_target is not None and self.move_to_callback is not None:
            self.move_to_callback(recovery_target)
            return

        # 没有目标回调时才退回随机转向。
        if self.turn_callback is not None:
            self.turn_callback(random.choice([-35.0, -20.0, 20.0, 35.0]))

        # 跳跃，尝试脱离小障碍。
        if self.jump_callback is not None:
            self.jump_callback()

        # 重新启动前进。
        if self.forward_callback is not None:
            self.forward_callback()

    def reset(self, current_position: Point | None = None) -> None:
        """重置位置历史。"""
        self.position_history.clear()
        if current_position is not None:
            self.position_history.append((time.time(), current_position))


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便做简单检测测试。"""
    parser = argparse.ArgumentParser(description="检测并恢复卡死状态。")
    parser.add_argument("--x", type=int, required=True, help="当前角色 x 坐标。")
    parser.add_argument("--y", type=int, required=True, help="当前角色 y 坐标。")
    parser.add_argument("--target-x", type=int, required=True, help="目标 x 坐标。")
    parser.add_argument("--target-y", type=int, required=True, help="目标 y 坐标。")
    return parser.parse_args()


def main() -> None:
    """命令行测试入口。"""
    args = parse_args()
    anti_stuck = AntiStuck()
    result = anti_stuck.recover_and_replan(
        current_position=(args.x, args.y),
        target=(args.target_x, args.target_y),
    )
    print(f"waypoints={len(result.waypoints)}")


if __name__ == "__main__":
    main()
