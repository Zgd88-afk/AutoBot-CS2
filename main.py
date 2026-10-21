import argparse
import ctypes
import json
import time
from pathlib import Path

import cv2
import numpy as np

from core.anti_stuck import AntiStuck
from core.arduino_control import ArduinoController, estimate_current_angle
from core.auto_navigation import AutoNavigator, distance
from core.navigator import Navigator
from core.player_locator import MinimapConfig, PlayerLocator
from core.place_detector import PlaceDetector
from core.role_detector import RoleDetection, RoleDetector


# 项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent
MINIMAP_CONFIG_FILE = PROJECT_ROOT / "configs" / "minimap.json"

# 每隔多少帧做一次识别和导航更新。
DETECT_INTERVAL_FRAMES = 8
VK_ESCAPE = 0x1B


class StatusLogger:
    """Rate-limit repeated runtime status messages."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.last_messages: dict[str, float] = {}

    def log(self, key: str, message: str, force: bool = False) -> None:
        now = time.time()
        last_time = self.last_messages.get(key, 0.0)
        if force or now - last_time >= self.interval_seconds:
            print(message)
            self.last_messages[key] = now


class PositionStabilizer:
    """过滤小地图定位的偶发跳点，等待位置稳定后再导航。"""

    def __init__(self, stable_frames: int = 3, stable_distance: float = 35.0, max_jump: float = 70.0) -> None:
        self.stable_frames = stable_frames
        self.stable_distance = stable_distance
        self.max_jump = max_jump
        self.samples: list[tuple[int, int]] = []
        self.last_position: tuple[int, int] | None = None
        self.previous_position: tuple[int, int] | None = None
        self.last_heading: float | None = None

    def update(self, position: tuple[int, int] | None) -> tuple[int, int] | None:
        """返回稳定后的坐标；不稳定时返回 None。"""
        if position is None:
            return None

        # 已经稳定后，如果突然跳太远，认为是匹配误差，沿用上一帧稳定坐标。
        if self.last_position is not None and distance(position, self.last_position) > self.max_jump:
            return self.last_position

        self.samples.append(position)
        self.samples = self.samples[-self.stable_frames :]

        if len(self.samples) < self.stable_frames:
            return None

        center_x = round(sum(point[0] for point in self.samples) / len(self.samples))
        center_y = round(sum(point[1] for point in self.samples) / len(self.samples))
        center = (center_x, center_y)

        if any(distance(point, center) > self.stable_distance for point in self.samples):
            return None

        if self.last_position != center:
            self.previous_position = self.last_position
        self.last_position = center
        return center

    def update_heading(
        self,
        position: tuple[int, int] | None,
        min_move_distance: float = 8.0,
    ) -> float | None:
        """优先用位置移动方向估计朝向，黄色箭头只做备用。"""
        if position is not None and self.previous_position is not None:
            dx = position[0] - self.previous_position[0]
            dy = position[1] - self.previous_position[1]
            if (dx * dx + dy * dy) ** 0.5 >= min_move_distance:
                heading = estimate_current_angle(self.previous_position, position)
                if heading is not None:
                    self.last_heading = heading
                    return self.last_heading

        return None


class ScreenCapture:
    """截图游戏画面。"""

    def __init__(self) -> None:
        try:
            import mss
        except ImportError as error:
            raise ImportError("Missing dependency: pip install mss") from error

        self.capture = mss.mss()
        self.monitor = self.capture.monitors[1]

    def grab(self) -> np.ndarray:
        """截取一帧，并转成 OpenCV BGR 格式。"""
        frame = np.array(self.capture.grab(self.monitor))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


class MovementCalibrator:
    """Measure initial heading and movement speed with short observable pulses."""

    def __init__(
        self,
        duration_seconds: float,
        min_distance: float,
        near_walkable_tolerance: float = 45.0,
    ) -> None:
        self.duration_seconds = max(0.08, duration_seconds)
        self.min_distance = min_distance
        self.near_walkable_tolerance = near_walkable_tolerance
        self.state = "pending"
        self.start_position: tuple[int, int] | None = None
        self.attempt_commands = ["forward", "right", "left", "backward"]
        self.attempt_index = 0
        self.active_command = "forward"
        self.attempt_started_at = 0.0
        self.last_pulse_time = 0.0
        self.moving_seconds = 0.0
        self.pulse_seconds = min(0.12, max(0.06, self.duration_seconds / 6.0))
        self.pulse_interval_seconds = 0.08
        self.settle_seconds = 0.25
        self.wait_timeout_seconds = max(1.0, self.duration_seconds + 0.6)

    def update(
        self,
        position: tuple[int, int],
        controller: "GameController",
        walkable_checker: AutoNavigator,
        status: StatusLogger,
    ) -> bool:
        if self.state == "done" or controller.arduino is None:
            self.state = "done"
            return True

        if self.state == "pending":
            if not walkable_checker.is_walkable(position):
                snapped = walkable_checker.nearest_walkable(position)
                snap_distance = distance(position, snapped)
                if snap_distance > self.near_walkable_tolerance:
                    controller.stop_all()
                    status.log(
                        "calibration_black",
                        f"wait: calibration start is black area {position}; movement stopped",
                    )
                    return False

                status.log(
                    "calibration_near_black",
                    f"calibration: treating near-wall map position as usable {position} -> {snapped}, dist={snap_distance:.1f}",
                )

            self.start_position = position
            self.active_command = self.attempt_commands[self.attempt_index]
            self.attempt_started_at = time.time()
            self.last_pulse_time = 0.0
            self.moving_seconds = 0.0
            self.state = "measuring"
            status.log(
                "calibration_start",
                f"calibration: {self.active_command} pulses up to {self.duration_seconds:.2f}s from {position}",
                force=True,
            )
            return False

        if self.start_position is None:
            self.state = "pending"
            return False

        moved_distance = distance(self.start_position, position)
        if moved_distance < self.min_distance:
            now = time.time()
            if (
                self.moving_seconds < self.duration_seconds
                and now - self.last_pulse_time >= self.pulse_interval_seconds
            ):
                self._send_probe_pulse(controller)
                return False

            controller.stop_all()
            exhausted = self.moving_seconds >= self.duration_seconds
            settled = now - self.last_pulse_time >= self.settle_seconds
            timed_out = now - self.attempt_started_at >= self.wait_timeout_seconds
            if (exhausted and settled) or timed_out:
                self._retry_next_command(status, moved_distance)
                return False

            status.log(
                "calibration_wait",
                f"wait: calibration {self.active_command} movement not visible yet start={self.start_position}, current={position}, dist={moved_distance:.1f}",
            )
            return False

        movement_angle = estimate_current_angle(self.start_position, position, min_move_distance=self.min_distance)
        heading = self._movement_angle_to_heading(movement_angle)
        speed = moved_distance / max(self.moving_seconds, self.pulse_seconds)
        controller.arduino.movement_speed = speed
        if heading is not None:
            controller.arduino.update_heading(heading)

        heading_text = "none" if heading is None else f"{heading:.1f}"
        move_text = "none" if movement_angle is None else f"{movement_angle:.1f}"
        status.log(
            "calibration_done",
            f"calibration done: command={self.active_command}, start={self.start_position}, end={position}, movement={move_text}, heading={heading_text}, speed={speed:.1f}px/s",
            force=True,
        )
        self.state = "done"
        return True

    def _send_probe_pulse(self, controller: "GameController") -> None:
        if controller.arduino is None:
            return

        remaining_seconds = max(0.0, self.duration_seconds - self.moving_seconds)
        pulse_seconds = min(self.pulse_seconds, remaining_seconds)
        if pulse_seconds <= 0.0:
            return

        controller.arduino.movement_pulse(self.active_command, pulse_seconds)
        self.moving_seconds += pulse_seconds
        self.last_pulse_time = time.time()

    def _retry_next_command(self, status: StatusLogger, moved_distance: float) -> None:
        self.attempt_index = (self.attempt_index + 1) % len(self.attempt_commands)
        self.state = "pending"
        self.start_position = None
        status.log(
            "calibration_retry",
            f"calibration retry: {self.active_command} moved only {moved_distance:.1f}px",
            force=True,
        )

    def reset_attempt(self) -> None:
        """Discard the current probe when recovery movement would pollute calibration."""
        self.state = "pending"
        self.start_position = None
        self.last_pulse_time = 0.0
        self.moving_seconds = 0.0

    def _movement_angle_to_heading(self, movement_angle: float | None) -> float | None:
        if movement_angle is None:
            return None

        if self.active_command == "forward":
            return movement_angle
        if self.active_command == "backward":
            return (movement_angle + 180.0) % 360.0
        if self.active_command == "right":
            return (movement_angle + 90.0) % 360.0
        if self.active_command == "left":
            return (movement_angle - 90.0) % 360.0
        return movement_angle


class RoleLock:
    """Lock auto-detected role only after the same result appears repeatedly."""

    def __init__(self, required_frames: int = 2) -> None:
        self.required_frames = max(1, required_frames)
        self.candidate_role: str | None = None
        self.candidate_source: str | None = None
        self.count = 0

    def update(self, role: str | None, source: str | None) -> str | None:
        if role is None:
            return None

        if role == self.candidate_role and source == self.candidate_source:
            self.count += 1
        else:
            self.candidate_role = role
            self.candidate_source = source
            self.count = 1

        if self.count >= self.required_frames:
            return role
        return None


class CombatController:
    """Global combat state: stop navigation and shoot only when enemy fire lane is safe."""

    def __init__(
        self,
        aim_gain: float,
        fire_cooldown: float,
        fire_deadzone: float,
        friendly_padding: float,
    ) -> None:
        self.aim_gain = aim_gain
        self.fire_cooldown = fire_cooldown
        self.fire_deadzone = fire_deadzone
        self.friendly_padding = friendly_padding
        self.last_fire_time = 0.0

    def update(
        self,
        frame: np.ndarray,
        role: str,
        controller: "GameController",
        detector: RoleDetector,
        status: StatusLogger,
    ) -> bool:
        detections = detector.detect_all(frame)
        if not detections:
            print("[战斗] 当前画面没有检测到任何角色")
            return False

        enemy_role = "t" if role == "ct" else "ct"
        enemies = [detection for detection in detections if detection.role == enemy_role]
        friends = [detection for detection in detections if detection.role == role]

        print(
            f"[战斗] 检测到角色数量: total={len(detections)}, "
            f"enemies={len(enemies)}, friends={len(friends)}, 当前阵营={role}"
        )

        if not enemies:
            print("[战斗] 没有发现敌人")
            return False

        target = self._select_enemy(enemies, frame.shape)

        print(
            f"[战斗] 发现敌人: role={target.role}, "
            f"class={target.class_name}, "
            f"conf={target.confidence:.2f}, "
            f"aim={target.aim_position}"
        )

        blocked_friend = self._blocking_friend(target, friends, frame.shape)
        controller.stop_all()

        if blocked_friend is not None:
            print(
                f"[战斗] 暂停开火：队友挡在线上，"
                f"class={blocked_friend.class_name}, "
                f"conf={blocked_friend.confidence:.2f}"
            )
            status.log(
                "combat_blocked",
                f"hold fire: teammate in lane class={blocked_friend.class_name} conf={blocked_friend.confidence:.2f}",
            )
            return True

        delta_x, delta_y = self._aim_delta(target, frame.shape)

        print(
            f"[战斗] 正在瞄准敌人: "
            f"dx={delta_x:.1f}, dy={delta_y:.1f}, "
            f"deadzone={self.fire_deadzone}"
        )

        controller.aim_at(target, frame.shape, self.aim_gain)

        if abs(delta_x) > self.fire_deadzone or abs(delta_y) > self.fire_deadzone:
            print("[战斗] 已发现敌人，但准星还没对准，暂时不开火")
            status.log(
                "combat_aim",
                f"aim enemy={target.role} class={target.class_name} dx={delta_x:.0f} dy={delta_y:.0f}",
            )
            return True

        now = time.time()
        if now - self.last_fire_time >= self.fire_cooldown:
            print("[战斗] 准星已对准，正在射击")
            controller.fire()
            self.last_fire_time = now
            status.log(
                "combat_fire",
                f"fire enemy={target.role} class={target.class_name} conf={target.confidence:.2f} pos={target.aim_position}",
                force=True,
            )
        else:
            print("[战斗] 准星已对准，但还在开火冷却中")

        return True

    def _select_enemy(self, enemies: list[RoleDetection], frame_shape: tuple[int, ...]) -> RoleDetection:
        center_x = frame_shape[1] / 2
        center_y = frame_shape[0] / 2

        def score(detection: RoleDetection) -> tuple[int, float, float]:
            aim_x, aim_y = detection.aim_position
            distance_to_crosshair = ((aim_x - center_x) ** 2 + (aim_y - center_y) ** 2) ** 0.5
            is_head = 1 if detection.class_name.endswith("_head") else 0
            return is_head, -distance_to_crosshair, detection.confidence

        return max(enemies, key=score)

    def _aim_delta(self, detection: RoleDetection, frame_shape: tuple[int, ...]) -> tuple[float, float]:
        frame_height, frame_width = frame_shape[:2]
        aim_x, aim_y = detection.aim_position
        return aim_x - frame_width / 2, aim_y - frame_height / 2

    def _blocking_friend(
        self,
        target: RoleDetection,
        friends: list[RoleDetection],
        frame_shape: tuple[int, ...],
    ) -> RoleDetection | None:
        frame_height, frame_width = frame_shape[:2]
        start = (frame_width / 2, frame_height / 2)
        end = target.aim_position
        target_box = self._expand_box(target.box, self.friendly_padding)
        for friend in friends:
            friend_box = self._expand_box(friend.box, self.friendly_padding)
            if self._boxes_overlap(friend_box, target_box):
                return friend
            if self._segment_intersects_box(start, end, friend_box):
                return friend
        return None

    def _expand_box(
        self,
        box: tuple[int, int, int, int],
        padding: float,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        return x1 - padding, y1 - padding, x2 + padding, y2 + padding

    def _boxes_overlap(
        self,
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> bool:
        return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]

    def _segment_intersects_box(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        box: tuple[float, float, float, float],
    ) -> bool:
        x1, y1, x2, y2 = box
        if self._point_in_box(start, box) or self._point_in_box(end, box):
            return True

        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        edges = zip(corners, corners[1:] + corners[:1])
        return any(self._segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in edges)

    def _point_in_box(self, point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
        return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

    def _segments_intersect(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        c: tuple[float, float],
        d: tuple[float, float],
    ) -> bool:
        def ccw(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


class SiteGuard:
    """Guard A site with short left/right patrol pulses and enemy detection."""

    def __init__(
        self,
        target_point: tuple[int, int],
        arrive_distance: float,
        patrol_seconds: float,
        patrol_interval: float,
        scan_turn_degrees: float,
        fire_cooldown: float,
        aim_gain: float,
    ) -> None:
        self.target_point = target_point
        self.arrive_distance = arrive_distance
        self.patrol_seconds = patrol_seconds
        self.patrol_interval = patrol_interval
        self.scan_turn_degrees = scan_turn_degrees
        self.fire_cooldown = fire_cooldown
        self.aim_gain = aim_gain
        self.active = False
        self.next_direction = "left"
        self.last_patrol_time = 0.0
        self.last_fire_time = 0.0
        self.last_scan_time = 0.0

    def should_activate(self, position: tuple[int, int] | None) -> bool:
        if position is None:
            return False
        return distance(position, self.target_point) <= self.arrive_distance

    def update(
        self,
        frame: np.ndarray,
        role: str,
        position: tuple[int, int],
        controller: "GameController",
        detector: RoleDetector,
        status: StatusLogger,
    ) -> None:
        if not self.active:
            self.active = True
            controller.stop_all()
            status.log("site_guard", f"A site guard active at {position}", force=True)

        enemy = detector.detect_enemy(frame, role)
        if enemy is not None:
            now = time.time()
            controller.aim_at(enemy, frame.shape, self.aim_gain)
            if now - self.last_fire_time >= self.fire_cooldown:
                controller.fire()
                self.last_fire_time = now
                status.log(
                    "site_guard_fire",
                    f"fire enemy={enemy.role} class={enemy.class_name} conf={enemy.confidence:.2f} pos={enemy.aim_position}",
                    force=True,
                )
            return

        now = time.time()
        if now - self.last_patrol_time < self.patrol_interval:
            return

        if controller.arduino is None:
            controller._log(f"guard patrol: {self.next_direction}")
        else:
            controller.arduino.movement_pulse(self.next_direction, self.patrol_seconds)
            if self.scan_turn_degrees > 0.0:
                scan_delta = self.scan_turn_degrees if self.next_direction == "right" else -self.scan_turn_degrees
                controller.arduino.turn_degrees(scan_delta)

        self.next_direction = "right" if self.next_direction == "left" else "left"
        self.last_patrol_time = now


class GameController:
    """统一封装 dry-run 和 Arduino 控制。"""

    def __init__(self, arduino: ArduinoController | None = None) -> None:
        self.arduino = arduino
        self.current_position: tuple[int, int] | None = None
        self.last_log_time = 0.0

    def update_position(self, position: tuple[int, int]) -> None:
        """更新当前地图坐标。"""
        self.current_position = position

    def update_heading(self, heading_degrees: float | None) -> None:
        """更新当前朝向角度。"""
        if heading_degrees is None or self.arduino is None:
            return
        self.arduino.update_heading(heading_degrees)

    def has_heading(self) -> bool:
        """判断控制器是否已有朝向估计。"""
        return self.arduino is not None and self.arduino.has_heading()

    def move_to(self, target: tuple[int, int], next_target: tuple[int, int] | None = None) -> None:
        """朝目标点移动。"""
        if self.arduino is None:
            self._log(f"target: {target}")
            return

        if self.current_position is None:
            self.arduino.stop()
            return

        self.arduino.move_to_waypoint(self.current_position, target, next_target=next_target)

    def stop_all(self) -> None:
        """停止移动。"""
        if self.arduino is None:
            self._log("stop")
            return
        self.arduino.stop()

    def move_backward(self, duration_seconds: float) -> None:
        """后退一小段时间。"""
        if self.arduino is None:
            self._log(f"backward: {duration_seconds:.2f}s")
            return
        self.arduino.backward(duration_seconds)

    def random_turn(self, angle: float) -> None:
        """恢复时转向。"""
        if self.arduino is None:
            self._log(f"turn: {angle:.1f}")
            return

        self.arduino.turn_degrees(angle)

    def jump(self) -> None:
        """跳跃。"""
        if self.arduino is None:
            self._log("jump")
            return
        self.arduino.jump()

    def aim_at(self, detection: RoleDetection, frame_shape: tuple[int, ...], aim_gain: float) -> None:
        """Move crosshair a little toward a detected enemy."""
        if self.arduino is None:
            self._log(f"aim: {detection.aim_position}")
            return

        frame_height, frame_width = frame_shape[:2]
        aim_x, aim_y = detection.aim_position
        delta_x = round((aim_x - frame_width / 2) * aim_gain)
        delta_y = round((aim_y - frame_height / 2) * aim_gain)
        max_step = 90
        delta_x = max(-max_step, min(max_step, delta_x))
        delta_y = max(-max_step, min(max_step, delta_y))
        self.arduino.aim_pixels(delta_x, delta_y)

    def fire(self) -> None:
        """Fire once."""
        if self.arduino is None:
            self._log("fire")
            return
        self.arduino.fire()

    def move_forward(self) -> None:
        """前进。"""
        if self.arduino is None:
            self._log("forward")
            return
        self.arduino.forward()

    def move_to_walkable(self, target: tuple[int, int]) -> None:
        """Move toward a nearby walkable point, using feedback if heading is unknown."""
        if self.arduino is None:
            self._log(f"walkable target: {target}")
            return

        if self.current_position is None:
            self.arduino.stop()
            return

        if self.arduino.has_heading():
            self.arduino.move_to_waypoint(self.current_position, target)
        else:
            self.arduino.probe_heading_by_mouse_scan()

    def close(self) -> None:
        """释放控制资源。"""
        if self.arduino is not None:
            self.arduino.close()

    def _log(self, message: str) -> None:
        """限制 dry-run 输出频率。"""
        now = time.time()
        if now - self.last_log_time >= 1.0:
            print(message)
            self.last_log_time = now


def create_navigator(
    role: str,
    controller: GameController,
    verbose: bool,
    simplify_step: int,
    preferred_clearance_pixels: float,
    clearance_cost_weight: float,
) -> Navigator:
    """创建从实时当前位置直接 A* 到 A 点的导航器。"""
    return Navigator(
        role=role,
        simplify_step=simplify_step,
        preferred_clearance_pixels=preferred_clearance_pixels,
        clearance_cost_weight=clearance_cost_weight,
        movement_callback=controller.move_to,
        recovery_callback=controller.move_to,
        # 卡死恢复动作由 AntiStuck 统一处理，Navigator 只负责规划和跟随。
        enable_stuck_recovery=False,
        verbose=verbose,
    )


def create_anti_stuck(controller: GameController, verbose: bool) -> AntiStuck:
    """创建防卡死模块。"""
    return AntiStuck(
        stop_callback=controller.stop_all,
        backward_callback=controller.move_backward,
        turn_callback=controller.random_turn,
        jump_callback=controller.jump,
        forward_callback=controller.move_forward,
        move_to_callback=controller.move_to,
        verbose=verbose,
    )


def update_navigation(
    role: str | None,
    position: tuple[int, int] | None,
    controller: GameController,
    navigator: Navigator | None,
    anti_stuck: AntiStuck | None,
    verbose: bool,
    enable_recovery: bool,
    max_snap_distance: float,
    black_snap_tolerance: float,
    start_max_snap_distance: float,
    simplify_step: int,
    preferred_clearance_pixels: float,
    clearance_cost_weight: float,
) -> tuple[Navigator | None, AntiStuck | None]:
    """根据阵营和当前位置更新导航。"""
    # 保留旧命令行参数兼容；当前策略是黑区必须先真实恢复到白区，不再吸附坐标。
    _ = black_snap_tolerance

    if role is None:
        if verbose:
            print("wait: role")
        return navigator, anti_stuck

    if position is None:
        if verbose:
            print("wait: position")
        return navigator, anti_stuck

    controller.update_position(position)

    if navigator is None or navigator.role != role:
        navigator = create_navigator(
            role,
            controller,
            verbose,
            simplify_step,
            preferred_clearance_pixels,
            clearance_cost_weight,
        )
        anti_stuck = create_anti_stuck(controller, verbose)
        print(f"role: {role}, route: dynamic A* to A")

    before_first_plan = not navigator.route_joined

    if not navigator.astar_navigator.is_walkable(position):
        recovery_limit = start_max_snap_distance if before_first_plan else max_snap_distance
        recovery_radius = 220 if recovery_limit <= 0 else max(1, int(recovery_limit))
        snapped = navigator.recovery_target_from_black(position, max_radius=recovery_radius)
        snap_distance = distance(position, snapped)
        if recovery_limit <= 0 or snap_distance <= recovery_limit:
            controller.update_position(position)
            controller.move_to_walkable(snapped)
            if verbose:
                phase = "start" if before_first_plan else "runtime"
                goal = navigator.recovery_goal()
                print(f"{phase} black recover: {position} -> {snapped}, goal={goal}, dist={snap_distance:.1f}")
            return navigator, anti_stuck

        controller.stop_all()
        if verbose:
            print(
                f"stop: black area too far from white pos={position} -> {snapped}, "
                f"dist={snap_distance:.1f}, limit={recovery_limit:.1f}"
            )
        return navigator, anti_stuck

    try:
        state = navigator.update(position)
    except ValueError:
        controller.stop_all()
        if verbose:
            print(f"wait: walkable position {position}")
        return navigator, anti_stuck

    if state.finished or state.target is None:
        controller.stop_all()
        print("navigation: done")
        return navigator, anti_stuck

    if enable_recovery and anti_stuck is not None and anti_stuck.update(position):
        result = anti_stuck.recover_and_replan(position, navigator.target_point)
        navigator.replace_route(result.waypoints, reason="stuck recovery")

    return navigator, anti_stuck


def load_minimap_config(args: argparse.Namespace) -> MinimapConfig:
    """读取小地图配置。"""
    if MINIMAP_CONFIG_FILE.exists():
        with MINIMAP_CONFIG_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return MinimapConfig(
            roi_left=int(data.get("roi_left", args.minimap_left)),
            roi_top=int(data.get("roi_top", args.minimap_top)),
            roi_width=int(data.get("roi_width", args.minimap_width)),
            roi_height=int(data.get("roi_height", args.minimap_height)),
            map_x_min=int(data.get("map_x_min", 0)),
            map_y_min=int(data.get("map_y_min", 0)),
            map_x_max=int(data.get("map_x_max", 1000)),
            map_y_max=int(data.get("map_y_max", 1000)),
            require_minimap_circle=bool(data.get("require_minimap_circle", not args.allow_no_minimap_circle)),
            calibration_points=data.get("calibration_points"),
        )

    return MinimapConfig(
        roi_left=args.minimap_left,
        roi_top=args.minimap_top,
        roi_width=args.minimap_width,
        roi_height=args.minimap_height,
        require_minimap_circle=not args.allow_no_minimap_circle,
    )


def infer_role_from_position(
    position: tuple[int, int] | None,
    navigator: AutoNavigator,
    max_spawn_distance: float = 180.0,
) -> str | None:
    """YOLO 失败时，用离出生区域参考点的距离粗略推断阵营。"""
    if position is None:
        return None

    ct_spawn = navigator.find_spawn("ct")
    t_spawn = navigator.find_spawn("t")
    ct_distance = distance(position, ct_spawn)
    t_distance = distance(position, t_spawn)

    if min(ct_distance, t_distance) > max_spawn_distance:
        return None

    return "ct" if ct_distance <= t_distance else "t"


def is_escape_pressed() -> bool:
    """检测 ESC，游戏窗口有焦点时也生效。"""
    return bool(ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="自动识别阵营并沿路线导航。")
    parser.add_argument("--arduino", action="store_true", help="启用 Arduino 串口控制。")
    parser.add_argument("--arduino-port", default="COM6", help="Arduino 串口号。")
    parser.add_argument("--baudrate", type=int, default=9600, help="Arduino 波特率。")
    parser.add_argument("--heading-offset", type=float, default=0.0, help="朝向角度偏移，用于校准转向。")
    parser.add_argument("--invert-turn", action="store_true", help="左右转向反了时启用。")
    parser.add_argument("--enable-recovery", action="store_true", default=True, help="启用卡死恢复动作。")
    parser.add_argument("--disable-recovery", dest="enable_recovery", action="store_false", help="关闭卡死恢复动作。")
    parser.add_argument("--angle-tolerance", type=float, default=4.0, help="Maximum heading error in degrees before moving forward.")
    parser.add_argument("--mouse-pixels-per-degree", type=float, default=2.25, help="Mouse pixels per view degree; calibrate this for your in-game sensitivity.")
    parser.add_argument("--turn-gain", type=float, default=1.25, help="Extra mouse turn multiplier when the view tends to under-rotate.")
    parser.add_argument("--heading-turn-trust", type=float, default=0.6, help="How much to trust a sent turn command when updating estimated heading.")
    parser.add_argument("--max-turn-degrees", type=float, default=180.0, help="Maximum one-shot turn angle.")
    parser.add_argument("--slow-test", action="store_true", help="Move in short pulses and pause after turn/move for debugging.")
    parser.add_argument("--forward-pulse-seconds", type=float, default=0.08, help="Forward key hold time in slow-test mode.")
    parser.add_argument("--action-pause-seconds", type=float, default=0.03, help="Pause after each movement pulse.")
    parser.add_argument("--turn-pause-seconds", type=float, default=0.08, help="Pause after each turn so the next frame can observe the new view.")
    parser.add_argument(
        "--continuous-heading-update",
        action="store_true",
        help="持续用小地图位移修正朝向；默认只在未初始化朝向时使用，避免撞墙滑动污染视角。",
    )
    parser.add_argument("--movement-speed", type=float, default=120.0, help="Estimated map pixels per second for timed forward movement.")
    parser.add_argument("--forward-distance-factor", type=float, default=0.65, help="Move this fraction of the remaining waypoint distance per pulse.")
    parser.add_argument("--min-forward-seconds", type=float, default=0.07, help="Minimum forward key hold time outside slow-test mode.")
    parser.add_argument("--max-forward-seconds", type=float, default=0.18, help="Maximum forward key hold time outside slow-test mode.")
    parser.add_argument("--skip-movement-calibration", action="store_true", help="Use --movement-speed directly instead of measuring speed at startup.")
    parser.add_argument("--calibration-forward-seconds", type=float, default=1.0, help="Maximum startup movement budget for heading/speed calibration.")
    parser.add_argument("--calibration-min-distance", type=float, default=12.0, help="Minimum map movement required for startup speed calibration.")
    parser.add_argument("--calibration-black-tolerance", type=float, default=45.0, help="Calibration can continue when map position is this close to white area.")
    parser.add_argument("--detect-interval-frames", type=int, default=2, help="Run minimap/navigation update every N frames.")
    parser.add_argument("--stable-frames", type=int, default=2, help="Number of stable minimap samples required before movement.")
    parser.add_argument("--max-position-jump", type=float, default=70.0, help="Ignore single-frame minimap jumps larger than this many map pixels.")
    parser.add_argument("--role", choices=["auto", "ct", "t"], default="auto", help="阵营。")
    parser.add_argument("--role-lock-frames", type=int, default=2, help="自动阵营连续确认多少帧后锁定。")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度。")
    parser.add_argument("--place-conf", type=float, default=0.25, help="环境识别模型置信度。")
    parser.add_argument(
        "--person-role-fallback",
        action="store_true",
        help="环境和出生点都无法判断时，才允许用人物模型兜底判断阵营。",
    )
    parser.add_argument("--combat", action="store_true", default=True, help="任何时候识别到敌方就暂停导航并进入射击状态。")
    parser.add_argument("--disable-combat", dest="combat", action="store_false", help="关闭全局敌方识别射击状态。")
    parser.add_argument("--combat-aim-gain", type=float, default=0.18, help="敌方检测框到鼠标微调的比例。")
    parser.add_argument("--combat-fire-cooldown", type=float, default=0.28, help="全局战斗开火冷却。")
    parser.add_argument("--combat-fire-deadzone", type=float, default=55.0, help="准星离敌方瞄准点小于该像素才开火。")
    parser.add_argument("--combat-friendly-padding", type=float, default=18.0, help="队友防误伤检测框扩张像素。")
    parser.add_argument("--site-guard", action="store_true", default=True, help="到 A site 后停止导航并守点。")
    parser.add_argument("--disable-site-guard", dest="site_guard", action="store_false", help="关闭 A site 守点模式。")
    parser.add_argument("--site-guard-distance", type=float, default=35.0, help="离 A site 多近时进入守点模式。")
    parser.add_argument("--site-patrol-seconds", type=float, default=0.12, help="守点时每次左右小范围移动时长。")
    parser.add_argument("--site-patrol-interval", type=float, default=0.75, help="守点左右巡逻间隔。")
    parser.add_argument("--site-scan-turn-degrees", type=float, default=6.0, help="守点巡逻时顺带左右扫视的小角度。")
    parser.add_argument("--site-fire-cooldown", type=float, default=0.28, help="识别到敌人后的开火冷却。")
    parser.add_argument("--site-aim-gain", type=float, default=0.18, help="person.pt 检测框到鼠标微调的比例。")
    parser.add_argument("--max-frames", type=int, default=0, help="最多运行帧数，0 表示一直运行。")
    parser.add_argument("--verbose", action="store_true", help="输出调试信息。")
    parser.add_argument("--max-snap-distance", type=float, default=180.0, help="运行中黑区位置离最近白区超过该距离时停止恢复；<=0 表示不限制。")
    parser.add_argument("--start-max-snap-distance", type=float, default=180.0, help="开局黑区位置离最近白区超过该距离时停止恢复。")
    parser.add_argument("--navigator-simplify-step", type=int, default=12, help="A* waypoint 间隔，越小转向越频繁。")
    parser.add_argument("--preferred-clearance-pixels", type=float, default=10.0, help="A* 优先远离黑区边缘的距离。")
    parser.add_argument("--clearance-cost-weight", type=float, default=2.0, help="A* 靠近黑区边缘的额外代价。")
    parser.add_argument("--minimap-left", type=int, default=0, help="小地图左上角 x。")
    parser.add_argument("--minimap-top", type=int, default=0, help="小地图左上角 y。")
    parser.add_argument("--minimap-width", type=int, default=320, help="小地图宽度。")
    parser.add_argument("--minimap-height", type=int, default=320, help="小地图高度。")
    parser.add_argument(
        "--allow-no-minimap-circle",
        action="store_true",
        help="调试用：找不到圆形雷达时仍尝试定位。正常运行不要打开。",
    )
    parser.add_argument("--black-snap-tolerance", type=float, default=3.0, help="保留兼容旧参数；当前黑区不会再被吸附成白区。")
    return parser.parse_args()


def main() -> None:
    """主循环。"""
    args = parse_args()

    arduino = None
    if args.arduino:
        arduino = ArduinoController(
            port=args.arduino_port,
            baudrate=args.baudrate,
            angle_tolerance=args.angle_tolerance,
            heading_offset=args.heading_offset,
            invert_turn=args.invert_turn,
            mouse_pixels_per_degree=args.mouse_pixels_per_degree,
            turn_gain=args.turn_gain,
            heading_turn_trust=args.heading_turn_trust,
            max_turn_degrees=args.max_turn_degrees,
            slow_mode=args.slow_test,
            forward_pulse_seconds=args.forward_pulse_seconds,
            action_pause_seconds=args.action_pause_seconds,
            turn_pause_seconds=args.turn_pause_seconds,
            movement_speed=args.movement_speed,
            forward_distance_factor=args.forward_distance_factor,
            min_forward_seconds=args.min_forward_seconds,
            max_forward_seconds=args.max_forward_seconds,
            dry_run=False,
            verbose=args.verbose,
        )

    minimap_config = load_minimap_config(args)
    status = StatusLogger(interval_seconds=1.0)
    calibration_count = len(minimap_config.calibration_points or [])

    controller = GameController(arduino=arduino)
    controller.stop_all()
    capture = ScreenCapture()
    place_detector = PlaceDetector(confidence=args.place_conf, verbose=args.verbose) if args.role == "auto" else None
    person_detector = (
        RoleDetector(confidence=args.conf, verbose=args.verbose)
        if (args.role == "auto" and args.person_role_fallback) or args.site_guard
        else None
    )
    locator = PlayerLocator(minimap_config, verbose=args.verbose)
    role_inferer = AutoNavigator()
    site_guard = SiteGuard(
        target_point=role_inferer.nearest_walkable(role_inferer.get_node("a_site")),
        arrive_distance=args.site_guard_distance,
        patrol_seconds=args.site_patrol_seconds,
        patrol_interval=args.site_patrol_interval,
        scan_turn_degrees=args.site_scan_turn_degrees,
        fire_cooldown=args.site_fire_cooldown,
        aim_gain=args.site_aim_gain,
    )
    stabilizer = PositionStabilizer(
        stable_frames=max(1, args.stable_frames),
        max_jump=args.max_position_jump,
    )

    navigator: Navigator | None = None
    anti_stuck: AntiStuck | None = None
    role: str | None = None if args.role == "auto" else args.role
    role_lock = RoleLock(required_frames=args.role_lock_frames)
    frame_index = 0
    detect_interval_frames = 1 if args.slow_test else max(1, args.detect_interval_frames)
    movement_calibrator = MovementCalibrator(
        duration_seconds=args.calibration_forward_seconds,
        min_distance=args.calibration_min_distance,
        near_walkable_tolerance=args.calibration_black_tolerance,
    )
    if args.skip_movement_calibration or not args.arduino:
        movement_calibrator.state = "done"

    mode = f"arduino:{args.arduino_port}" if args.arduino else "dry-run"
    print(f"main: start ({mode}), ESC/Ctrl+C to exit")
    print(
        "minimap config: "
        f"roi=({minimap_config.roi_left},{minimap_config.roi_top},"
        f"{minimap_config.roi_width},{minimap_config.roi_height}), "
        f"calibration_points={calibration_count}"
    )
    print("movement gate: waiting for detected and stable minimap position")
    if args.site_guard:
        print(
            "site guard: "
            f"A_distance={args.site_guard_distance:.1f}, "
            f"patrol={args.site_patrol_seconds:.2f}s, "
            f"fire_cooldown={args.site_fire_cooldown:.2f}s"
        )
    print(
        "motion control: "
        f"detect_interval={detect_interval_frames}, stable_frames={args.stable_frames}, "
        f"movement_speed={args.movement_speed:.1f}, "
        f"forward_time={args.min_forward_seconds:.2f}-{args.max_forward_seconds:.2f}s, "
        f"pause={args.action_pause_seconds:.2f}s, "
        f"turn_gain={args.turn_gain:.2f}, heading_trust={args.heading_turn_trust:.2f}, "
        f"waypoint_step={args.navigator_simplify_step}"
    )
    if movement_calibrator.state != "done":
        print(
            "movement calibration: "
            f"budget={args.calibration_forward_seconds:.2f}s, "
            f"pulse={movement_calibrator.pulse_seconds:.2f}s, "
            f"min_distance={args.calibration_min_distance:.1f}px"
        )
    if args.slow_test:
        print(
            "slow-test: "
            f"forward_pulse={args.forward_pulse_seconds:.2f}s, "
            f"pause={args.action_pause_seconds:.2f}s, detect_interval=1"
        )

    try:
        while True:
            frame = capture.grab()

            if frame_index % detect_interval_frames == 0:
                locator_role = role if role is not None else args.role
                location = locator.locate(frame, role=locator_role)
                if location is None:
                    controller.stop_all()
                    status.log(
                        "minimap_missing",
                        "wait: game/minimap not found; movement stopped",
                    )
                    frame_index += 1
                    time.sleep(0.01)
                    continue

                raw_position = location.position
                position = stabilizer.update(raw_position)
                if position is None:
                    controller.stop_all()
                    status.log(
                        "position_unstable",
                        f"wait: minimap position not stable yet raw={raw_position}; movement stopped",
                    )
                    frame_index += 1
                    time.sleep(0.01)
                    continue

                if args.role == "auto" and role is None:
                    candidate_role: str | None = None
                    candidate_source: str | None = None
                    place_role = place_detector.detect_role(frame) if place_detector is not None else None
                    if place_role is not None:
                        candidate_role = place_role
                        candidate_source = "place"
                    else:
                        inferred_role = infer_role_from_position(position, role_inferer)
                        if inferred_role is not None:
                            candidate_role = inferred_role
                            candidate_source = "spawn"
                        elif person_detector is not None:
                            detected_role = person_detector.detect_role(frame)
                            if detected_role is not None:
                                candidate_role = detected_role
                                candidate_source = "person fallback"

                    locked_role = role_lock.update(candidate_role, candidate_source)
                    if locked_role is not None:
                        role = locked_role
                        status.log("role", f"role locked from {candidate_source}: {role}", force=True)
                    elif candidate_role is not None:
                        status.log(
                            "role_pending",
                            f"role confirm: {candidate_role} from {candidate_source} "
                            f"({role_lock.count}/{role_lock.required_frames})",
                        )

                if role is None:
                    controller.stop_all()
                    status.log(
                        "role_missing",
                        f"wait: role not detected; minimap stable at {position}; movement stopped",
                    )
                    frame_index += 1
                    time.sleep(0.01)
                    continue

                heading = stabilizer.update_heading(position)
                if heading is not None:
                    status.log("heading", f"heading: {heading:.1f}")
                if args.continuous_heading_update or not controller.has_heading():
                    controller.update_heading(heading)

                if movement_calibrator.state != "done" and not role_inferer.is_walkable(position):
                    recovery_radius = 220 if args.start_max_snap_distance <= 0 else max(
                        1,
                        int(args.start_max_snap_distance),
                    )
                    snapped = role_inferer.recovery_target_toward(
                        position,
                        role_inferer.get_node("a_site"),
                        max_radius=recovery_radius,
                    )
                    snap_distance = distance(position, snapped)
                    if snap_distance <= args.calibration_black_tolerance:
                        status.log(
                            "calibration_near_black",
                            f"calibration near black: {position} -> {snapped}, dist={snap_distance:.1f}",
                        )
                    elif args.start_max_snap_distance <= 0 or snap_distance <= args.start_max_snap_distance:
                        movement_calibrator.reset_attempt()
                        controller.update_position(position)
                        controller.move_to_walkable(snapped)
                        status.log(
                            "calibration_black_recover",
                            f"calibration black recover: {position} -> {snapped}, dist={snap_distance:.1f}",
                        )
                        frame_index += 1
                        time.sleep(0.01)
                        continue
                    else:
                        controller.stop_all()
                        status.log(
                            "calibration_black_far",
                            f"wait: calibration black area too far from white {position} -> {snapped}, dist={snap_distance:.1f}",
                        )
                        frame_index += 1
                        time.sleep(0.01)
                        continue

                if not movement_calibrator.update(position, controller, role_inferer, status):
                    frame_index += 1
                    time.sleep(0.01)
                    continue

                status.log(
                    "minimap_stable",
                    f"minimap stable: raw={raw_position}, map={position}, role={role}",
                )

                navigator, anti_stuck = update_navigation(
                    role=role,
                    position=position,
                    controller=controller,
                    navigator=navigator,
                    anti_stuck=anti_stuck,
                    verbose=args.verbose,
                    enable_recovery=args.arduino and args.enable_recovery,
                    max_snap_distance=args.max_snap_distance,
                    black_snap_tolerance=args.black_snap_tolerance,
                    start_max_snap_distance=args.start_max_snap_distance,
                    simplify_step=args.navigator_simplify_step,
                    preferred_clearance_pixels=args.preferred_clearance_pixels,
                    clearance_cost_weight=args.clearance_cost_weight,
                )

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break

            if is_escape_pressed():
                print("main: esc")
                break

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("main: exit")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
