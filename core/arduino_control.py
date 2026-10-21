import argparse
import math
import time
from pathlib import Path
from typing import Iterable

try:
    import serial
except ImportError:
    serial = None


# 计算项目根目录，方便后续扩展读取配置文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 像素坐标点，格式为 (x, y)。
Point = tuple[int, int]


class RateLimitedLogger:
    """限制控制台日志频率，避免串口控制时刷屏。"""

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


class ArduinoController:
    """通过串口向 Arduino 发送移动和视角控制命令。"""

    def __init__(
        self,
        port: str = "COM6",
        baudrate: int = 9600,
        timeout: float = 1.0,
        command_delay: float = 0.05,
        angle_tolerance: float = 5.0,
        heading_offset: float = 0.0,
        invert_turn: bool = False,
        mouse_pixels_per_degree: float = 2.25,
        turn_gain: float = 1.25,
        heading_turn_trust: float = 0.6,
        max_turn_degrees: float = 180.0,
        slow_mode: bool = False,
        forward_pulse_seconds: float = 0.08,
        action_pause_seconds: float = 0.25,
        turn_pause_seconds: float = 0.08,
        movement_speed: float = 120.0,
        forward_distance_factor: float = 0.65,
        min_forward_seconds: float = 0.04,
        max_forward_seconds: float = 0.14,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        # 保存串口和控制参数。
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.command_delay = command_delay
        self.angle_tolerance = angle_tolerance
        self.heading_offset = heading_offset
        self.invert_turn = invert_turn
        self.mouse_pixels_per_degree = mouse_pixels_per_degree
        self.turn_gain = max(0.1, turn_gain)
        self.heading_turn_trust = max(0.0, min(1.0, heading_turn_trust))
        self.max_turn_degrees = max_turn_degrees
        self.slow_mode = slow_mode
        self.forward_pulse_seconds = forward_pulse_seconds
        self.action_pause_seconds = action_pause_seconds
        self.turn_pause_seconds = turn_pause_seconds
        self.movement_speed = movement_speed
        self.forward_distance_factor = max(0.1, min(1.0, forward_distance_factor))
        self.min_forward_seconds = min_forward_seconds
        self.max_forward_seconds = max_forward_seconds
        self.dry_run = dry_run
        self.logger = RateLimitedLogger(enabled=verbose)
        self.connection = None
        self.verbose = verbose
        self.heading_probe_duration = 0.12
        self.heading_probe_interval = 0.8
        self.last_heading_probe_time = 0.0
        self.heading_probe_count = 0
        self.max_heading_probe_count = 2
        self.feedback_commands = ["forward", "right", "left", "backward"]
        self.feedback_index = 0
        self.active_feedback_command: str | None = None
        self.last_feedback_target: Point | None = None
        self.last_feedback_distance: float | None = None
        self.last_feedback_time = 0.0
        self.feedback_interval = 0.35
        self.feedback_improve_distance = 1.5
        self.turn_step_degrees = 8.0
        self.rotate_speed_degrees = 180.0
        self.last_rotate_time = time.time()
        self.min_rotate_interval = 0.04
        self.heading_correction_alpha = 0.75
        self.last_turn_time = 0.0
        self.heading_update_cooldown = 0.2
        self.require_recheck_after_turn = False
        self.heading_scan_interval = 0.75
        self.heading_scan_turn_degrees = 35.0
        self.heading_scan_forward_seconds = 0.10
        self.last_heading_scan_time = 0.0
        self.heading_scan_count = 0
        self.corner_prepare_distance = 34.0
        self.corner_turn_distance = 14.0
        self.corner_angle_threshold = 45.0
        self.corner_forward_seconds = 0.055
        self.continuous_forward_distance = 24.0

        # 当前朝向角度，单位为度；0 表示朝右，90 表示朝上。
        self.heading_degrees: float | None = None

        # dry_run 模式只打印命令，不打开串口，方便本机调试。
        if not self.dry_run:
            self._connect()

    def _connect(self) -> None:
        """初始化与 Arduino 的串口通信。"""
        if serial is None:
            raise ImportError("缺少 pySerial，请先安装：pip install pyserial")

        self.connection = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
        )

        # 给 Arduino 一点启动时间，避免刚连接时丢命令。
        time.sleep(2.0)
        self.stop()
        self.logger.log(f"已连接 Arduino：{self.port}")

    def close(self) -> None:
        """关闭串口连接。"""
        if self.connection is not None and self.connection.is_open:
            self.stop()
            time.sleep(0.05)
            self.stop()
            self.connection.close()
            self.logger.log("串口已关闭")

    def send_command(self, command: str) -> None:
        """通过串口发送一条控制命令。"""
        clean_command = command.strip()
        if not clean_command:
            return

        # 命令以换行结尾，方便 Arduino 端按行读取。
        payload = f"{clean_command}\n".encode("utf-8")

        if self.dry_run:
            self.logger.log(f"arduino: {clean_command}")
            return

        if self.connection is None or not self.connection.is_open:
            raise RuntimeError("Arduino 串口未连接。")

        self.connection.write(payload)
        self.connection.flush()
        time.sleep(self.command_delay)

    def stop(self) -> None:
        """停止所有移动。"""
        self.send_command("stop")

    def forward(self) -> None:
        """控制角色前进。"""
        self.send_command("forward")

    def forward_pulse(self, duration_seconds: float | None = None) -> None:
        """Tap forward briefly, then stop so the next camera frame can update position."""
        duration = self.forward_pulse_seconds if duration_seconds is None else duration_seconds
        self.forward()
        time.sleep(max(0.0, duration))
        self.stop()
        if self.action_pause_seconds > 0:
            time.sleep(self.action_pause_seconds)

    def forward_by_distance(self, distance_to_target: float) -> None:
        """Move for time estimated from map distance and calibrated speed."""
        if distance_to_target > self.continuous_forward_distance:
            if self.verbose:
                self.logger.log(f"forward hold: distance={distance_to_target:.1f}")
            self.forward()
            return

        if self.movement_speed <= 0:
            duration = self.forward_pulse_seconds
        else:
            duration = (distance_to_target * self.forward_distance_factor) / self.movement_speed

        duration = max(self.min_forward_seconds, min(self.max_forward_seconds, duration))
        if self.verbose:
            self.logger.log(
                f"forward timed: distance={distance_to_target:.1f}, "
                f"speed={self.movement_speed:.1f}, seconds={duration:.2f}"
            )
        self.forward_pulse(duration)

    def backward(self, duration_seconds: float = 0.35) -> None:
        """控制角色后退一小段时间。"""
        duration_ms = max(0, int(round(duration_seconds * 1000)))
        self.send_command(f"backward:{duration_ms}")
        time.sleep(max(0.0, duration_seconds))
        self.stop()

    def movement_pulse(self, command: str, duration_seconds: float) -> None:
        """Press one movement direction briefly, then release all movement keys."""
        if command == "forward":
            self.forward()
        elif command == "left":
            self.left()
        elif command == "right":
            self.right()
        elif command == "backward":
            duration_ms = max(0, int(round(duration_seconds * 1000)))
            self.send_command(f"backward:{duration_ms}")
            time.sleep(max(0.0, duration_seconds))
            self.stop()
            return
        else:
            self.stop()
            return

        time.sleep(max(0.0, duration_seconds))
        self.stop()

    def left(self) -> None:
        """控制角色向左移动。"""
        self.send_command("left")

    def right(self) -> None:
        """控制角色向右移动。"""
        self.send_command("right")

    def jump(self) -> None:
        """控制角色跳跃。"""
        self.send_command("jump")

    def aim_pixels(self, x_pixels: int, y_pixels: int) -> None:
        """Move mouse by a small screen-space correction."""
        self.stop()
        self.send_command(f"aim:{int(x_pixels)},{int(y_pixels)}")

    def fire(self) -> None:
        """Click the left mouse button once."""
        print("[射击] 已向 Arduino 发送 fire 命令")
        self.send_command("fire")

    def turn_left(self) -> None:
        """控制视角向左转。"""
        self.send_command("turn_left")
        self._apply_turn_estimate(self.turn_step_degrees)
        self.last_turn_time = time.time()

    def turn_right(self) -> None:
        """控制视角向右转。"""
        self.send_command("turn_right")
        self._apply_turn_estimate(-self.turn_step_degrees)
        self.last_turn_time = time.time()

    def turn_degrees(self, delta_degrees: float) -> None:
        """Turn once by the requested angle delta using calibrated mouse pixels."""
        clamped_delta = max(-self.max_turn_degrees, min(self.max_turn_degrees, delta_degrees))
        # 地图角度为逆时针正方向；鼠标水平正方向通常是向右转，所以默认取反。
        pixel_delta = int(round(-clamped_delta * self.mouse_pixels_per_degree * self.turn_gain))
        if pixel_delta == 0:
            return

        if self.invert_turn:
            pixel_delta = -pixel_delta

        if self.verbose:
            print(f"turn once: delta={clamped_delta:.1f}, pixels={pixel_delta}")

        self.stop()
        self.send_command(f"turn:{pixel_delta}")
        if self.heading_degrees is not None:
            self.heading_degrees = normalize_angle(
                self.heading_degrees + clamped_delta * self.heading_turn_trust
            )
        self.last_turn_time = time.time()
        self.require_recheck_after_turn = True
        if self.turn_pause_seconds > 0:
            time.sleep(self.turn_pause_seconds)

    def update_heading(self, heading_degrees: float) -> None:
        """更新当前角色朝向角度。"""
        measured_heading = normalize_angle(heading_degrees + self.heading_offset)
        if self.heading_degrees is None:
            self.heading_degrees = measured_heading
            self.heading_scan_count = 0
            return

        if time.time() - self.last_turn_time < self.heading_update_cooldown:
            return

        self.heading_degrees = lerp_angle(
            self.heading_degrees,
            measured_heading,
            self.heading_correction_alpha,
        )
        self.heading_scan_count = 0

    def has_heading(self) -> bool:
        """判断是否已经初始化朝向。"""
        return self.heading_degrees is not None

    def _apply_turn_estimate(self, delta_degrees: float) -> None:
        """根据自己发送的转向命令同步更新内部朝向估计。"""
        if self.heading_degrees is None:
            return
        if self.invert_turn:
            delta_degrees = -delta_degrees
        self.heading_degrees = normalize_angle(
            self.heading_degrees + delta_degrees * self.heading_turn_trust
        )

    def angle_to_target(self, current_position: Point, target: Point) -> float:
        """计算当前坐标到目标 waypoint 的地图角度。"""
        return calculate_target_angle(current_position, target)

    def angle_difference(self, current_position: Point, target: Point) -> float:
        """计算当前朝向和目标方向之间的最短角度差。"""
        if self.heading_degrees is None:
            return 0.0
        target_angle = self.angle_to_target(current_position, target)
        return shortest_angle_delta(self.heading_degrees, target_angle)

    def face_target(self, current_position: Point, target: Point) -> float:
        """根据角度差控制视角转向，使角色朝向目标。"""
        if self.heading_degrees is None:
            return 0.0

        target_angle = self.angle_to_target(current_position, target)
        angle_delta = shortest_angle_delta(self.heading_degrees, target_angle)
        if self.verbose:
            heading_text = "none" if self.heading_degrees is None else f"{self.heading_degrees:.1f}"
            print(
                f"heading={heading_text}, "
                f"target_angle={target_angle:.1f}, delta={angle_delta:.1f}"
            )

        if abs(angle_delta) <= self.angle_tolerance:
            return angle_delta

        self.turn_degrees(angle_delta)
        self.last_rotate_time = time.time()

        return angle_delta

    def move_to_waypoint(
        self,
        current_position: Point,
        target: Point,
        next_target: Point | None = None,
    ) -> None:
        """朝单个 waypoint 转向并前进。"""
        if self.heading_degrees is None:
            self.probe_initial_heading()
            return

        distance_to_target = point_distance(current_position, target)
        corner_turn = self._corner_turn_degrees(current_position, target, next_target)
        if corner_turn is not None and distance_to_target <= self.corner_turn_distance:
            angle_delta = self.face_target(current_position, next_target)  # type: ignore[arg-type]
            if abs(angle_delta) <= self.angle_tolerance and not self.require_recheck_after_turn:
                self.forward_pulse(self.corner_forward_seconds)
            else:
                self.stop()
            return

        angle_delta = self.face_target(current_position, target)

        # 角度基本对齐后再前进，减少撞墙概率。
        if abs(angle_delta) <= self.angle_tolerance:
            if self.require_recheck_after_turn:
                self.require_recheck_after_turn = False
                self.stop()
                return

            if self.slow_mode:
                self.forward_pulse()
            elif corner_turn is not None and distance_to_target <= self.corner_prepare_distance:
                self.forward_pulse(self.corner_forward_seconds)
            else:
                self.forward_by_distance(distance_to_target)
        else:
            self.stop()

    def _corner_turn_degrees(
        self,
        current_position: Point,
        target: Point,
        next_target: Point | None,
    ) -> float | None:
        """Return the upcoming corner angle when the next segment is a sharp turn."""
        if next_target is None or next_target == target:
            return None

        incoming_angle = calculate_target_angle(current_position, target)
        outgoing_angle = calculate_target_angle(target, next_target)
        turn_degrees = abs(shortest_angle_delta(incoming_angle, outgoing_angle))
        if turn_degrees < self.corner_angle_threshold:
            return None
        return turn_degrees

    def probe_initial_heading(self) -> None:
        """开局只短促前进一步，用原点和新点计算初始朝向。"""
        if self.heading_probe_count >= self.max_heading_probe_count:
            self.stop()
            return

        now = time.time()
        if now - self.last_heading_probe_time < self.heading_probe_interval:
            self.stop()
            return

        self.heading_probe_count += 1
        self.last_heading_probe_time = now
        if self.verbose:
            self.logger.log("heading probe once")
        self.forward()
        time.sleep(self.heading_probe_duration)
        self.stop()

    def move_by_distance_feedback(self, current_position: Point, target: Point) -> None:
        """不识别朝向时，用距离反馈自动选择 W/A/D/S 中更接近目标的方向。"""
        now = time.time()
        current_distance = point_distance(current_position, target)

        if self.last_feedback_target != target:
            self.stop()
            self.feedback_index = 0
            self.feedback_commands = self._ordered_feedback_commands(current_position, target)
            self.active_feedback_command = None
            self.last_feedback_distance = current_distance
            self.last_feedback_target = target
            self.last_feedback_time = 0.0

        if self.active_feedback_command is None:
            self._start_feedback_command()
            self.last_feedback_distance = current_distance
            self.last_feedback_time = now
            return

        if now - self.last_feedback_time < self.feedback_interval:
            self._send_feedback_command(self.active_feedback_command)
            return

        improved = (
            self.last_feedback_distance is not None
            and current_distance < self.last_feedback_distance - self.feedback_improve_distance
        )
        if improved:
            self.last_feedback_distance = current_distance
            self.last_feedback_time = now
            self._send_feedback_command(self.active_feedback_command)
            if self.verbose:
                self.logger.log(f"feedback keep: {self.active_feedback_command}")
            return

        self.stop()
        self.feedback_index = (self.feedback_index + 1) % len(self.feedback_commands)
        self._start_feedback_command()
        self.last_feedback_distance = current_distance
        self.last_feedback_time = now

    def _start_feedback_command(self) -> None:
        """切换到下一个反馈移动方向。"""
        self.active_feedback_command = self.feedback_commands[self.feedback_index]
        if self.verbose:
            self.logger.log(f"feedback try: {self.active_feedback_command}")
        self._send_feedback_command(self.active_feedback_command)

    def _ordered_feedback_commands(self, current_position: Point, target: Point) -> list[str]:
        """没有朝向时按地图目标方向排序盲试移动，避免总是先向左。"""
        dx = target[0] - current_position[0]
        dy = target[1] - current_position[1]

        horizontal = "right" if dx >= 0 else "left"
        opposite_horizontal = "left" if horizontal == "right" else "right"
        vertical = "forward" if dy <= 0 else "backward"
        opposite_vertical = "backward" if vertical == "forward" else "forward"

        if abs(dx) >= abs(dy):
            return [horizontal, vertical, opposite_vertical, opposite_horizontal]
        return [vertical, horizontal, opposite_horizontal, opposite_vertical]

    def _send_feedback_command(self, command: str) -> None:
        """发送反馈控制选中的移动命令。"""
        if command == "forward":
            self.forward()
        elif command == "left":
            self.left()
        elif command == "right":
            self.right()
        elif command == "backward":
            self.backward(0.12)

    def probe_heading_by_mouse_scan(self) -> None:
        """没有朝向时，用鼠标扫视加短促前进制造位移，再由主循环反推朝向。"""
        now = time.time()
        if now - self.last_heading_scan_time < self.heading_scan_interval:
            self.stop()
            return

        self.last_heading_scan_time = now
        scan_turns = (0.0, -35.0, 70.0, -70.0, 70.0, -70.0)
        turn_delta = scan_turns[self.heading_scan_count % len(scan_turns)]
        self.heading_scan_count += 1

        if self.verbose:
            print(f"heading scan: turn={turn_delta:.1f}, forward={self.heading_scan_forward_seconds:.2f}s")

        if abs(turn_delta) > 0.1:
            self.turn_degrees(turn_delta)
        self.forward_pulse(self.heading_scan_forward_seconds)

    def wait_for_heading(self) -> None:
        """没有可靠朝向时先停止，避免为了探测方向而撞进黑色障碍区。"""
        now = time.time()
        if now - self.last_heading_probe_time < self.heading_probe_interval:
            self.stop()
            return

        self.last_heading_probe_time = now
        if self.verbose:
            self.logger.log("wait heading")
        self.stop()

    def follow_waypoints(
        self,
        waypoints: Iterable[Point],
        current_position: Point,
        waypoint_delay: float = 0.5,
    ) -> None:
        """按顺序向 Arduino 发送 waypoint 跟随命令。"""
        for waypoint in waypoints:
            self.move_to_waypoint(current_position, waypoint)
            time.sleep(waypoint_delay)

    def recover_from_stuck(self) -> None:
        """卡死恢复：停止、后退、随机转向、跳跃、继续前进。"""
        self.stop()
        self.backward(0.35)

        # 根据当前时间简单切换左右方向，避免每次都往同一侧修正。
        if int(time.time() * 10) % 2 == 0:
            self.turn_left()
        else:
            self.turn_right()

        self.jump()
        self.forward()

    def replan_and_move(self, current_position: Point, target: Point) -> None:
        """路径修正后继续朝目标移动；A* 重规划由导航模块负责。"""
        self.move_to_waypoint(current_position, target)


def normalize_angle(angle: float) -> float:
    """把角度归一化到 0 到 360 度。"""
    return angle % 360.0


def estimate_current_angle(
    previous_position: Point,
    current_position: Point,
    min_move_distance: float = 1.0,
) -> float | None:
    """Estimate heading with FPS convention: 0=right, 90=up, 180=left, 270=down."""
    dx = current_position[0] - previous_position[0]
    dy = current_position[1] - previous_position[1]
    if math.hypot(dx, dy) < min_move_distance:
        return None
    return normalize_angle(math.degrees(math.atan2(-dy, dx)))


def calculate_target_angle(current_position: Point, target: Point) -> float:
    """Calculate target heading with FPS convention: 0=right, 90=up."""
    dx = target[0] - current_position[0]
    dy = target[1] - current_position[1]
    return normalize_angle(math.degrees(math.atan2(-dy, dx)))


def smooth_rotate_angle(
    current_angle: float,
    target_angle: float,
    rotate_speed: float,
    delta_time: float,
) -> float:
    """Move current_angle toward target_angle through the shortest arc."""
    angle_delta = shortest_angle_delta(current_angle, target_angle)
    max_step = max(0.0, rotate_speed * delta_time)

    if abs(angle_delta) <= max_step:
        return normalize_angle(target_angle)

    if angle_delta > 0:
        return normalize_angle(current_angle + max_step)
    return normalize_angle(current_angle - max_step)


def lerp_angle(current_angle: float, target_angle: float, alpha: float) -> float:
    """Interpolate angles while respecting 0/360 wrapping."""
    alpha = max(0.0, min(1.0, alpha))
    angle_delta = shortest_angle_delta(current_angle, target_angle)
    return normalize_angle(current_angle + angle_delta * alpha)


def point_distance(a: Point, b: Point) -> float:
    """计算两个地图坐标之间的距离。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def shortest_angle_delta(current_angle: float, target_angle: float) -> float:
    """计算从当前角度转到目标角度的最短差值，范围为 -180 到 180。"""
    return (target_angle - current_angle + 180.0) % 360.0 - 180.0


def parse_args() -> argparse.Namespace:
    """解析命令行参数，方便手动测试串口命令。"""
    parser = argparse.ArgumentParser(description="Arduino 串口控制测试。")
    parser.add_argument("--port", default="COM3", help="Arduino 串口号，默认 COM3。")
    parser.add_argument("--baudrate", type=int, default=9600, help="串口波特率。")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不打开串口。")
    parser.add_argument("--command", default="stop", help="要发送的测试命令。")
    return parser.parse_args()


def main() -> None:
    """命令行测试入口。"""
    args = parse_args()
    controller = ArduinoController(
        port=args.port,
        baudrate=args.baudrate,
        dry_run=args.dry_run,
        verbose=True,
    )

    try:
        controller.send_command(args.command)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
