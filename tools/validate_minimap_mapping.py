import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "minimap.json"
DEFAULT_MAP_IMAGE = PROJECT_ROOT / "maps" / "mirage_map.jpg"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "debug" / "minimap_mapping_validation.png"

Point = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate minimap-to-map calibration points.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="minimap.json path.")
    parser.add_argument("--map-image", type=Path, default=DEFAULT_MAP_IMAGE, help="large map image path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="debug image output path.")
    parser.add_argument("--warn-error", type=float, default=25.0, help="warning threshold in map pixels.")
    parser.add_argument("--bad-error", type=float, default=50.0, help="bad threshold in map pixels.")
    parser.add_argument(
        "--point",
        nargs=2,
        type=int,
        metavar=("MINIMAP_X", "MINIMAP_Y"),
        help="map one extra minimap point using all calibration points.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def calibration_points(config: dict[str, Any]) -> list[dict[str, Any]]:
    points = config.get("calibration_points") or []
    valid_points = []
    for point in points:
        if "minimap" not in point or "map" not in point:
            continue
        if len(point["minimap"]) != 2 or len(point["map"]) != 2:
            continue
        valid_points.append(point)
    return valid_points


def fit_xy_mapping(points: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None

    src = np.array([point["minimap"] for point in points], dtype=np.float64)
    dst = np.array([point["map"] for point in points], dtype=np.float64)
    source_x = np.column_stack([src[:, 0], np.ones(len(src), dtype=np.float64)])
    source_y = np.column_stack([src[:, 1], np.ones(len(src), dtype=np.float64)])
    scale_x, offset_x = np.linalg.lstsq(source_x, dst[:, 0], rcond=None)[0]
    scale_y, offset_y = np.linalg.lstsq(source_y, dst[:, 1], rcond=None)[0]
    return float(scale_x), float(offset_x), float(scale_y), float(offset_y)


def map_point(minimap_point: Point, mapping: tuple[float, float, float, float]) -> Point:
    scale_x, offset_x, scale_y, offset_y = mapping
    map_x = int(round(scale_x * minimap_point[0] + offset_x))
    map_y = int(round(scale_y * minimap_point[1] + offset_y))
    return max(0, min(1000, map_x)), max(0, min(1000, map_y))


def point_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def draw_triangle(image: np.ndarray, center: Point, color: tuple[int, int, int]) -> None:
    x, y = center
    points = np.array(
        [
            [x, y - 10],
            [x - 9, y + 8],
            [x + 9, y + 8],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, points, color)
    cv2.polylines(image, [points], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def error_label(error: float, warn_error: float, bad_error: float) -> str:
    if error >= bad_error:
        return "BAD"
    if error >= warn_error:
        return "WARN"
    return "OK"


def validate_fit(
    points: list[dict[str, Any]],
    mapping: tuple[float, float, float, float],
    warn_error: float,
    bad_error: float,
) -> list[dict[str, Any]]:
    rows = []
    for point in points:
        name = str(point.get("name", "unnamed"))
        minimap_point = (int(point["minimap"][0]), int(point["minimap"][1]))
        expected = (int(point["map"][0]), int(point["map"][1]))
        predicted = map_point(minimap_point, mapping)
        error = point_distance(predicted, expected)
        rows.append(
            {
                "name": name,
                "minimap": minimap_point,
                "expected": expected,
                "predicted": predicted,
                "error": error,
                "label": error_label(error, warn_error, bad_error),
            }
        )
    return rows


def leave_one_out(
    points: list[dict[str, Any]],
    warn_error: float,
    bad_error: float,
) -> list[dict[str, Any]]:
    rows = []
    if len(points) < 3:
        return rows

    for index, point in enumerate(points):
        training_points = points[:index] + points[index + 1 :]
        mapping = fit_xy_mapping(training_points)
        if mapping is None:
            continue

        name = str(point.get("name", "unnamed"))
        minimap_point = (int(point["minimap"][0]), int(point["minimap"][1]))
        expected = (int(point["map"][0]), int(point["map"][1]))
        predicted = map_point(minimap_point, mapping)
        error = point_distance(predicted, expected)
        rows.append(
            {
                "name": name,
                "expected": expected,
                "predicted": predicted,
                "error": error,
                "label": error_label(error, warn_error, bad_error),
            }
        )
    return rows


def print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  not enough calibration points")
        return

    for row in rows:
        print(
            f"  {row['label']:4} {row['name']:<16} "
            f"expected={row['expected']} predicted={row['predicted']} "
            f"error={row['error']:.2f}px"
        )

    errors = [row["error"] for row in rows]
    print(
        f"  summary: avg={sum(errors) / len(errors):.2f}px, "
        f"max={max(errors):.2f}px, points={len(rows)}"
    )


def draw_debug_image(
    map_image_path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    warn_error: float,
    bad_error: float,
) -> None:
    image = cv2.imread(str(map_image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"\ndebug image skipped: map image not found: {map_image_path}")
        return

    for row in rows:
        expected = row["expected"]
        predicted = row["predicted"]
        error = row["error"]
        color = (0, 200, 0)
        if error >= bad_error:
            color = (0, 0, 255)
        elif error >= warn_error:
            color = (0, 180, 255)

        cv2.circle(image, expected, 7, (0, 200, 0), -1)
        cv2.circle(image, expected, 8, (0, 0, 0), 1, cv2.LINE_AA)
        draw_triangle(image, predicted, color)
        cv2.line(image, expected, predicted, color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{row['name']} {error:.1f}",
            (predicted[0] + 8, predicted[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(image, "green circle=expected, triangle=predicted", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"\ndebug image saved: {output_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    points = calibration_points(config)
    if len(points) < 2:
        raise ValueError("need at least 2 calibration_points in minimap.json")

    mapping = fit_xy_mapping(points)
    if mapping is None:
        raise ValueError("failed to fit minimap mapping")

    scale_x, offset_x, scale_y, offset_y = mapping
    print(f"config: {args.config}")
    print(f"mapping: map_x = {scale_x:.6f} * minimap_x + {offset_x:.3f}")
    print(f"mapping: map_y = {scale_y:.6f} * minimap_y + {offset_y:.3f}")

    fit_rows = validate_fit(points, mapping, args.warn_error, args.bad_error)
    print_rows("Fit error using all calibration points", fit_rows)

    loo_rows = leave_one_out(points, args.warn_error, args.bad_error)
    print_rows("Leave-one-out error", loo_rows)

    if args.point is not None:
        minimap_point = (int(args.point[0]), int(args.point[1]))
        predicted = map_point(minimap_point, mapping)
        print(f"\nextra point: minimap={minimap_point} -> map={predicted}")

    draw_debug_image(args.map_image, args.output, fit_rows, args.warn_error, args.bad_error)


if __name__ == "__main__":
    main()
