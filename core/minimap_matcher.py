from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# 项目路径和调试文件。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_IMAGE = PROJECT_ROOT / "maps" / "mirage_minimap_ref.png"
DEBUG_IMAGE = PROJECT_ROOT / "runs" / "debug" / "minimap_match_debug.png"

# 像素坐标点，格式为 (x, y)。
Point = tuple[int, int]


@dataclass
class MatchResult:
    """保存一次小地图匹配结果。"""

    matrix: np.ndarray
    matches: int
    inliers: int


class MinimapMatcher:
    """使用 ORB 匹配当前小地图和参考小地图，估计坐标变换。"""

    def __init__(
        self,
        reference_path: Path = DEFAULT_REF_IMAGE,
        max_features: int = 1500,
        min_matches: int = 12,
        verbose: bool = False,
    ) -> None:
        self.reference_path = reference_path
        self.max_features = max_features
        self.min_matches = min_matches
        self.verbose = verbose

        self.reference_image = cv2.imread(str(self.reference_path), cv2.IMREAD_COLOR)
        if self.reference_image is None:
            raise FileNotFoundError(f"找不到参考小地图：{self.reference_path}")

        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reference_keypoints, self.reference_descriptors = self._detect(self.reference_image)

    def match(self, current_minimap: np.ndarray) -> MatchResult | None:
        """匹配当前小地图到参考小地图，并返回仿射变换矩阵。"""
        current_keypoints, current_descriptors = self._detect(current_minimap)
        if current_descriptors is None or self.reference_descriptors is None:
            return None

        matches = self.matcher.match(current_descriptors, self.reference_descriptors)
        matches = sorted(matches, key=lambda item: item.distance)
        good_matches = matches[:80]

        if len(good_matches) < self.min_matches:
            return None

        current_points = np.float32([current_keypoints[item.queryIdx].pt for item in good_matches])
        reference_points = np.float32([self.reference_keypoints[item.trainIdx].pt for item in good_matches])

        # 估计相似/仿射变换：当前小地图 -> 参考小地图。
        matrix, inliers = cv2.estimateAffinePartial2D(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )
        if matrix is None or inliers is None:
            return None

        inlier_count = int(inliers.sum())
        if inlier_count < self.min_matches:
            return None

        if self.verbose:
            self._save_debug_image(current_minimap, current_keypoints, good_matches, inliers)
            print(f"match: matches={len(good_matches)}, inliers={inlier_count}")

        return MatchResult(
            matrix=matrix,
            matches=len(good_matches),
            inliers=inlier_count,
        )

    def transform_point(self, point: Point, matrix: np.ndarray) -> Point:
        """把当前小地图坐标转换到参考小地图坐标。"""
        source = np.array([point[0], point[1], 1.0], dtype=np.float32)
        mapped = matrix.dot(source)
        return int(round(float(mapped[0]))), int(round(float(mapped[1])))

    def _detect(self, image: np.ndarray):
        """提取 ORB 特征。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(gray, None)

    def _save_debug_image(
        self,
        current_minimap: np.ndarray,
        current_keypoints,
        matches,
        inliers: np.ndarray,
    ) -> None:
        """保存匹配调试图。"""
        inlier_matches = [
            match for match, is_inlier in zip(matches, inliers.ravel().tolist()) if is_inlier
        ]
        debug = cv2.drawMatches(
            current_minimap,
            current_keypoints,
            self.reference_image,
            self.reference_keypoints,
            inlier_matches[:40],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        DEBUG_IMAGE.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG_IMAGE), debug)
