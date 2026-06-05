"""YOLO 人员检测通道 — 从CCTV画面实时检测人群位置和密度.

与VLM语义理解通道互补:
  YOLO通道: 快速(5ms)、结构化(boxes+counts)、每帧可跑
  VLM通道:  慢速(1.5s)、语义化(文本描述)、定时触发

双通道融合后喂给LLM, 提供"数值精确+语义丰富"的完整场景感知.

依赖: ultralytics (pip install ultralytics)
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DetectionBox:
    """单个检测框."""
    x: float          # 中心x (世界坐标, 米)
    y: float          # 中心y (世界坐标, 米)
    w: float          # 宽度 (米)
    h: float          # 高度 (米)
    confidence: float # 置信度 0-1
    cls: int          # 0=person, ...


@dataclass
class YOLOResult:
    """YOLO 检测的结构化输出."""
    boxes: List[DetectionBox] = field(default_factory=list)
    person_count: int = 0
    density_hotspots: List[Dict] = field(default_factory=list)
    abnormal_events: List[str] = field(default_factory=list)
    compute_time_ms: float = 0.0


class YOLODetector:
    """YOLO 实时人员检测器.

    Usage:
        det = YOLODetector(model_name="yolov8n.pt")
        det.initialize()

        for frame in cctv_frames:
            result = det.detect(frame)
            # result.person_count, result.boxes, ...
    """

    def __init__(self, model_name: str = "yolov8n.pt",
                 confidence_threshold: float = 0.35,
                 device: str = "cuda"):
        self.model_name = model_name
        self.conf_thresh = confidence_threshold
        self.device = device

        self.model = None
        self._initialized = False

        # 图像→世界坐标变换参数 (由外部标定)
        self.img_width: float = 640
        self.img_height: float = 480
        self.world_width: float = 100.0
        self.world_height: float = 60.0

        # 统计
        self.total_frames: int = 0
        self.total_time: float = 0.0

    def initialize(self):
        if self._initialized:
            return

        print(f"[YOLO] Loading {self.model_name} ...")
        t0 = time.time()

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            # 预热
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            elapsed = time.time() - t0
            self._initialized = True
            print(f"[YOLO] Loaded in {elapsed:.1f}s. Ready.")
        except ImportError:
            print("[YOLO] ultralytics not installed. "
                  "YOLO detection disabled. (pip install ultralytics)")
            self.model = None
        except Exception as e:
            print(f"[YOLO] Failed to load: {e}. Running without detection.")
            self.model = None

    def set_calibration(self, img_w: float, img_h: float,
                        world_w: float, world_h: float):
        """设置图像→世界坐标变换."""
        self.img_width = img_w
        self.img_height = img_h
        self.world_width = world_w
        self.world_height = world_h

    def detect(self, frame: np.ndarray) -> YOLOResult:
        """对一帧画面做人员检测.

        Args:
            frame: (H, W, 3) numpy RGB uint8

        Returns:
            YOLOResult with boxes, count, hotspots
        """
        if self.model is None:
            return YOLOResult()

        t0 = time.time()

        try:
            results = self.model(frame, verbose=False, conf=self.conf_thresh)
        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            return YOLOResult()

        elapsed = (time.time() - t0) * 1000
        self.total_frames += 1
        self.total_time += elapsed

        yolo_result = YOLOResult(compute_time_ms=elapsed)

        if len(results) == 0 or results[0].boxes is None:
            return yolo_result

        boxes_data = results[0].boxes
        h, w = frame.shape[:2]

        person_boxes = []
        for i in range(len(boxes_data)):
            cls_id = int(boxes_data.cls[i])
            if cls_id != 0:  # 只关心 person 类
                continue
            conf = float(boxes_data.conf[i])
            xyxy = boxes_data.xyxy[i].cpu().numpy()

            # 图像坐标 → 世界坐标 (简单线性映射)
            x1, y1, x2, y2 = xyxy
            cx_img = (x1 + x2) / 2
            cy_img = (y1 + y2) / 2

            world_x = (cx_img / w) * self.world_width
            world_y = (cy_img / h) * self.world_height
            box_w = ((x2 - x1) / w) * self.world_width
            box_h = ((y2 - y1) / h) * self.world_height

            person_boxes.append(DetectionBox(
                x=float(world_x), y=float(world_y),
                w=float(box_w), h=float(box_h),
                confidence=float(conf), cls=0,
            ))

        yolo_result.boxes = person_boxes
        yolo_result.person_count = len(person_boxes)

        # 密度热点分析 (5m 网格)
        if person_boxes:
            yolo_result.density_hotspots = self._analyze_density(person_boxes)
            yolo_result.abnormal_events = self._detect_abnormal(person_boxes)

        return yolo_result

    def _analyze_density(self, boxes: List[DetectionBox],
                         grid_size: float = 5.0) -> List[Dict]:
        """分析人群密度热点."""
        cols = int(self.world_width / grid_size) + 1
        rows = int(self.world_height / grid_size) + 1
        density = np.zeros((rows, cols), dtype=np.int32)

        for b in boxes:
            c = int(b.x / grid_size)
            r = int(b.y / grid_size)
            if 0 <= r < rows and 0 <= c < cols:
                density[r, c] += 1

        hotspots = []
        threshold = max(5, int(len(boxes) * 0.1))  # top 10% density
        for r in range(rows):
            for c in range(cols):
                if density[r, c] >= threshold:
                    hotspots.append({
                        "center": (c * grid_size + grid_size / 2,
                                   r * grid_size + grid_size / 2),
                        "count": int(density[r, c]),
                        "grid_size": grid_size,
                    })

        return sorted(hotspots, key=lambda h: h["count"], reverse=True)[:5]

    def _detect_abnormal(self, boxes: List[DetectionBox]) -> List[str]:
        """检测异常行为 (摔倒/推挤/逆行). 基于启发式规则."""
        events = []

        # 摔倒检测: 宽高比异常 (人站立时 h>w, 摔倒了可能 w>h)
        fallen = [b for b in boxes
                  if b.confidence > 0.5 and b.w > b.h * 1.5]
        if fallen:
            events.append(f"疑似摔倒: {len(fallen)}人")

        # 极端密度检测
        if len(boxes) > 0:
            # 检查是否有局部区域 > 3人/m²
            cols = int(self.world_width / 2) + 1
            rows = int(self.world_height / 2) + 1
            fine_density = np.zeros((rows, cols), dtype=np.int32)
            for b in boxes:
                c = int(b.x / 2)
                r = int(b.y / 2)
                if 0 <= r < rows and 0 <= c < cols:
                    fine_density[r, c] += 1
            max_density = fine_density.max()
            if max_density > 12:  # 12人/4m² = 3人/m²
                events.append(f"极端拥挤: 局部密度{max_density/4:.0f}人/m²")

        return events

    @property
    def stats(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "avg_time_ms": self.total_time / max(1, self.total_frames),
        }

    def shutdown(self):
        if self.model is not None:
            del self.model
            self.model = None
