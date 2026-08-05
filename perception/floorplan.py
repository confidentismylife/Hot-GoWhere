"""Floor plan loader — real building layouts for evacuation simulation.

Supports:
  1. Programmatic mall layout (北京朝阳大悦城 1F inspired)
  2. PNG image loading (black = wall, white = walkable, red = exit)
  3. YAML wall-segment definitions

The floor plan grid is 0.5m resolution, matching the disaster simulation grid.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import yaml
import os


@dataclass
class FloorPlan:
    """2D floor plan for evacuation simulation.

    Attributes:
        name: Human-readable name
        width: World width in meters
        height: World height in meters
        grid: (H, W) uint8 array, 0=walkable, 1=wall, 2=exit_zone
        exits: List of (x, y) exit center positions
        exit_names: Chinese labels for each exit
        fire_stairs: (x, y) positions of fire stair doors
        obstacles: List of (center, radius) for non-wall obstacles (pillars, etc.)
        disaster_origin_default: Recommended fire origin (x, y)
        rooms: Optional room/shop labels for visualization
    """
    name: str
    width: float
    height: float
    grid: np.ndarray          # (H, W) 0=walkable 1=wall 2=exit_zone
    exits: List[Tuple[float, float]]
    exit_names: List[str]
    fire_stairs: List[Tuple[float, float]]
    obstacles: List[dict]     # {center: [x,y], radius: r, label: str}
    disaster_origin_default: Tuple[float, float]
    rooms: Optional[List[dict]] = None  # [{x,y,w,h,label}]

    @property
    def grid_shape(self):
        return self.grid.shape

    def is_walkable(self, x: float, y: float) -> bool:
        """Check if world coordinate (x, y) is walkable."""
        resolution = self.width / self.grid.shape[1]
        col = int(x / resolution)
        row = int(y / resolution)
        if row < 0 or row >= self.grid.shape[0]:
            return False
        if col < 0 or col >= self.grid.shape[1]:
            return False
        return self.grid[row, col] != 1

    def wall_distance(self, x: float, y: float, max_dist: float = 5.0) -> float:
        """Minimum distance to nearest wall cell within max_dist. Returns max_dist if no wall nearby."""
        resolution = self.width / self.grid.shape[1]
        cells = int(max_dist / resolution) + 1
        col = int(x / resolution)
        row = int(y / resolution)
        min_dist = max_dist
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                r = row + dr
                c = col + dc
                if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
                    if self.grid[r, c] == 1:
                        wx = (c + 0.5) * resolution
                        wy = (r + 0.5) * resolution
                        d = np.sqrt((x - wx)**2 + (y - wy)**2)
                        if d < min_dist:
                            min_dist = d
        return min_dist

    def wall_normal(self, x: float, y: float) -> Tuple[float, float]:
        """Approximate wall normal direction (away from nearest wall)."""
        resolution = self.width / self.grid.shape[1]
        cells = 3
        col = int(x / resolution)
        row = int(y / resolution)
        fx, fy = 0.0, 0.0
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                r = row + dr
                c = col + dc
                if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
                    if self.grid[r, c] == 1:
                        wx = (c + 0.5) * resolution
                        wy = (r + 0.5) * resolution
                        dx = x - wx
                        dy = y - wy
                        dist_sq = dx*dx + dy*dy + 1e-6
                        fx += dx / dist_sq
                        fy += dy / dist_sq
        norm = np.sqrt(fx*fx + fy*fy)
        if norm < 1e-8:
            return (0.0, 0.0)
        return (fx / norm, fy / norm)


# ================================================================
# Built-in floor plans
# ================================================================

def build_chaoyang_joycity_1f() -> FloorPlan:
    """北京朝阳大悦城 1F inspired layout.

    150m × 80m, central atrium, 4 corridors, 8 exits, 30+ shops.
    Realistic Chinese shopping mall floor plan.
    """
    resolution = 0.5  # meters per cell
    width = 150.0
    height = 80.0
    cols = int(width / resolution)   # 300
    rows = int(height / resolution)  # 160

    grid = np.zeros((rows, cols), dtype=np.uint8)

    # === Outer walls (4 sides, 2m thick = 4 cells) ===
    wall_thick = 4
    grid[:wall_thick, :] = 1           # top wall
    grid[-wall_thick:, :] = 1          # bottom wall
    grid[:, :wall_thick] = 1           # left wall
    grid[:, -wall_thick:] = 1          # right wall

    # === Central Atrium (open space, 40×25m, centered) ===
    # Atrium is bounded by shops on all sides, creating a ring corridor
    atrium_cx = cols // 2   # 150
    atrium_cy = rows // 2   # 80
    atrium_w = int(40 / resolution)   # 80 cells
    atrium_h = int(25 / resolution)   # 50 cells

    # Atrium is walkable (already 0), so no walls needed
    # But the atrium has escalators in the center

    # Escalator set 1 (center of atrium)
    esc_w = int(3 / resolution)   # 6 cells
    esc_h = int(6 / resolution)   # 12 cells
    esc1_cx = atrium_cx - int(5 / resolution)
    esc1_cy = atrium_cy
    esc2_cx = atrium_cx + int(5 / resolution)
    esc2_cy = atrium_cy
    for esc_cx, esc_cy in [(esc1_cx, esc1_cy), (esc2_cx, esc2_cy)]:
        r1 = esc_cy - esc_h // 2
        r2 = esc_cy + esc_h // 2
        c1 = esc_cx - esc_w // 2
        c2 = esc_cx + esc_w // 2
        grid[r1:r2, c1:c2] = 1

    # === Shop walls around the atrium ===
    # North corridor shops (4 shops, 25m deep each)
    shop_depth_n = int(18 / resolution)  # 36 cells
    # Shop walls divide the north side into 4 segments
    n_shop_start = wall_thick
    n_shop_end = wall_thick + shop_depth_n
    n_corridor = n_shop_end  # corridor is just south of shops

    # The corridor is the space between shops and atrium
    # North shops wall (bottom edge of shops)
    # grid[n_corridor, :] = already walkable

    # East corridor shops (15m deep)
    shop_depth_e = int(15 / resolution)
    e_shop_start = cols - wall_thick - shop_depth_e
    # grid[:, e_shop_start] = shop wall, but we'll define as obstacles

    # West corridor shops (15m deep)
    shop_depth_w = int(15 / resolution)
    w_shop_end = wall_thick + shop_depth_w

    # South corridor shops (20m deep)
    shop_depth_s = int(20 / resolution)
    s_shop_start = rows - wall_thick - shop_depth_s

    # === Internal walls separating shops (north side) ===
    # Divide north shops into 4 sections
    n_divs = [int(cols * 0.15), int(cols * 0.35), int(cols * 0.55), int(cols * 0.75)]
    for div_x in n_divs:
        # Vertical wall from north outer wall to corridor
        grid[wall_thick:n_corridor, div_x:div_x+2] = 1

    # === Internal walls separating shops (south side) ===
    s_divs = [int(cols * 0.20), int(cols * 0.42), int(cols * 0.60), int(cols * 0.78)]
    for div_x in s_divs:
        grid[s_shop_start:rows-wall_thick, div_x:div_x+2] = 1

    # === Internal walls separating shops (west side) ===
    w_divs = [int(rows * 0.22), int(rows * 0.45), int(rows * 0.72)]
    for div_y in w_divs:
        grid[div_y:div_y+2, wall_thick:w_shop_end] = 1

    # === Internal walls separating shops (east side) ===
    e_divs = [int(rows * 0.18), int(rows * 0.40), int(rows * 0.65)]
    for div_y in e_divs:
        grid[div_y:div_y+2, e_shop_start:cols-wall_thick] = 1

    # === Shop depth boundary walls (creating the ring corridor) ===
    # North shops southern wall (corridor side)
    grid[n_corridor:n_corridor+1, wall_thick:cols-wall_thick] = 1
    # Punch holes for shop entrances
    for div_x in n_divs:
        hole_start = div_x + 2
        hole_end = hole_start + int(8 / resolution)
        grid[n_corridor:n_corridor+1, hole_start:hole_end] = 0

    # South shops northern wall
    grid[s_shop_start-1:s_shop_start, wall_thick:cols-wall_thick] = 1
    for div_x in s_divs:
        hole_start = div_x + 2
        hole_end = hole_start + int(8 / resolution)
        grid[s_shop_start-1:s_shop_start, hole_start:hole_end] = 0

    # West shops eastern wall
    grid[wall_thick:rows-wall_thick, w_shop_end:w_shop_end+1] = 1
    for div_y in w_divs:
        hole_start = div_y + 2
        hole_end = hole_start + int(6 / resolution)
        grid[hole_start:hole_end, w_shop_end:w_shop_end+1] = 0

    # East shops western wall
    grid[wall_thick:rows-wall_thick, e_shop_start-1:e_shop_start] = 1
    for div_y in e_divs:
        hole_start = div_y + 2
        hole_end = hole_start + int(6 / resolution)
        grid[hole_start:hole_end, e_shop_start-1:e_shop_start] = 0

    # === Pillars in corridors (every 10m) ===
    pillar_r = int(0.8 / resolution)  # 1.6 cells
    # North corridor pillars
    for px in range(wall_thick + int(10/resolution), cols - wall_thick, int(10/resolution)):
        py = n_corridor + int(5 / resolution)
        for dr in range(-pillar_r, pillar_r + 1):
            for dc in range(-pillar_r, pillar_r + 1):
                if dr*dr + dc*dc <= pillar_r*pillar_r:
                    if 0 <= py+dr < rows and 0 <= px+dc < cols:
                        grid[py+dr, px+dc] = 1

    # South corridor pillars
    for px in range(wall_thick + int(10/resolution), cols - wall_thick, int(10/resolution)):
        py = s_shop_start - int(5 / resolution)
        for dr in range(-pillar_r, pillar_r + 1):
            for dc in range(-pillar_r, pillar_r + 1):
                if dr*dr + dc*dc <= pillar_r*pillar_r:
                    if 0 <= py+dr < rows and 0 <= px+dc < cols:
                        grid[py+dr, px+dc] = 1

    # === Exit zones ===
    exits = [
        (25.0, 2.0),     # 北1 - 消防楼梯
        (75.0, 2.0),     # 北2 - 主入口 (大门)
        (125.0, 2.0),    # 北3 - 消防楼梯
        (2.0, 25.0),     # 西1 - 消防楼梯
        (2.0, 55.0),     # 西2 - 侧门
        (148.0, 25.0),   # 东1 - 消防楼梯
        (148.0, 55.0),   # 东2 - 侧门
        (75.0, 78.0),    # 南 - 主入口 (大门)
    ]
    exit_names = [
        "北1-消防楼梯", "北2-主入口", "北3-消防楼梯",
        "西1-消防楼梯", "西2-侧门",
        "东1-消防楼梯", "东2-侧门",
        "南-主入口",
    ]
    fire_stairs = [(25.0, 4.0), (125.0, 4.0), (4.0, 25.0), (148.0, 25.0)]

    # Mark exit zones on grid
    for ex, ey in exits:
        ez = int(3 / resolution)  # 3m exit zone
        col = int(ex / resolution)
        row = int(ey / resolution)
        for dr in range(-ez, ez + 1):
            for dc in range(-ez, ez + 1):
                if 0 <= row+dr < rows and 0 <= col+dc < cols:
                    grid[row+dr, col+dc] = 0  # Ensure walkable

    # Carve exit openings in outer walls
    for ex, ey in exits:
        col = int(ex / resolution)
        row = int(ey / resolution)
        opening = int(3 / resolution)  # 3m opening
        if ey < 5:  # North exits
            grid[0:wall_thick, col-opening:col+opening] = 0
        elif ey > height - 5:  # South exits
            grid[-wall_thick:, col-opening:col+opening] = 0
        elif ex < 5:  # West exits
            grid[row-opening:row+opening, 0:wall_thick] = 0
        elif ex > width - 5:  # East exits
            grid[row-opening:row+opening, -wall_thick:] = 0

    # Obstacles (for compatibility with existing code)
    obstacles = [
        # Atrium center escalators are already walls in grid
        # Pillars
        {"center": [40.0, 35.0], "radius": 1.0, "label": "中庭柱"},
        {"center": [110.0, 35.0], "radius": 1.0, "label": "中庭柱"},
        {"center": [40.0, 48.0], "radius": 1.0, "label": "中庭柱"},
        {"center": [110.0, 48.0], "radius": 1.0, "label": "中庭柱"},
        # Info desk in atrium
        {"center": [75.0, 40.0], "radius": 2.5, "label": "服务台"},
    ]

    # Default fire origin — kitchen area (southeast, near restaurant zone)
    fire_origin = (120.0, 65.0)

    # Room/shop labels for visualization
    rooms = [
        {"x": 10, "y": 20, "w": 10, "h": 14, "label": "ZARA"},
        {"x": 35, "y": 20, "w": 12, "h": 14, "label": "H&M"},
        {"x": 55, "y": 20, "w": 12, "h": 14, "label": "优衣库"},
        {"x": 75, "y": 20, "w": 14, "h": 14, "label": "海底捞"},
        {"x": 95, "y": 20, "w": 12, "h": 14, "label": "西贝莜面村"},
        {"x": 115, "y": 20, "w": 12, "h": 14, "label": "星巴克"},
        {"x": 10, "y": 60, "w": 12, "h": 16, "label": "电影院"},
        {"x": 35, "y": 62, "w": 10, "h": 14, "label": "大疆"},
        {"x": 55, "y": 65, "w": 10, "h": 11, "label": "Apple"},
        {"x": 75, "y": 65, "w": 12, "h": 11, "label": "华为"},
        {"x": 95, "y": 62, "w": 10, "h": 14, "label": "泡泡玛特"},
        {"x": 115, "y": 60, "w": 12, "h": 16, "label": "餐饮区"},
    ]

    # Flip grid vertically so row 0 = y=0 (bottom), matching environment coordinate system
    # (grid was built with row 0 = top for readability, environment uses row index = y/resolution)
    grid = np.flipud(grid)

    return FloorPlan(
        name="北京朝阳大悦城 1F",
        width=width,
        height=height,
        grid=grid,
        exits=exits,
        exit_names=exit_names,
        fire_stairs=fire_stairs,
        obstacles=obstacles,
        disaster_origin_default=fire_origin,
        rooms=rooms,
    )


def load_floorplan_from_image(image_path: str, resolution: float = 0.5,
                               name: str = "Custom Floor Plan") -> FloorPlan:
    """Load floor plan from PNG image.

    Color mapping:
        Black (R<50, G<50, B<50) → wall
        Red   (R>200, G<50, B<50) → exit zone
        Green (R<50, G>200, B<50) → fire origin marker
        Other → walkable

    Exits are detected as red pixel clusters.
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    data = np.array(img)
    h_px, w_px = data.shape[:2]

    cols = w_px
    rows = h_px
    width = cols * resolution
    height = rows * resolution

    grid = np.zeros((rows, cols), dtype=np.uint8)

    # Parse pixels
    exit_pixels = []
    fire_pixels = []

    for r in range(rows):
        for c in range(cols):
            R, G, B = data[r, c]
            if R < 80 and G < 80 and B < 80:
                grid[r, c] = 1  # wall
            elif R > 200 and G < 100 and B < 100:
                exit_pixels.append((r, c))
            elif R < 100 and G > 200 and B < 100:
                fire_pixels.append((r, c))

    # Cluster exit pixels to find exit centers
    from scipy.ndimage import label as connected_label
    exit_mask = np.zeros((rows, cols), dtype=bool)
    for r, c in exit_pixels:
        exit_mask[r, c] = True

    labeled, n_exits = connected_label(exit_mask)
    exits = []
    exit_names = []
    for i in range(1, n_exits + 1):
        coords = np.argwhere(labeled == i)
        center_r = coords[:, 0].mean()
        center_c = coords[:, 1].mean()
        exits.append((float(center_c * resolution), float(center_r * resolution)))
        exit_names.append(f"出口{i}")

    # Fire origin
    if fire_pixels:
        fire_origin = (float(np.mean([p[1] for p in fire_pixels]) * resolution),
                       float(np.mean([p[0] for p in fire_pixels]) * resolution))
    else:
        fire_origin = (width / 2, height / 2)

    return FloorPlan(
        name=name,
        width=width,
        height=height,
        grid=grid,
        exits=exits,
        exit_names=exit_names,
        fire_stairs=[],
        obstacles=[],
        disaster_origin_default=fire_origin,
    )


# ================================================================
# Built-in floor plans registry
# ================================================================

def build_wuhan_baoli_1f() -> FloorPlan:
    """武汉保利广场 1F — 真实商场平面图.

    约 200m × 120m，多翼结构，中央中庭 + 南北走廊 + 东西两翼.
    6 个出口，分布在商场各方位.
    """
    resolution = 0.5
    width = 200.0
    height = 120.0
    cols = int(width / resolution)    # 400
    rows = int(height / resolution)   # 240

    grid = np.zeros((rows, cols), dtype=np.uint8)
    wall_thick = 4

    # === 外墙 ===
    grid[:wall_thick, :] = 1
    grid[-wall_thick:, :] = 1
    grid[:, :wall_thick] = 1
    grid[:, -wall_thick:] = 1

    # === 中央中庭 (50×30m, 略偏北) ===
    atrium_cx = cols // 2          # 200
    atrium_cy = rows // 2 + 20     # 140 (偏北)
    atrium_w = int(50 / resolution)  # 100
    atrium_h = int(30 / resolution)  # 60

    # 中庭周围墙体 (回廊)
    a_left = atrium_cx - atrium_w // 2
    a_right = atrium_cx + atrium_w // 2
    a_top = atrium_cy - atrium_h // 2
    a_bottom = atrium_cy + atrium_h // 2

    # 中庭内部开放 (不做墙)
    # 中庭中央有扶梯
    esc_w = int(3 / resolution)
    esc_h = int(8 / resolution)
    for esc_x in [atrium_cx - int(8/resolution), atrium_cx + int(8/resolution)]:
        r1 = atrium_cy - esc_h // 2
        r2 = atrium_cy + esc_h // 2
        c1 = esc_x - esc_w // 2
        c2 = esc_x + esc_w // 2
        grid[r1:r2, c1:c2] = 1

    # === 北翼走廊 (中庭向北到北墙) ===
    # 两条平行走廊中间是店铺
    n_corridor_w = int(6 / resolution)   # 走廊宽6m
    # 北走廊从北墙到中庭

    # === 南翼走廊 (中庭向南) ===
    s_corridor_w = int(6 / resolution)

    # === 东翼 (餐厅区, 延伸出去) ===
    e_wing_start = a_right + int(8 / resolution)
    e_wing_end = cols - wall_thick
    # 东翼走廊
    e_corridor_top = atrium_cy - int(8 / resolution)
    e_corridor_bottom = atrium_cy + int(8 / resolution)

    # === 西翼 (影院区, 延伸出去) ===
    w_wing_start = wall_thick
    w_wing_end = a_left - int(8 / resolution)
    w_corridor_top = atrium_cy - int(8 / resolution)
    w_corridor_bottom = atrium_cy + int(8 / resolution)

    # === 店铺分隔墙 ===
    # 北区店铺 (中庭上方, 4排店铺)
    n_shop_zone = a_top - wall_thick
    n_divs = [
        int(cols * 0.15), int(cols * 0.30), int(cols * 0.45),
        int(cols * 0.60), int(cols * 0.75), int(cols * 0.90)
    ]
    for div_x in n_divs:
        grid[wall_thick + int(20/resolution):n_shop_zone, div_x:div_x+2] = 1

    # 南区店铺 (中庭下方)
    s_shop_zone = a_bottom
    s_divs = [
        int(cols * 0.12), int(cols * 0.28), int(cols * 0.44),
        int(cols * 0.58), int(cols * 0.72), int(cols * 0.88)
    ]
    for div_x in s_divs:
        grid[s_shop_zone:rows-wall_thick-int(20/resolution), div_x:div_x+2] = 1

    # 东翼店铺分隔
    e_divs = [
        int(a_top + int(5/resolution)),
        int(atrium_cy - int(14/resolution)),
        int(atrium_cy + int(14/resolution)),
        int(a_bottom - int(5/resolution)),
    ]
    for div_y in e_divs:
        grid[div_y:div_y+2, e_wing_start:cols-wall_thick] = 1

    # 西翼店铺分隔
    w_divs = [
        int(a_top + int(5/resolution)),
        int(atrium_cy - int(14/resolution)),
        int(atrium_cy + int(14/resolution)),
        int(a_bottom - int(5/resolution)),
    ]
    for div_y in w_divs:
        grid[div_y:div_y+2, wall_thick:w_wing_end] = 1

    # === 走廊墙体 (形成回廊) ===
    # 中庭北边界
    grid[a_top:a_top+1, a_left:a_right+1] = 1
    for div_x in [int(cols*0.25), int(cols*0.40), int(cols*0.55), int(cols*0.70)]:
        hole_s = div_x + 2
        hole_e = hole_s + int(10/resolution)
        grid[a_top:a_top+1, hole_s:hole_e] = 0

    # 中庭南边界
    grid[a_bottom-1:a_bottom, a_left:a_right+1] = 1
    for div_x in [int(cols*0.22), int(cols*0.38), int(cols*0.54), int(cols*0.72)]:
        hole_s = div_x + 2
        hole_e = hole_s + int(10/resolution)
        grid[a_bottom-1:a_bottom, hole_s:hole_e] = 0

    # 中庭西边界
    grid[a_top:a_bottom, a_left:a_left+1] = 1
    for div_y in [int(atrium_cy - int(18/resolution)), atrium_cy, int(atrium_cy + int(18/resolution))]:
        hole_s = div_y + 2
        hole_e = hole_s + int(8/resolution)
        grid[hole_s:hole_e, a_left:a_left+1] = 0

    # 中庭东边界
    grid[a_top:a_bottom, a_right-1:a_right] = 1
    for div_y in [int(atrium_cy - int(18/resolution)), atrium_cy, int(atrium_cy + int(18/resolution))]:
        hole_s = div_y + 2
        hole_e = hole_s + int(8/resolution)
        grid[hole_s:hole_e, a_right-1:a_right] = 0

    # === 柱子 (回廊中每隔12m) ===
    pillar_r = int(0.8 / resolution)
    # 北回廊
    for px in range(a_left + int(8/resolution), a_right, int(12/resolution)):
        py = a_top - int(6/resolution)
        for dr in range(-pillar_r, pillar_r+1):
            for dc in range(-pillar_r, pillar_r+1):
                if dr*dr + dc*dc <= pillar_r*pillar_r:
                    grid[py+dr, px+dc] = 1
    # 南回廊
    for px in range(a_left + int(8/resolution), a_right, int(12/resolution)):
        py = a_bottom + int(6/resolution)
        for dr in range(-pillar_r, pillar_r+1):
            for dc in range(-pillar_r, pillar_r+1):
                if dr*dr + dc*dc <= pillar_r*pillar_r:
                    grid[py+dr, px+dc] = 1

    # === 出口 (6个) ===
    exits = [
        (55.0, 2.0),      # 北1 — 北侧主入口
        (145.0, 2.0),     # 北2 — 北侧次入口
        (2.0, 45.0),      # 西1 — 消防楼梯
        (2.0, 85.0),      # 西2 — 影院出口
        (198.0, 60.0),    # 东1 — 餐厅区出口
        (100.0, 118.0),   # 南 — 南主入口 (正门)
    ]
    exit_names = [
        "北1-主入口", "北2-次入口",
        "西1-消防楼梯", "西2-影院出口",
        "东1-餐厅出口", "南-正门入口",
    ]
    fire_stairs = [(55.0, 4.0), (4.0, 45.0), (198.0, 60.0)]

    # 切出出口开口
    for ex, ey in exits:
        col = int(ex / resolution)
        row = int(ey / resolution)
        opening = int(3 / resolution)
        if ey < 5:
            grid[0:wall_thick, col-opening:col+opening] = 0
        elif ey > height - 5:
            grid[-wall_thick:, col-opening:col+opening] = 0
        elif ex < 5:
            grid[row-opening:row+opening, 0:wall_thick] = 0
        elif ex > width - 5:
            grid[row-opening:row+opening, -wall_thick:] = 0

    # 障碍物
    obstacles = [
        {"center": [80.0, 50.0], "radius": 1.2, "label": "中庭柱"},
        {"center": [120.0, 50.0], "radius": 1.2, "label": "中庭柱"},
        {"center": [80.0, 75.0], "radius": 1.2, "label": "中庭柱"},
        {"center": [120.0, 75.0], "radius": 1.2, "label": "中庭柱"},
        {"center": [100.0, 60.0], "radius": 3.0, "label": "服务台"},
        {"center": [130.0, 30.0], "radius": 1.5, "label": "电梯间"},
        {"center": [70.0, 30.0], "radius": 1.5, "label": "电梯间"},
    ]

    # 火源 — 东南餐饮区
    fire_origin = (170.0, 100.0)

    # 店铺标注
    rooms = [
        {"x": 10, "y": 15, "w": 18, "h": 12, "label": "万达影城"},
        {"x": 45, "y": 10, "w": 14, "h": 14, "label": "优衣库"},
        {"x": 80, "y": 10, "w": 16, "h": 14, "label": "ZARA"},
        {"x": 115, "y": 10, "w": 15, "h": 14, "label": "H&M"},
        {"x": 145, "y": 10, "w": 14, "h": 14, "label": "MUJI"},
        {"x": 170, "y": 15, "w": 14, "h": 12, "label": "迪卡侬"},
        {"x": 5, "y": 55, "w": 12, "h": 10, "label": "星巴克"},
        {"x": 5, "y": 70, "w": 12, "h": 12, "label": "海底捞"},
        {"x": 170, "y": 55, "w": 14, "h": 12, "label": "西贝莜面"},
        {"x": 175, "y": 75, "w": 12, "h": 12, "label": "太二酸菜鱼"},
        {"x": 175, "y": 92, "w": 12, "h": 12, "label": "麦当劳"},
        {"x": 55, "y": 100, "w": 14, "h": 12, "label": "Apple Store"},
        {"x": 80, "y": 100, "w": 12, "h": 12, "label": "华为"},
        {"x": 50, "y": 80, "w": 12, "h": 14, "label": "泡泡玛特"},
    ]

    grid = np.flipud(grid)

    return FloorPlan(
        name="武汉保利广场 1F",
        width=width,
        height=height,
        grid=grid,
        exits=exits,
        exit_names=exit_names,
        fire_stairs=fire_stairs,
        obstacles=obstacles,
        disaster_origin_default=fire_origin,
        rooms=rooms,
    )


BUILTIN_FLOORPLANS = {
    "chaoyang_joycity_1f": build_chaoyang_joycity_1f,
    "wuhan_baoli_1f": build_wuhan_baoli_1f,
}


def get_floorplan(name: str) -> FloorPlan:
    """Get a built-in floor plan by name."""
    if name in BUILTIN_FLOORPLANS:
        return BUILTIN_FLOORPLANS[name]()
    raise ValueError(f"Unknown floor plan: {name}. Available: {list(BUILTIN_FLOORPLANS.keys())}")
