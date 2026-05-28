"""Simplified disaster simulation using cellular automata + Gaussian spread.
Runs entirely on CPU — fast enough for real-time with hundreds of agents."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class HazardCell:
    smoke: float = 0.0         # 0-1
    temperature: float = 25.0  # Celsius
    structural: float = 1.0    # 1.0 = intact, 0.0 = collapsed
    on_fire: bool = False


@dataclass
class EnvironmentSnapshot:
    """The environment as agents perceive it at a given tick."""
    tick: int
    timestamp: float
    width: float
    height: float
    grid: np.ndarray           # 2D array of HazardCell-like data
    grid_resolution: float     # meters per cell

    exits: List[Tuple[float, float]]
    obstacles: List[dict]      # [{center: (x,y), radius: r}, ...]
    official_broadcast: str = ""
    disaster_type: str = "fire"

    def smoke_at(self, pos: np.ndarray) -> float:
        r, c = self._to_grid(pos)
        if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
            return float(self.grid[r, c, 0])
        return 0.0

    def temperature_at(self, pos: np.ndarray) -> float:
        r, c = self._to_grid(pos)
        if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
            return float(self.grid[r, c, 1])
        return 25.0

    def structural_at(self, pos: np.ndarray) -> float:
        r, c = self._to_grid(pos)
        if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
            return float(self.grid[r, c, 2])
        return 1.0

    def is_on_fire(self, pos: np.ndarray) -> bool:
        r, c = self._to_grid(pos)
        if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
            return bool(self.grid[r, c, 3])
        return False

    def _to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        return (int(pos[1] / self.grid_resolution), int(pos[0] / self.grid_resolution))


class DisasterSimulator:
    """Fast cellular automata disaster spread model.

    Fire spread uses a modified Gaussian kernel. Flood and earthquake
    use simpler propagation rules. All tuned to run in <1ms per tick
    on a 100×60m grid at 0.5m resolution.
    """

    def __init__(self, width: float, height: float,
                 disaster_type: str,
                 origin: Tuple[float, float],
                 spread_rate: float,
                 resolution: float = 0.5):
        self.width = width
        self.height = height
        self.disaster_type = disaster_type
        self.origin = np.array(origin, dtype=np.float32)
        self.spread_rate = spread_rate
        self.resolution = resolution

        self.rows = int(height / resolution)
        self.cols = int(width / resolution)

        # Grid channels: smoke, temperature, structural, on_fire
        self.grid = np.zeros((self.rows, self.cols, 4), dtype=np.float32)
        self.grid[:, :, 1] = 25.0   # ambient temperature
        self.grid[:, :, 2] = 1.0    # structural integrity

        # Initialize disaster origin
        or_r, or_c = self._world_to_grid(self.origin)
        self._ignite_cell(or_r, or_c, intensity=1.0)

        # Pre-compute Gaussian kernel for fire spread
        self.kernel = self._make_kernel(sigma=1.5)

    def _world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        r = int(pos[1] / self.resolution)
        c = int(pos[0] / self.resolution)
        return (max(0, min(self.rows - 1, r)),
                max(0, min(self.cols - 1, c)))

    def _make_kernel(self, sigma: float) -> np.ndarray:
        size = int(sigma * 4) | 1  # odd size
        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def _ignite_cell(self, r: int, c: int, intensity: float = 1.0):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r, c, 3] = 1.0                      # on fire
            self.grid[r, c, 0] = min(1.0, intensity)      # smoke
            self.grid[r, c, 1] = 300.0 + intensity * 500  # temperature

    def step(self, dt: float):
        """Advance disaster one time step."""

        if self.disaster_type == "fire":
            self._step_fire(dt)
        elif self.disaster_type == "flood":
            self._step_flood(dt)
        elif self.disaster_type == "earthquake":
            self._step_earthquake(dt)

        # Decay structural integrity near fire
        fire_mask = self.grid[:, :, 3] > 0.5
        self.grid[fire_mask, 2] = np.maximum(
            0.0, self.grid[fire_mask, 2] - 0.002 * dt
        )

    def _step_fire(self, dt: float):
        """Fire spread: burning cells ignite neighbors + smoke diffuses."""
        # Find all burning cells
        fire_mask = self.grid[:, :, 3] > 0.5
        fire_rows, fire_cols = np.where(fire_mask)

        if len(fire_rows) == 0:
            return

        # Fire front advance: advance_per_tick = spread_rate * dt (meters)
        # A burning boundary cell has ~3 outward-facing neighbors
        # Each ignited neighbor advances the front by ~resolution meters
        # So: ignite_prob * 3 * resolution = spread_rate * dt
        # → ignite_prob = spread_rate * dt / (3 * resolution)
        outward_neighbors = 3.0
        ignite_prob = self.spread_rate * dt / (outward_neighbors * self.resolution)

        new_fire_mask = np.zeros_like(fire_mask)
        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        for r, c in zip(fire_rows, fire_cols):
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and not fire_mask[nr, nc]
                        and np.random.random() < ignite_prob):
                    new_fire_mask[nr, nc] = True

        self.grid[new_fire_mask, 3] = 1.0

        # Smoke: diffusion + production from fire
        smoke = self.grid[:, :, 0]
        # Simple diffusion (average with neighbors)
        smoke_padded = np.pad(smoke, 1, mode='edge')
        smoke_diffused = (
            smoke_padded[1:-1, 1:-1] * 0.6 +
            smoke_padded[2:, 1:-1] * 0.1 +
            smoke_padded[:-2, 1:-1] * 0.1 +
            smoke_padded[1:-1, 2:] * 0.1 +
            smoke_padded[1:-1, :-2] * 0.1
        )
        # Production from fire
        smoke_new = smoke_diffused + self.grid[:, :, 3] * 0.05 * dt
        self.grid[:, :, 0] = np.clip(smoke_new, 0.0, 1.0)

        # Temperature: rises near fire, slowly dissipates elsewhere
        fire_temp = 300.0 + self.grid[:, :, 3] * 500.0
        ambient = 25.0
        dissipation_rate = 0.01 * dt
        self.grid[:, :, 1] = np.where(
            self.grid[:, :, 3] > 0.5,
            fire_temp,
            self.grid[:, :, 1] * (1 - dissipation_rate) + ambient * dissipation_rate
        )

    def _convolve_2d(self, padded: np.ndarray, kernel: np.ndarray,
                     pad_h: int, pad_w: int) -> np.ndarray:
        """Manual 2D convolution (avoids scipy dependency)."""
        kh, kw = kernel.shape
        out_h = padded.shape[0] - kh + 1
        out_w = padded.shape[1] - kw + 1
        result = np.zeros((out_h, out_w), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                result += padded[i:i+out_h, j:j+out_w] * kernel[i, j]
        return result

    def _step_flood(self, dt: float):
        """Simplified flood: water level rises and spreads outward."""
        # Increase water level at origin, spread to lower neighboring cells
        water = self.grid[:, :, 0]  # repurpose smoke channel as water depth
        or_r, or_c = self._world_to_grid(self.origin)
        water[or_r, or_c] += 0.05 * dt * self.spread_rate

        # Spread to neighbors
        padded = np.pad(water, 1, mode='edge')
        water_new = padded[1:-1, 1:-1] + self.spread_rate * dt * 0.02 * (
            padded[2:, 1:-1] + padded[:-2, 1:-1] +
            padded[1:-1, 2:] + padded[1:-1, :-2] -
            4 * padded[1:-1, 1:-1]
        )
        self.grid[:, :, 0] = np.clip(water_new, 0, 5.0)

    def _step_earthquake(self, dt: float):
        """Earthquake: structural damage spreads from origin."""
        or_r, or_c = self._world_to_grid(self.origin)
        dist = np.sqrt(
            (np.arange(self.rows)[:, None] - or_r) ** 2 +
            (np.arange(self.cols)[None, :] - or_c) ** 2
        ) * self.resolution

        damage = np.exp(-dist / (self.spread_rate * 10)) * dt * 0.01
        self.grid[:, :, 2] = np.maximum(0.0, self.grid[:, :, 2] - damage)
        self.grid[:, :, 1] = 25.0  # temperature unaffected

    def snapshot(self, tick: int, timestamp: float,
                 exits: List[Tuple[float, float]],
                 obstacles: List[dict],
                 official_broadcast: str = "") -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            tick=tick,
            timestamp=timestamp,
            width=self.width,
            height=self.height,
            grid=self.grid.copy(),
            grid_resolution=self.resolution,
            exits=exits,
            obstacles=obstacles,
            official_broadcast=official_broadcast,
            disaster_type=self.disaster_type,
        )
