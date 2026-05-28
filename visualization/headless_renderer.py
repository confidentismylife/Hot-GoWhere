"""Headless renderer — saves simulation frames as images, no display needed.

Runs on headless servers (AutoDL, cloud GPU). Output: PNG frames → MP4/GIF.
"""

import os
import numpy as np
from typing import List
from decision.agent_state import Agent, Speed
from perception.environment import EnvironmentSnapshot


# Same color maps as the Pygame renderer
def _fear_color(fear_level: float) -> tuple:
    t = fear_level / 10.0
    if t < 0.5:
        return (t * 2, 1.0, 0.2)
    else:
        return (1.0, 1.0 - (t - 0.5) * 2, 0.2)


class HeadlessRenderer:
    """Save simulation frames to disk. No GPU/display required."""

    def __init__(self, output_dir: str = "./frames",
                 frame_interval: int = 10,  # Save every 10 ticks (1 frame/sec at dt=0.1)
                 dpi: int = 100):
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.dpi = dpi
        self.frame_count = 0
        self._imported = False

    def initialize(self, width: float, height: float):
        os.makedirs(self.output_dir, exist_ok=True)
        # Clear old frames
        for f in os.listdir(self.output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(self.output_dir, f))
        self.world_w = width
        self.world_h = height
        print(f"[Renderer] Saving frames to {self.output_dir}/ every {self.frame_interval} ticks")

    def render(self, agents: List[Agent], env: EnvironmentSnapshot,
               tick: int, sim_time: float,
               evacuated: int, casualties: int, decisions: int) -> bool:
        """Save a frame if it's time. Returns True to continue."""
        if tick % self.frame_interval != 0:
            return True

        # Lazy import matplotlib (slow first time)
        if not self._imported:
            import matplotlib
            matplotlib.use('Agg')  # No display needed
            import matplotlib.pyplot as plt
            self.plt = plt
            self._imported = True

        fig, ax = self.plt.subplots(figsize=(12, 7), dpi=self.dpi)
        ax.set_xlim(0, self.world_w)
        ax.set_ylim(0, self.world_h)
        ax.set_aspect('equal')
        ax.set_facecolor('#F0F0F5')

        # --- Smoke overlay ---
        step = max(1, env.grid.shape[1] // 60)
        for r in range(0, env.grid.shape[0], step):
            for c in range(0, env.grid.shape[1], step):
                smoke = env.grid[r, c, 0]
                if smoke < 0.05:
                    continue
                wx = c * env.grid_resolution
                wy = r * env.grid_resolution
                cell_w = env.grid_resolution * step
                rect = self.plt.Rectangle(
                    (wx, wy), cell_w, cell_w,
                    facecolor='gray', alpha=min(0.7, smoke * 0.8),
                    edgecolor='none'
                )
                ax.add_patch(rect)

        # --- Fire overlay ---
        fire_mask = env.grid[:, :, 3] > 0.5
        if fire_mask.any():
            rows, cols = np.where(fire_mask)
            for r, c in zip(rows[::step], cols[::step]):
                wx = c * env.grid_resolution
                wy = r * env.grid_resolution
                cell_w = env.grid_resolution * step
                rect = self.plt.Rectangle(
                    (wx, wy), cell_w, cell_w,
                    facecolor='#FF6414', alpha=0.6, edgecolor='none'
                )
                ax.add_patch(rect)

        # --- Obstacles ---
        for obs in env.obstacles:
            circle = self.plt.Circle(
                obs["center"], obs["radius"],
                facecolor='#B4B4B9', edgecolor='#8C8C91', linewidth=1.5
            )
            ax.add_patch(circle)

        # --- Exits ---
        for i, exit_pos in enumerate(env.exits):
            ex = exit_pos[0]
            ey = exit_pos[1]
            exit_smoke = env.smoke_at(np.array(exit_pos, dtype=np.float64))
            color = 'green' if exit_smoke < 0.3 else ('orange' if exit_smoke < 0.6 else 'red')
            rect = self.plt.Rectangle(
                (ex - 1.2, ey - 1.2), 2.4, 2.4,
                facecolor=color, alpha=0.7, edgecolor='darkgreen', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(ex + 1.5, ey, f'E{i+1}', fontsize=8, fontweight='bold')

        # --- Agents ---
        active = [a for a in agents if a.dynamic.alive and not a.dynamic.evacuated]
        if active:
            positions = np.array([a.position for a in active])
            colors = [_fear_color(a.dynamic.fear_level) for a in active]
            sizes = [max(8, min(20, a.profile.max_speed * 10)) for a in active]
            ax.scatter(positions[:, 0], positions[:, 1],
                      c=colors, s=sizes, alpha=0.85, edgecolors='white', linewidth=0.3)

        # --- HUD ---
        hud_lines = [
            f"Time: {sim_time:.0f}s | Tick: {tick}",
            f"Active: {len(active)} | Evacuated: {evacuated} | Casualties: {casualties}",
            f"LLM Decisions: {decisions}",
            f"Evac Rate: {evacuated/(evacuated+casualties+len(active))*100:.1f}%"
        ]
        hud_text = "\n".join(hud_lines)
        ax.text(0.02, 0.98, hud_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # --- Fear legend ---
        ax.text(0.02, 0.02, "Calm", transform=ax.transAxes,
                fontsize=7, color='green')
        ax.text(0.12, 0.02, "Nervous", transform=ax.transAxes,
                fontsize=7, color='#BBBB00')
        ax.text(0.25, 0.02, "Panicked", transform=ax.transAxes,
                fontsize=7, color='red')

        ax.set_xticks([])
        ax.set_yticks([])

        filename = os.path.join(self.output_dir, f"frame_{self.frame_count:05d}.png")
        fig.savefig(filename, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
        self.plt.close(fig)
        self.frame_count += 1
        return True

    def close(self):
        print(f"[Renderer] Saved {self.frame_count} frames to {self.output_dir}/")
        print(f"[Renderer] To create video on server: "
              f"ffmpeg -r 10 -i {self.output_dir}/frame_%05d.png -vcodec libx264 -pix_fmt yuv420p evacuation.mp4")
        print(f"[Renderer] To create GIF: "
              f"ffmpeg -r 10 -i {self.output_dir}/frame_%05d.png -vf 'fps=10,scale=800:-1' evacuation.gif")
