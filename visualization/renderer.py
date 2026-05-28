"""Pygame-based 2D visualization renderer.

Lightweight renderer for real-time simulation display:
- Agents as colored circles (color = fear level)
- Disaster overlay (smoke/heat as semi-transparent grid)
- Exits, obstacles, and agent trails
- HUD with stats

Runs at 30 FPS, independent of simulation speed.
"""

import math
import numpy as np
from typing import List
from decision.agent_state import Agent, Speed
from perception.environment import EnvironmentSnapshot


# Color maps
def fear_color(fear_level: float) -> tuple:
    """Green (calm) → Yellow (nervous) → Red (panicked)."""
    t = fear_level / 10.0
    if t < 0.5:
        # Green → Yellow
        r = int(255 * t * 2)
        g = 255
        b = 50
    else:
        # Yellow → Red
        r = 255
        g = int(255 * (1 - (t - 0.5) * 2))
        b = 50
    return (r, g, b)


def smoke_color(density: float) -> tuple:
    """White → Gray → Dark gray based on smoke density."""
    alpha = int(128 * density)
    return (alpha, alpha, alpha, alpha)


def exit_color(status: str) -> tuple:
    if status == "通畅":
        return (0, 200, 0)
    elif status == "有烟雾":
        return (200, 200, 0)
    else:
        return (200, 50, 50)


class PygameRenderer:
    def __init__(self, config: dict, world_w: float, world_h: float):
        self.world_w = world_w
        self.world_h = world_h
        self.screen_w = config.get("width", 1200)
        self.screen_h = config.get("height", 720)
        self.target_fps = config.get("fps", 30)

        # Scale: world coords → screen pixels
        self.scale_x = self.screen_w / world_w
        self.scale_y = self.screen_h / world_h

        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        # Trail tracking (agent ID → last N positions)
        self.trails = {}
        self._running = True

    def initialize(self):
        import pygame
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("LLM-Powered Crowd Evacuation Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 18)

    def world_to_screen(self, wx: float, wy: float) -> tuple:
        return (int(wx * self.scale_x), int(wy * self.scale_y))

    def render(self, agents: List[Agent], env: EnvironmentSnapshot,
               tick: int, sim_time: float,
               evacuated: int, casualties: int,
               decisions: int) -> bool:
        """Render one frame. Returns False if user closed the window."""
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self._running = False
                    return False

        # Background
        self.screen.fill((240, 240, 245))

        # --- Disaster overlay (smoke grid) ---
        self._draw_smoke_overlay(env)

        # --- Obstacles ---
        for obs in env.obstacles:
            cx, cy = obs["center"]
            r = obs["radius"]
            sx, sy = self.world_to_screen(cx, cy)
            sr = int(r * self.scale_x)
            pygame.draw.circle(self.screen, (180, 180, 185), (sx, sy), sr)
            pygame.draw.circle(self.screen, (140, 140, 145), (sx, sy), sr, 2)

        # --- Exits ---
        for i, exit_pos in enumerate(env.exits):
            sx, sy = self.world_to_screen(*exit_pos)
            exit_smoke = env.smoke_at(np.array(exit_pos, dtype=np.float64))
            if exit_smoke < 0.3:
                color = (0, 180, 0)
            elif exit_smoke < 0.6:
                color = (200, 180, 0)
            else:
                color = (200, 50, 50)

            # Draw exit marker
            pygame.draw.rect(self.screen, color,
                           (sx - 10, sy - 10, 20, 20), border_radius=3)
            pygame.draw.rect(self.screen, (0, 100, 0),
                           (sx - 10, sy - 10, 20, 20), 2, border_radius=3)

            # Label
            label = self.small_font.render(f"E{i+1}", True, (0, 0, 0))
            self.screen.blit(label, (sx + 12, sy - 5))

        # --- Fire visualization ---
        self._draw_fire_overlay(env)

        # --- Agents ---
        active_agents = [a for a in agents if a.dynamic.alive and not a.dynamic.evacuated]
        for agent in active_agents:
            self._draw_agent(agent)

        # --- HUD ---
        self._draw_hud(tick, sim_time, len(active_agents),
                       evacuated, casualties, decisions)

        pygame.display.flip()
        self.clock.tick(self.target_fps)
        return True

    def _draw_smoke_overlay(self, env: EnvironmentSnapshot):
        """Draw smoke as a coarse heat map."""
        # Downsample the grid for performance — only draw every Nth cell
        step = max(1, int(env.grid.shape[1] / 80))  # ~80 cells across
        cell_px = self.scale_x * env.grid_resolution * step

        if cell_px < 2:  # Too small to draw individually
            return

        for r in range(0, env.grid.shape[0], step):
            for c in range(0, env.grid.shape[1], step):
                smoke = env.grid[r, c, 0]
                if smoke < 0.05:
                    continue

                wx = c * env.grid_resolution
                wy = r * env.grid_resolution
                sx, sy = self.world_to_screen(wx, wy)

                alpha = min(180, int(smoke * 200))
                smoke_surf = self.pygame.Surface((int(cell_px) + 1, int(cell_px) + 1))
                smoke_surf.set_alpha(alpha)
                smoke_surf.fill((80, 80, 80))
                self.screen.blit(smoke_surf, (sx, sy))

    def _draw_fire_overlay(self, env: EnvironmentSnapshot):
        """Draw fire as orange/red cells."""
        fire_mask = env.grid[:, :, 3] > 0.5
        if not fire_mask.any():
            return

        step = max(1, int(env.grid.shape[1] / 80))
        cell_px = self.scale_x * env.grid_resolution * step

        if cell_px < 2:
            return

        rows, cols = np.where(fire_mask)
        for r, c in zip(rows[::step], cols[::step]):
            wx = c * env.grid_resolution
            wy = r * env.grid_resolution
            sx, sy = self.world_to_screen(wx, wy)

            fire_surf = self.pygame.Surface((int(cell_px) + 1, int(cell_px) + 1))
            fire_surf.set_alpha(150)
            fire_surf.fill((255, 100, 20))
            self.screen.blit(fire_surf, (sx, sy))

    def _draw_agent(self, agent: Agent):
        sx, sy = self.world_to_screen(agent.position[0], agent.position[1])
        radius = max(3, min(8, int(5 * self.scale_x / 10)))

        # Color by fear level
        color = fear_color(agent.dynamic.fear_level)

        # Outline for special states
        outline_color = None
        outline_width = 0
        if agent.dynamic.cooperation_choice.value == "lead_others":
            outline_color = (0, 100, 255)
            outline_width = 2
        elif agent.dynamic.has_children:
            outline_color = (255, 200, 0)
            outline_width = 1

        if outline_color:
            self.pygame.draw.circle(self.screen, outline_color,
                                   (sx, sy), radius + outline_width)

        self.pygame.draw.circle(self.screen, color, (sx, sy), radius)

        # Speed indicator
        if agent.dynamic.speed_choice == Speed.RUN:
            # Small arrow in movement direction
            end_x = sx + agent.dynamic.velocity[0] * self.scale_x * 2
            end_y = sy + agent.dynamic.velocity[1] * self.scale_y * 2
            if abs(end_x - sx) + abs(end_y - sy) > 2:
                self.pygame.draw.line(self.screen, (255, 255, 255),
                                    (sx, sy), (end_x, end_y), 1)

    def _draw_hud(self, tick: int, sim_time: float,
                  active: int, evacuated: int, casualties: int, decisions: int):
        """Draw on-screen stats overlay."""
        lines = [
            f"Time: {sim_time:.1f}s  Tick: {tick}",
            f"Active: {active}  Evacuated: {evacuated}  Casualties: {casualties}",
            f"LLM Decisions: {decisions}",
            "[Q/ESC] Quit",
        ]

        # Semi-transparent background
        hud_surf = self.pygame.Surface((350, 100))
        hud_surf.set_alpha(200)
        hud_surf.fill((20, 20, 30))
        self.screen.blit(hud_surf, (10, 10))

        for i, line in enumerate(lines):
            text = self.small_font.render(line, True, (220, 220, 230))
            self.screen.blit(text, (20, 15 + i * 22))

        # Color legend
        legend_y = self.screen_h - 40
        for i, (label, level) in enumerate([("Calm", 0), ("Nervous", 5), ("Panic", 10)]):
            color = fear_color(level)
            self.pygame.draw.circle(self.screen, color,
                                  (20 + i * 120, legend_y), 6)
            text = self.small_font.render(label, True, (60, 60, 60))
            self.screen.blit(text, (32 + i * 120, legend_y - 8))

    def close(self):
        if self.screen:
            self.pygame.quit()
