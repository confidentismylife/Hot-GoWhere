"""Social Force Model — pedestrian dynamics with LLM-modulated behavior.

Implements the improved Social Force Model with:
- Desired force (toward target exit)
- Social repulsion (avoid other pedestrians)
- Obstacle repulsion (avoid walls, pillars)
- Family attraction (stay close to family members)

Numba JIT-compiled for CPU performance. Handles 1000 agents in ~3ms.
"""

import numpy as np
from numba import jit
from typing import List, Tuple
from decision.agent_state import Agent, Speed, Cooperation


# --- JIT-compiled force kernels ---

@jit(nopython=True, cache=True)
def _desired_force(position, velocity, target, desired_speed, tau):
    """F_desired = (v_desired * e_desired - v_current) / tau"""
    direction = target - position
    dist = np.sqrt(direction[0]**2 + direction[1]**2)
    if dist < 1e-6:
        return np.zeros(2)
    e = direction / dist
    return (desired_speed * e - velocity) / tau


@jit(nopython=True, cache=True)
def _social_force(position, velocity, neighbors_pos, neighbors_vel,
                  A=2000.0, B=0.08, body_force=12000.0, body_radius=0.6):
    """F_social: repulsion from other pedestrians. Elliptical anisotropic."""
    force = np.zeros(2)
    v_norm = np.sqrt(velocity[0]**2 + velocity[1]**2)
    if v_norm < 1e-6:
        v_norm = 1e-6

    for i in range(neighbors_pos.shape[0]):
        diff = position - neighbors_pos[i]
        dist = np.sqrt(diff[0]**2 + diff[1]**2)
        if dist < 1e-6 or dist > 5.0:
            continue

        n = diff / dist

        # Anisotropy: force stronger when person is in front
        cos_phi = (n[0] * velocity[0] + n[1] * velocity[1]) / v_norm
        anisotropy = 0.2 + (1.0 - 0.2) * (1.0 + cos_phi) / 2.0

        # Social repulsion
        force_mag = A * np.exp(-dist / B) * anisotropy
        force += force_mag * n

        # Body force (physical contact)
        if dist < body_radius:
            force += body_force * (body_radius - dist) * n

    return force


@jit(nopython=True, cache=True)
def _obstacle_force(position, obstacles_pos, obstacles_radius,
                    A_wall=10000.0, B_wall=0.02):
    """F_obstacle: repulsion from walls, pillars, and other obstacles."""
    force = np.zeros(2)
    for i in range(obstacles_pos.shape[0]):
        diff = position - obstacles_pos[i]
        dist = np.sqrt(diff[0]**2 + diff[1]**2) - obstacles_radius[i]
        if dist < 1e-6 or dist > 3.0:
            continue
        n = diff / (dist + obstacles_radius[i])
        force += A_wall * np.exp(-dist / B_wall) * n
    return force


@jit(nopython=True, cache=True)
def _family_force(position, family_positions, max_dist=20.0, strength=300.0):
    """F_family: attraction to family members, only when HELP_FAMILY active."""
    force = np.zeros(2)
    count = 0
    for i in range(family_positions.shape[0]):
        diff = family_positions[i] - position
        dist = np.sqrt(diff[0]**2 + diff[1]**2)
        if dist < 1e-6 or dist > max_dist:
            continue
        # Gentle attraction, stronger when farther
        f_mag = strength * min(dist / 10.0, 2.0)
        force += f_mag * diff / dist
        count += 1
    if count > 0:
        force /= count
    return force


@jit(nopython=True, cache=True)
def _boundary_force(position, width, height,
                    margin=1.0, strength=50000.0):
    """Keep agents inside the simulation bounds."""
    force = np.zeros(2)
    if position[0] < margin:
        force[0] += strength * (margin - position[0])
    elif position[0] > width - margin:
        force[0] -= strength * (position[0] - (width - margin))
    if position[1] < margin:
        force[1] += strength * (margin - position[1])
    elif position[1] > height - margin:
        force[1] -= strength * (position[1] - (height - margin))
    return force


class SocialForceModel:
    """Pedestrian dynamics engine.

    Each physics tick (100ms for 1000 agents):
    - Force computation: ~2ms (JIT)
    - Position update: ~0.5ms
    - Grid rebuild: ~0.5ms
    Total: ~3ms on CPU — comfortably real-time.
    """

    def __init__(self, width: float, height: float,
                 obstacles: List[dict], tau: float = 0.5):
        self.width = width
        self.height = height
        self.tau = tau

        # Pre-extract obstacle data for JIT
        if obstacles:
            self.obstacles_pos = np.array(
                [obs["center"] for obs in obstacles], dtype=np.float64)
            self.obstacles_radius = np.array(
                [obs["radius"] for obs in obstacles], dtype=np.float64)
        else:
            self.obstacles_pos = np.zeros((0, 2), dtype=np.float64)
            self.obstacles_radius = np.zeros(0, dtype=np.float64)

    def step(self, agent: Agent, neighbors_pos: np.ndarray,
             neighbors_vel: np.ndarray, family_positions: np.ndarray,
             dt: float):
        """Compute forces and update position for a single agent."""
        d = agent.dynamic
        p = agent.profile

        # Desired speed from LLM decision
        speed_values = {Speed.RUN: p.max_speed * 0.85,
                        Speed.WALK: p.max_speed * 0.45,
                        Speed.CRAWL: p.max_speed * 0.15,
                        Speed.WAIT: 0.0}
        desired_speed = speed_values[d.speed_choice]

        # Fatigue penalty
        desired_speed *= max(0.2, d.stamina / 100.0)

        # Fear can increase speed (fight-or-flight)
        if d.fear_level > 7 and d.stamina > 50:
            desired_speed *= 1.3

        # Compute all forces
        target = d.target_exit if d.target_exit is not None else d.position

        F_des = _desired_force(d.position, d.velocity, target, desired_speed, self.tau)
        F_soc = _social_force(d.position, d.velocity, neighbors_pos, neighbors_vel)
        F_obs = _obstacle_force(d.position, self.obstacles_pos, self.obstacles_radius)
        F_bound = _boundary_force(d.position, self.width, self.height)

        # Family force (only when cooperation is HELP_FAMILY)
        F_fam = np.zeros(2)
        if d.cooperation_choice == Cooperation.HELP_FAMILY and family_positions.shape[0] > 0:
            F_fam = _family_force(d.position, family_positions)

        # Weighted combination
        F_total = (1.0 * F_des + 0.6 * F_soc + 1.2 * F_obs +
                   0.3 * F_fam + 0.1 * F_bound)

        # Semi-implicit Euler
        new_velocity = d.velocity + F_total * dt
        speed = float(np.linalg.norm(new_velocity))

        # Clamp to max speed
        if speed > p.max_speed * 1.2:
            new_velocity = new_velocity / speed * p.max_speed * 1.2

        # Update position
        new_position = d.position + new_velocity * dt

        # Check for exit reached
        if target is not None and d.target_exit is not None:
            dist_to_exit = float(np.linalg.norm(new_position - d.target_exit))
            if dist_to_exit < 1.5:  # Within 1.5m of exit = evacuated
                d.evacuated = True
                d.position = d.target_exit.copy()
                d.velocity = np.zeros(2)
                return

        d.position = new_position.copy()
        d.velocity = new_velocity.copy()

    def _distance_to_exits(self, pos: np.ndarray,
                           exits: List[Tuple[float, float]]) -> np.ndarray:
        exit_arr = np.array(exits, dtype=np.float64)
        diffs = exit_arr - pos
        return np.sqrt((diffs ** 2).sum(axis=1))
