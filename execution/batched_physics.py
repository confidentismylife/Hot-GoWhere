"""Batched Social Force Model — all agents in one JIT kernel.

Eliminates the per-agent Python/JIT dispatch overhead that plagues the
original per-agent loop. 1000 agents in ~3ms instead of ~525ms.

Key insight: Numba JIT dispatch costs ~50µs per call. With 5 force
functions × 1000 agents = 5000 calls/tick, that's 250ms of pure overhead.
One single JIT call that processes everything internally fixes this.
"""

import numpy as np
from numba import jit
from typing import List

from decision.agent_state import Agent, Speed, Cooperation


# ================================================================
# Single-kernel batched physics (ALL agents in one JIT call)
# ================================================================

@jit(nopython=True, cache=True)
def _step_all(
    # Agent arrays (N = active agent count)
    pos_x, pos_y,           # float64[N] — current positions
    vel_x, vel_y,           # float64[N] — current velocities
    target_x, target_y,     # float64[N] — target exit positions
    desired_speeds,         # float64[N]
    max_speeds,             # float64[N]
    stamina,                # float64[N]
    fear,                   # float64[N]
    coop_mode,              # int32[N] — 0=none, 1=help_family
    family_ids,             # int32[N×4] — -1 = no family member, padded
    # Obstacles
    obs_x, obs_y,           # float64[K]
    obs_radius,             # float64[K]
    # Environment
    width, height,          # float64
    tau,                    # float64
    dt,                     # float64
    # Spatial grid params
    cell_size,              # float64
):
    """
    Process ALL agents in one JIT-compiled kernel.

    Returns updated positions, velocities, and evacuated flags.
    """
    N = pos_x.shape[0]
    K = obs_x.shape[0]

    # --- Phase 1: Build spatial hash grid (array-based, no dicts) ---
    cols = int(width / cell_size) + 1
    rows = int(height / cell_size) + 1

    # cell_offsets[cell_idx] = start index in cell_agents
    n_cells = rows * cols
    cell_counts = np.zeros(n_cells, dtype=np.int32)

    # Count agents per cell (first pass)
    for i in range(N):
        cx = int(pos_x[i] // cell_size)
        cy = int(pos_y[i] // cell_size)
        if cx < 0: cx = 0
        if cx >= cols: cx = cols - 1
        if cy < 0: cy = 0
        if cy >= rows: cy = rows - 1
        cell_idx = cy * cols + cx
        cell_counts[cell_idx] += 1

    # Compute cell offsets (prefix sum)
    cell_offsets = np.zeros(n_cells + 1, dtype=np.int32)
    for i in range(n_cells):
        cell_offsets[i + 1] = cell_offsets[i] + cell_counts[i]

    # Reset counts for second pass
    cell_counts[:] = 0

    # Fill agent indices into cells (second pass)
    cell_agents = np.zeros(N, dtype=np.int32)
    agent_cells = np.zeros(N, dtype=np.int32)
    for i in range(N):
        cx = int(pos_x[i] // cell_size)
        cy = int(pos_y[i] // cell_size)
        if cx < 0: cx = 0
        if cx >= cols: cx = cols - 1
        if cy < 0: cy = 0
        if cy >= rows: cy = rows - 1
        cell_idx = cy * cols + cx
        offset = cell_offsets[cell_idx] + cell_counts[cell_idx]
        cell_agents[offset] = i
        cell_counts[cell_idx] += 1
        agent_cells[i] = cell_idx

    # --- Phase 2: Force computation for all agents ---
    new_vel_x = np.empty(N, dtype=np.float64)
    new_vel_y = np.empty(N, dtype=np.float64)
    new_pos_x = np.empty(N, dtype=np.float64)
    new_pos_y = np.empty(N, dtype=np.float64)
    evacuated = np.zeros(N, dtype=np.int32)

    # Neighbor search directions (9 cells including self)
    nbr_dc = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    nbr_dr = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)

    for i in range(N):
        px = pos_x[i]
        py = pos_y[i]
        vx = vel_x[i]
        vy = vel_y[i]
        tx = target_x[i]
        ty = target_y[i]
        ds = desired_speeds[i]

        # --- Find neighbors in adjacent cells ---
        cell = agent_cells[i]
        cy = cell // cols
        cx = cell % cols

        nbr_count = 0
        nbr_px = np.empty(50, dtype=np.float64)  # Max 50 neighbors
        nbr_py = np.empty(50, dtype=np.float64)
        nbr_vx = np.empty(50, dtype=np.float64)
        nbr_vy = np.empty(50, dtype=np.float64)

        for j in range(9):
            nc = cx + nbr_dc[j]
            nr = cy + nbr_dr[j]
            if nc < 0 or nc >= cols or nr < 0 or nr >= rows:
                continue
            n_cell = nr * cols + nc
            start = cell_offsets[n_cell]
            end = cell_offsets[n_cell + 1]
            for k in range(start, end):
                other = cell_agents[k]
                if other == i:
                    continue
                ox = pos_x[other]
                oy = pos_y[other]
                dx = px - ox
                dy = py - oy
                d2 = dx * dx + dy * dy
                if d2 < 25.0 and d2 > 1e-8:  # Within 5m
                    if nbr_count < 50:
                        nbr_px[nbr_count] = ox
                        nbr_py[nbr_count] = oy
                        nbr_vx[nbr_count] = vel_x[other]
                        nbr_vy[nbr_count] = vel_y[other]
                        nbr_count += 1

        # --- Force 1: Desired force ---
        dx_t = tx - px
        dy_t = ty - py
        dist_t = np.sqrt(dx_t * dx_t + dy_t * dy_t)
        if dist_t > 1e-6:
            ex = dx_t / dist_t
            ey = dy_t / dist_t
            F_dx = (ds * ex - vx) / tau
            F_dy = (ds * ey - vy) / tau
        else:
            F_dx = 0.0
            F_dy = 0.0

        # --- Force 2: Social repulsion ---
        F_sx = 0.0
        F_sy = 0.0
        v_norm = np.sqrt(vx * vx + vy * vy + 1e-6)
        for j in range(nbr_count):
            dx = px - nbr_px[j]
            dy = py - nbr_py[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                continue

            nx = dx / dist
            ny = dy / dist
            cos_phi = (nx * vx + ny * vy) / v_norm
            anisotropy = 0.2 + 0.8 * (1.0 + cos_phi) / 2.0
            f_mag = 2000.0 * np.exp(-dist / 0.08) * anisotropy
            F_sx += f_mag * nx
            F_sy += f_mag * ny

            if dist < 0.6:
                F_sx += 12000.0 * (0.6 - dist) * nx
                F_sy += 12000.0 * (0.6 - dist) * ny

        # --- Force 3: Obstacle repulsion ---
        F_ox = 0.0
        F_oy = 0.0
        for j in range(K):
            dx = px - obs_x[j]
            dy = py - obs_y[j]
            raw_dist = np.sqrt(dx * dx + dy * dy)
            dist = raw_dist - obs_radius[j]
            if dist < 1e-6 or dist > 3.0:
                continue
            nx = dx / raw_dist
            ny = dy / raw_dist
            f = 10000.0 * np.exp(-dist / 0.02)
            F_ox += f * nx
            F_oy += f * ny

        # --- Force 4: Boundary ---
        F_bx = 0.0
        F_by = 0.0
        margin = 1.0
        if px < margin:
            F_bx += 50000.0 * (margin - px)
        elif px > width - margin:
            F_bx -= 50000.0 * (px - (width - margin))
        if py < margin:
            F_by += 50000.0 * (margin - py)
        elif py > height - margin:
            F_by -= 50000.0 * (py - (height - margin))

        # --- Force 5: Family attraction ---
        F_fx = 0.0
        F_fy = 0.0
        if coop_mode[i] == 1:
            fam_count = 0
            for j in range(family_ids.shape[1]):
                fid = family_ids[i, j]
                if fid < 0:
                    continue
                fpx = pos_x[fid]
                fpy = pos_y[fid]
                dx = fpx - px
                dy = fpy - py
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > 1e-6 and dist < 20.0:
                    f_mag = 300.0 * min(dist / 10.0, 2.0)
                    F_fx += f_mag * dx / dist
                    F_fy += f_mag * dy / dist
                    fam_count += 1
            if fam_count > 0:
                F_fx /= fam_count
                F_fy /= fam_count

        # --- Combine forces ---
        F_tx = 1.0 * F_dx + 0.6 * F_sx + 1.2 * F_ox + 0.3 * F_fx + 0.1 * F_bx
        F_ty = 1.0 * F_dy + 0.6 * F_sy + 1.2 * F_oy + 0.3 * F_fy + 0.1 * F_by

        # --- Semi-implicit Euler ---
        nvx = vx + F_tx * dt
        nvy = vy + F_ty * dt
        spd = np.sqrt(nvx * nvx + nvy * nvy)
        max_spd = max_speeds[i] * 1.2
        if spd > max_spd:
            nvx = nvx / spd * max_spd
            nvy = nvy / spd * max_spd

        npx = px + nvx * dt
        npy = py + nvy * dt

        # --- Exit check ---
        dx_exit = tx - npx
        dy_exit = ty - npy
        if np.sqrt(dx_exit * dx_exit + dy_exit * dy_exit) < 1.5:
            evacuated[i] = 1
            npx = tx
            npy = ty
            nvx = 0.0
            nvy = 0.0

        new_pos_x[i] = npx
        new_pos_y[i] = npy
        new_vel_x[i] = nvx
        new_vel_y[i] = nvy

    return new_pos_x, new_pos_y, new_vel_x, new_vel_y, evacuated


class BatchedPhysics:
    """Wrapper that extracts agent data into flat arrays, calls the JIT
    kernel, and writes results back to agent objects."""

    def __init__(self, width: float, height: float,
                 obstacles: List[dict], tau: float = 0.5):
        self.width = width
        self.height = height

        if obstacles:
            self.obs_x = np.array([o["center"][0] for o in obstacles], dtype=np.float64)
            self.obs_y = np.array([o["center"][1] for o in obstacles], dtype=np.float64)
            self.obs_r = np.array([o["radius"] for o in obstacles], dtype=np.float64)
        else:
            self.obs_x = np.zeros(0, dtype=np.float64)
            self.obs_y = np.zeros(0, dtype=np.float64)
            self.obs_r = np.zeros(0, dtype=np.float64)

        self.tau = tau
        self.cell_size = 5.0

        # Speed lookup table
        self._speed_map = {
            Speed.RUN: 0.85, Speed.WALK: 0.45,
            Speed.CRAWL: 0.15, Speed.WAIT: 0.0,
        }

    def step_all(self, agents: List[Agent], dt: float):
        """Process all active agents in one batched JIT call."""

        # Only process alive, non-evacuated agents
        active = [a for a in agents if a.dynamic.alive and not a.dynamic.evacuated]
        if not active:
            return

        N = len(active)

        # Extract data into flat arrays
        pos_x = np.empty(N, dtype=np.float64)
        pos_y = np.empty(N, dtype=np.float64)
        vel_x = np.empty(N, dtype=np.float64)
        vel_y = np.empty(N, dtype=np.float64)
        tgt_x = np.empty(N, dtype=np.float64)
        tgt_y = np.empty(N, dtype=np.float64)
        desired_speeds = np.empty(N, dtype=np.float64)
        max_speeds = np.empty(N, dtype=np.float64)
        stamina_arr = np.empty(N, dtype=np.float64)
        fear_arr = np.empty(N, dtype=np.float64)
        coop_arr = np.empty(N, dtype=np.int32)

        # Family mapping: find max family size, build padded array
        max_fam = 0
        agent_indices = {}  # agent.id → array index
        for i, a in enumerate(active):
            agent_indices[a.id] = i
            max_fam = max(max_fam, len(a.dynamic.family_member_ids))
        max_fam = max(1, max_fam)

        family_ids = np.full((N, max_fam), -1, dtype=np.int32)

        for i, a in enumerate(active):
            d = a.dynamic
            pos_x[i] = d.position[0]
            pos_y[i] = d.position[1]
            vel_x[i] = d.velocity[0]
            vel_y[i] = d.velocity[1]

            if d.target_exit is not None:
                tgt_x[i] = d.target_exit[0]
                tgt_y[i] = d.target_exit[1]
            else:
                # No decision yet — set target impossibly far so exit check won't fire
                tgt_x[i] = -9999.0
                tgt_y[i] = -9999.0

            max_speeds[i] = a.profile.max_speed
            stamina_arr[i] = d.stamina
            fear_arr[i] = d.fear_level
            coop_arr[i] = 1 if d.cooperation_choice.value == "help_family" else 0

            # Desired speed from decision + stamina + fear
            if d.target_exit is not None:
                base_speed = self._speed_map.get(d.speed_choice, 0.45)
                ds = a.profile.max_speed * base_speed
                ds *= max(0.2, d.stamina / 100.0)
                if d.fear_level > 7 and d.stamina > 50:
                    ds *= 1.3
            else:
                ds = 0.0  # No decision yet, stand still
            desired_speeds[i] = ds

            # Family member indices
            for j, fid in enumerate(d.family_member_ids):
                if fid in agent_indices:
                    family_ids[i, j] = agent_indices[fid]

        # Single JIT call — all agents at once
        new_px, new_py, new_vx, new_vy, evacuated = _step_all(
            pos_x, pos_y, vel_x, vel_y,
            tgt_x, tgt_y,
            desired_speeds, max_speeds,
            stamina_arr, fear_arr,
            coop_arr, family_ids,
            self.obs_x, self.obs_y, self.obs_r,
            self.width, self.height,
            self.tau, dt,
            self.cell_size,
        )

        # Write results back to agent objects
        for i, a in enumerate(active):
            d = a.dynamic
            d.position = np.array([new_px[i], new_py[i]], dtype=np.float64)
            d.velocity = np.array([new_vx[i], new_vy[i]], dtype=np.float64)
            if evacuated[i]:
                d.evacuated = True
                d.position = d.target_exit.copy() if d.target_exit is not None else d.position
                d.velocity = np.zeros(2, dtype=np.float64)
