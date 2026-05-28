"""Group intelligence — information propagation, rumor dynamics, consensus.

Simulates how information spreads through a crowd during evacuation.
No longer depends on SpatialHashGrid; uses direct distance checks since
group intelligence runs once per tick and is not the bottleneck.
"""

import random
import numpy as np
from typing import List


class GroupIntelligence:

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.talk_range = 3.0
        self.shout_range = 15.0
        self.fear_contagion_range = 5.0
        self.fear_contagion_rate = 0.05

    # ================================================================
    # Information propagation
    # ================================================================

    def propagate(self, agents: List, official_broadcast: str, dt: float):
        """Run one tick of information propagation."""
        active = [a for a in agents if a.dynamic.alive and not a.dynamic.evacuated]
        n = len(active)
        if n < 2:
            return

        # Build position array for fast distance checks
        positions = np.array([a.position for a in active], dtype=np.float64)
        agent_by_idx = {i: a for i, a in enumerate(active)}

        # 1. Official broadcast
        if official_broadcast:
            for agent in active:
                if random.random() < 0.3:
                    self._add_to_memory(agent, f"官方广播: {official_broadcast}", 0.9)

        # 2. Peer-to-peer exchange (sample-based, not all pairs)
        for i, agent in enumerate(active):
            if random.random() > 0.1:  # 10% chance to talk per tick
                continue

            # Find nearby agents within talk_range
            diffs = positions - positions[i]
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            nearby = np.where((dists > 1e-6) & (dists <= self.talk_range))[0]

            if len(nearby) == 0:
                continue

            # Pick one random neighbor
            other_idx = nearby[random.randint(0, len(nearby) - 1)]
            other = agent_by_idx[other_idx]

            if other.dynamic.memory_events:
                info = random.choice(other.dynamic.memory_events[-5:])
                self._add_to_memory(
                    agent,
                    f"从{other.id}听说: {info.get('desc', '')}",
                    credibility=info.get('credibility', 0.5) * 0.8
                )

        # 3. Fear contagion
        for i, agent in enumerate(active):
            diffs = positions - positions[i]
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            nearby = np.where((dists > 1e-6) & (dists <= self.fear_contagion_range))[0]

            panicked = 0
            for j in nearby:
                if agent_by_idx[j].dynamic.fear_level > 7:
                    panicked += 1

            if panicked > 0:
                agent.dynamic.fear_level += panicked * self.fear_contagion_rate * dt

    # ================================================================
    # State updates
    # ================================================================

    def update_fear_levels(self, agents: List, env_snapshot, dt: float):
        for agent in agents:
            if agent.dynamic.evacuated or not agent.dynamic.alive:
                continue
            d = agent.dynamic
            pos = agent.position

            smoke = env_snapshot.smoke_at(pos)
            d.fear_level += smoke * 1.5 * dt
            if env_snapshot.is_on_fire(pos):
                d.fear_level += 3.0 * dt
            temp = env_snapshot.temperature_at(pos)
            if temp > 50:
                d.fear_level += (temp - 50) / 100 * dt
            if d.stamina < 30:
                d.fear_level += 1.0 * dt
            if smoke < 0.1 and temp < 40:
                d.fear_level -= 0.2 * dt

            d.fear_level = max(0.0, min(10.0, d.fear_level))

    def update_stamina(self, agents: List, dt: float):
        for agent in agents:
            if agent.dynamic.evacuated or not agent.dynamic.alive:
                continue
            d = agent.dynamic

            if d.speed_choice.value == "run":
                d.stamina -= 5.0 * dt
            elif d.speed_choice.value == "walk":
                d.stamina -= 0.5 * dt
            elif d.speed_choice.value == "wait":
                d.stamina += 2.0 * dt
            if agent.profile.age > 60:
                d.stamina -= 1.0 * dt
            if d.has_children:
                d.stamina -= 1.5 * dt

            d.stamina = max(0.0, min(100.0, d.stamina))
            if d.stamina <= 0 and random.random() < 0.01 * dt:
                d.alive = False

    # ================================================================
    # Internal helpers
    # ================================================================

    def _add_to_memory(self, agent, desc: str, credibility: float = 0.5):
        d = agent.dynamic
        d.memory_events.append({
            "time": "recent",
            "desc": desc,
            "credibility": credibility,
        })
        if len(d.memory_events) > 20:
            d.memory_events = d.memory_events[-20:]

        if credibility < 0.4:
            d.received_rumors.append({"content": desc, "credibility": credibility})
            if len(d.received_rumors) > 5:
                d.received_rumors = d.received_rumors[-5:]

        # Only trigger re-decision if this is genuinely new info
        # (broadcasts & peer exchanges are too frequent to re-decide every time)
        if credibility > 0.8:  # Only high-credibility info triggers re-decision
            d.has_new_info = True
