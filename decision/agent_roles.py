"""Multi-role agent definitions for layered command structure.

Extends the civilian evacuation model with:
  GlobalCommander — strategic oversight, area prioritization, public broadcasts
  AreaCommander   — zone-level guidance, bottleneck detection
  Firefighter     — fire suppression, trapped-person rescue
  Guide           — leads groups of civilians to assigned exits
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


class AgentRole(str, Enum):
    CIVILIAN = "civilian"
    GLOBAL_COMMANDER = "global_commander"
    AREA_COMMANDER = "area_commander"
    FIREFIGHTER = "firefighter"
    GUIDE = "guide"


# ------------------------------------------------------------------
# Role-specific profiles (extend the base AgentProfile)
# ------------------------------------------------------------------

@dataclass
class CommanderProfile:
    """Additional attributes for command-role agents."""
    authority_level: int = 3        # 1-5, affects civilian obedience probability
    comm_range: float = 999.0       # Global commander sees everything
    zone_id: int = 0                # Which zone this commander oversees
    zone_bounds: Tuple[float, float, float, float] = (0, 100, 0, 60)  # (xmin, xmax, ymin, ymax)


@dataclass
class FirefighterProfile:
    """Additional attributes for firefighter agents."""
    fire_resistance: float = 0.8    # 0-1, damage multiplier in fire (0=immune)
    smoke_resistance: float = 0.7   # 0-1, speed multiplier in smoke
    rescue_capacity: int = 2        # Max people that can be carried/escorted
    rescued_ids: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=lambda: ["呼吸器", "灭火器", "对讲机"])


@dataclass
class GuideProfile:
    """Additional attributes for guide agents."""
    assigned_exit_idx: int = 0      # Which exit this guide leads people to
    follower_capacity: int = 30     # Max civilians this guide can lead
    current_followers: List[str] = field(default_factory=list)
    is_active: bool = True          # Whether currently guiding


# ------------------------------------------------------------------
# Role-specific decision output types
# ------------------------------------------------------------------

@dataclass
class CommanderDecision:
    """Output from GlobalCommander / AreaCommander LLM call."""
    agent_id: str
    role: AgentRole

    # Strategic
    situation_assessment: str = ""
    area_priorities: List[dict] = field(default_factory=list)
    # [{zone_id: 0, priority: "high", reason: "火源位于此区域"}]

    # Communication (injected into civilian prompts)
    broadcast_message: str = ""

    # Resource allocation
    resource_allocations: List[dict] = field(default_factory=list)
    # [{zone_id: 1, action: "dispatch_firefighter", count: 2}]

    compute_time: float = 0.0


@dataclass
class FirefighterDecision:
    """Output from Firefighter LLM call."""
    agent_id: str
    role: AgentRole

    action: str = "move"            # move | suppress_fire | rescue | report
    target_position: Tuple[float, float] = (0.0, 0.0)
    rescue_target_ids: List[str] = field(default_factory=list)
    fire_suppression_point: Tuple[float, float] = (0.0, 0.0)
    speed: str = "run"
    reasoning: str = ""
    compute_time: float = 0.0


@dataclass
class GuideDecision:
    """Output from Guide LLM call."""
    agent_id: str
    role: AgentRole

    target_exit_idx: int = 0
    route_description: str = ""     # Natural language for civilians to follow
    speed: str = "walk"
    call_for_followers: bool = True  # Whether to shout for people to follow
    reasoning: str = ""
    compute_time: float = 0.0


# ------------------------------------------------------------------
# Zone definitions (for area commanders)
# ------------------------------------------------------------------

def define_zones(exits: List[Tuple[float, float]],
                 width: float, height: float) -> List[dict]:
    """Divide the environment into quadrant zones.

    Creates 4 spatial quadrants (NW, NE, SW, SE) based on the midpoint
    of the floorplan, with each zone associated with the nearest exits.
    """
    mid_x, mid_y = width / 2.0, height / 2.0
    quadrant_bounds = [
        (0.0, mid_x, mid_y, height, "西北区"),      # NW
        (mid_x, width, mid_y, height, "东北区"),     # NE
        (0.0, mid_x, 0.0, mid_y, "西南区"),          # SW
        (mid_x, width, 0.0, mid_y, "东南区"),        # SE
    ]
    zones = []
    for i, (x_min, x_max, y_min, y_max, label) in enumerate(quadrant_bounds):
        primary = []
        for e_idx, (ex, ey) in enumerate(exits):
            if x_min - 10 <= ex <= x_max + 10 and y_min - 10 <= ey <= y_max + 10:
                primary.append(e_idx)
        if not primary:
            primary = list(range(len(exits)))
        zones.append({
            "id": i,
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "primary_exits": primary,
            "label": label,
        })
    return zones
