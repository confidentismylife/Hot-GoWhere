"""Agent state data model — the full cognitive profile for each simulated person."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict
import uuid
import numpy as np


class Speed(str, Enum):
    RUN = "run"
    WALK = "walk"
    CRAWL = "crawl"
    WAIT = "wait"


class Cooperation(str, Enum):
    NONE = "none"
    HELP_FAMILY = "help_family"
    FOLLOW_CROWD = "follow_crowd"
    LEAD_OTHERS = "lead_others"


@dataclass
class AgentProfile:
    """Static identity — set at spawn, immutable during simulation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    age: int = 30
    gender: str = "male"
    occupation: str = "office_worker"

    # 0 = first time here, 1 = knows every corner
    familiarity: float = 0.5

    # m/s, individual max speed (varies by age/fitness)
    max_speed: float = 1.5

    # Personality traits (0-1)
    risk_aversion: float = 0.5       # 1 = extremely cautious
    altruism: float = 0.3            # 1 = selfless helper
    trust_authority: float = 0.7     # 1 = fully trusts official info
    conformity: float = 0.5          # 1 = follows the crowd


@dataclass
class AgentDynamic:
    """Mutable state — changes every tick."""

    # --- Physical ---
    position: np.ndarray            # (x, y)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    stamina: float = 100.0          # 0-100, drains when running
    injured: bool = False
    alive: bool = True
    evacuated: bool = False

    # --- Psychological ---
    fear_level: float = 0.0         # 0-10
    trust_official_now: float = 0.7 # Dynamic trust (may change during event)

    # --- Social ---
    family_member_ids: List[str] = field(default_factory=list)
    has_children: bool = False
    has_elderly: bool = False

    # --- Knowledge ---
    known_exit_positions: List[Tuple[float, float]] = field(default_factory=list)
    memory_events: List[Dict] = field(default_factory=list)   # capped at 20
    received_rumors: List[Dict] = field(default_factory=list)

    # --- Current Decision ---
    target_exit: Optional[np.ndarray] = None
    speed_choice: Speed = Speed.WALK
    cooperation_choice: Cooperation = Cooperation.NONE
    reasoning_text: str = ""

    # --- Decision bookkeeping ---
    last_decision_tick: int = -999
    has_new_info: bool = True       # Forces re-decision


@dataclass
class Agent:
    """Top-level agent combining profile + dynamic state."""
    profile: AgentProfile = field(default_factory=AgentProfile)
    dynamic: AgentDynamic = field(default_factory=AgentDynamic)

    @property
    def id(self) -> str:
        return self.profile.id

    @property
    def position(self) -> np.ndarray:
        return self.dynamic.position

    @property
    def needs_decision(self) -> bool:
        return self.dynamic.has_new_info

    def to_context_dict(self) -> dict:
        """Serialize agent state for LLM prompt rendering."""
        d = self.dynamic
        p = self.profile
        return {
            "id": p.id,
            "age": p.age,
            "occupation": p.occupation,
            "familiarity": f"{'熟悉' if p.familiarity > 0.6 else '不太熟悉'}",
            "stamina": int(d.stamina),
            "fear": f"{d.fear_level:.1f}/10",
            "trust_official": "信任" if d.trust_official_now > 0.5 else "不太信任",
            "has_children": d.has_children,
            "has_elderly": d.has_elderly,
            "position": f"({d.position[0]:.1f}, {d.position[1]:.1f})",
            "speed": d.speed_choice.value,
            "family_nearby": len([fid for fid in d.family_member_ids]) > 0,
        }
