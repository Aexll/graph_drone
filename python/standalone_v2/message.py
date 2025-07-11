"""
Classes de messages pour la communication entre drones
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class MessageType(Enum):
    """Types de messages échangés entre drones"""
    XI_UPDATE = "xi_update"
    OMEGA_UPDATE = "omega_update"
    CRITICAL_EDGE_CHECK = "critical_edge_check"
    NODE_COUNT_ESTIMATE = "node_count_estimate"


@dataclass
class Message:
    """Message échangé entre drones"""
    sender_id: int
    receiver_id: int
    msg_type: MessageType
    data: Dict[str, Any]
    timestamp: int = 0
