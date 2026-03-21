"""
CircleManager: Managing networks of fabric relationships.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict

class CircleRole(Enum):
    OWNER   = 'owner'
    ADMIN   = 'admin'
    MEMBER  = 'member'
    GUEST   = 'guest'

@dataclass
class Circle:
    circle_id:    str
    name:         str
    circle_type:  str
    owner_id:     str
    members:      Dict[str, CircleRole]

class CircleManager:
    def create_circle(self, owner_fabric, name, circle_type):
        return Circle(f"c_{name}", name, circle_type, owner_fabric.fabric_id, {owner_fabric.fabric_id: CircleRole.OWNER})
