"""
FabricHandshake: The protocol for fabrics communicating.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Set, Optional, Dict

class HandshakeType(Enum):
    PERSONAL     = 'personal'
    FAMILY       = 'family'
    TEAM         = 'team'
    ORGANIZATION = 'organization'
    TEMPORARY    = 'temporary'

class HandshakeStatus(Enum):
    PENDING      = 'pending'
    ACTIVE       = 'active'
    SUSPENDED    = 'suspended'
    TERMINATED   = 'terminated'

@dataclass
class HandshakePermissions:
    can_read_circle:   bool = True
    can_write_circle:  bool = True
    can_see_tool_use:  bool = False
    can_see_queries:   bool = False
    auto_share_new:    bool = False
    max_zone_size:     int  = 1000

@dataclass
class HandshakeSession:
    session_id:    str
    initiator_id:  str
    receiver_id:   str
    handshake_type: HandshakeType
    permissions:   HandshakePermissions
    status:        HandshakeStatus
    shared_attractors: Set[int]

class FabricHandshake:
    def initiate(self, my_fabric, partner_id, handshake_type, permissions):
        return f"sess_{my_fabric.fabric_id}_{partner_id}"
