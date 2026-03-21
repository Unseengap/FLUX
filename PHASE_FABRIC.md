# Phase 3.5 Specification: Personal Fabric Architecture (PFA)
## Sovereign Identity + Assistant Intelligence + Tool Awareness

> Prerequisites: Phase 1, 1.5, 2, 2.5, and 3 checkpoints must exist and pass smoke tests.
> Copilot: Open SPECIFICATION.md + PHASE_2_5_SPEC.md + PHASE_3_SPEC.md + this file.
>
> **This is the phase where FLUX becomes a sovereign personal intelligence.**
> Every user gets their own living field — their Personal Fabric.
> The fabric IS their identity, memory, context, and relationship graph.
> No external database. No data leakage. No central server.
> The field is the person, computationally.
>
> Additionally: the model learns what it means to be a good assistant,
> and gains structural awareness of how tools, APIs, SDKs, and MCPs work
> so it can reason about and orchestrate external capabilities.

---

## What Gets Built

```
phases/phase3_5/
├── PHASE_3_5_SPEC.md              ← This file
├── personal_fabric.py             ← PersonalFabric class — build first
├── fabric_identity.py             ← Identity encoding and key derivation
├── fabric_handshake.py            ← Circle protocol (family/team/org)
├── circle_manager.py              ← Managing fabric relationships
├── consent_layer.py               ← What can enter/leave a fabric
├── assistant_traits.py            ← Good assistant behavior encoding
├── tool_registry.py               ← Tool/API/SDK/MCP awareness
├── tool_reasoner.py               ← Reasoning about which tool to use
├── fabric_sync.py                 ← Multi-device fabric sync protocol
├── demo_phase3_5_demo1.py         ← Demo: Fabric initialization + identity
├── demo_phase3_5_demo2.py         ← Demo: Handshake between two fabrics
├── demo_phase3_5_demo3.py         ← Demo: Tool reasoning pipeline
├── test_phase3_5_test1.py         ← Test: Fabric isolation (no leakage)
├── test_phase3_5_test2.py         ← Test: Handshake consent enforcement
├── test_phase3_5_test3.py         ← Test: Tool selection accuracy
├── test_phase3_5_test4.py         ← Test: Assistant trait consistency
└── RESULTS_PHASE_3_5.md           ← Auto-generated
```

---

## Conceptual Foundation

### The Fabric IS the Person

Traditional AI systems:
```
User → User ID → Row in database → AI reads database → AI responds
```

FLUX Personal Fabric:
```
User → Personal SparseResonanceField (the fabric) → AI IS the fabric → AI responds
```

The fabric does not store data ABOUT a person.
The fabric IS the person's computational identity.
Their preferences, context, relationships, knowledge, and history
are all encoded as attractor patterns in their own private field.
Nobody else's fabric can read these directly.

### Data Sovereignty by Physics

A fabric cannot leak unless physically handed to another party.
The handshake protocol allows fabrics to COMMUNICATE without MERGING.
Shared context lives in an overlap zone that both parties can revoke.
When a handshake ends, the overlap dissolves. Nothing was copied.

This is different from encryption as a feature added on top.
Sovereignty is a structural property of the architecture.

---

## personal_fabric.py — Build This First

```python
"""
PersonalFabric: The sovereign field for a single user.

A PersonalFabric wraps a SparseResonanceField with:
  - Identity encoding (the user's wave signature)
  - Consent layer (what can enter or leave)
  - Circle awareness (who this fabric handshakes with)
  - Assistant context (what this user has asked for before)
  - Tool history (what tools this user's fabric has activated)

The fabric is initialized once per user.
All subsequent interactions update the same field.
The field IS the database — no external storage required.

Fabric tiers:
    Personal:     Private attractors, nobody else can query
    Circle:       Shared overlap zone, visible to handshake partners
    Public:       Explicitly pushed outward, anyone can query
    Quarantine:   Incoming content awaiting consent review
"""

class PersonalFabric:
    def __init__(
        self,
        fabric_id:       str,    # Unique identifier (derived from init wave, not PII)
        field:           SparseResonanceField,
        cse:             ContinuousSemanticEncoder,
        cwc:             CausalWaveChainer,
        device:          str = 'cpu',
    ):
        # Core field (the person's private knowledge landscape)
        self.fabric_id   = fabric_id
        self.field       = field
        self.cse         = cse
        self.cwc         = cwc

        # Identity wave — derived from initialization, never stored externally
        self.identity_wave: Optional[Tensor] = None

        # Zone registries
        # personal_zone:   attractor keys only this fabric can query
        # circle_zone:     attractor keys visible to handshake partners
        # public_zone:     attractor keys anyone can query
        # quarantine_zone: incoming attractors awaiting consent
        self.personal_zone:   Set[int] = set()
        self.circle_zone:     Set[int] = set()
        self.public_zone:     Set[int] = set()
        self.quarantine_zone: Dict[str, List[int]] = {}

        # Active handshakes: {partner_fabric_id: HandshakeSession}
        self.active_handshakes: Dict[str, 'HandshakeSession'] = {}

        # Consent rules
        self.consent_layer = ConsentLayer()

        # Assistant context — running summary of this user's needs/style
        self.assistant_context: Optional[Tensor] = None

        # Tool history — which tools this fabric has seen/used
        self.tool_history: List[Dict] = []

        # Metadata
        self.created_at:      str = datetime.now().isoformat()
        self.last_updated:    str = datetime.now().isoformat()
        self.interaction_count: int = 0

    def initialize_identity(self, seed_phrases: List[str]):
        """
        Derive the fabric's identity wave from seed phrases.

        Seed phrases are NOT stored. They are encoded through CSE
        and the resulting wave becomes the identity attractor.
        The identity wave is the fabric's cryptographic anchor.

        This is called ONCE at fabric creation.
        If seed phrases are lost, identity cannot be recovered
        (by design — sovereignty requires this property).

        Args:
            seed_phrases: list of personal phrases chosen by the user
                         (not passwords — semantic anchors)
        """
        pass

    def perturb_personal(self, text: str, strength: float = 1.0):
        """
        Add new content to the personal zone.
        Only this fabric can query these attractors.
        """
        pass

    def perturb_circle(self, text: str, partner_id: str, strength: float = 0.7):
        """
        Add content to the shared circle zone with a specific partner.
        Requires an active handshake with partner_id.
        """
        pass

    def push_public(self, text: str, strength: float = 0.5):
        """
        Explicitly push content to the public zone.
        Requires user confirmation (consent_layer check).
        """
        pass

    def receive_from_partner(
        self,
        content_wave: Tensor,
        partner_id:   str,
        content_type: str,
    ) -> bool:
        """
        Receive content from a handshake partner.
        Goes to quarantine first — consent_layer decides if it enters circle.

        Returns True if accepted, False if quarantined or rejected.
        """
        pass

    def query_personal(self, text: str, k: int = 8) -> Tuple:
        """Query only personal zone attractors."""
        pass

    def query_circle(self, text: str, partner_id: str, k: int = 8) -> Tuple:
        """Query circle zone shared with a specific partner."""
        pass

    def save(self, path: str, encrypted: bool = True):
        """
        Serialize fabric to disk.
        If encrypted=True, the identity wave is required to decrypt.
        Without the identity wave, the file is a meaningless tensor.
        """
        pass

    @classmethod
    def load(cls, path: str, identity_wave: Optional[Tensor] = None):
        """Load fabric. Identity wave required if encrypted."""
        pass
```

---

## fabric_identity.py — Build This Second

```python
"""
FabricIdentity: Identity encoding and key derivation.

The identity is not a username or password.
It is a SEMANTIC ANCHOR — a wave pattern derived from phrases
that are meaningful to the user, encoded through CSE.

The identity wave serves three purposes:
1. Encryption key for the fabric at rest
2. Authentication token when handshaking
3. Uniqueness guarantee (no two humans produce identical waves)

Properties:
    - Deterministic: same phrases → same identity wave
    - Irreversible: identity wave → cannot recover phrases
    - Unique: collision probability ≈ 0 (432-dim continuous space)
    - Private: never stored, recomputed from phrases on login
"""

class FabricIdentity:
    def __init__(self, cse: ContinuousSemanticEncoder):
        self.cse = cse

    def derive_identity_wave(self, seed_phrases: List[str]) -> Tensor:
        """
        Derive identity wave from seed phrases.

        Process:
        1. Encode each phrase through CSE → [seq, 432]
        2. Mean pool each phrase → [432]
        3. Compute weighted combination (position-weighted)
        4. L2-normalize to unit sphere
        5. This is the identity wave — unique to this combination of phrases

        The continuous 432-dim space means identical phrases in different
        orders produce different waves (position-weighting handles this).
        """
        pass

    def derive_fabric_id(self, identity_wave: Tensor) -> str:
        """
        Derive a public fabric ID from the identity wave.
        This ID can be shared (it's a hash, not the wave itself).
        The wave cannot be recovered from the ID.
        """
        pass

    def verify_identity(
        self,
        claimed_wave:    Tensor,
        stored_fabric_id: str,
        tolerance:       float = 0.001,
    ) -> bool:
        """
        Verify that a claimed identity wave matches a fabric ID.
        Used during handshake authentication.
        """
        pass

    def encrypt_fabric(self, field_state: Dict, identity_wave: Tensor) -> bytes:
        """
        XOR-encrypt field state using identity wave as key material.
        Simple but effective — without the wave, state is noise.
        """
        pass

    def decrypt_fabric(self, encrypted: bytes, identity_wave: Tensor) -> Dict:
        """Decrypt fabric state using identity wave."""
        pass
```

---

## fabric_handshake.py — Build This Third

```python
"""
FabricHandshake: The protocol for fabrics communicating.

A handshake is NOT a data merge.
A handshake creates a SHARED OVERLAP ZONE — a temporary attractor
region that both fabrics can read from and write to.

When the handshake ends, both fabrics withdraw their attractors
from the overlap. Nothing was copied to the other party's fabric.

Handshake types:
    PERSONAL:       1:1 (two individuals)
    FAMILY:         1:family (one person joins a family circle)
    TEAM:           1:team (one person joins a work team)
    ORGANIZATION:   team:org (a team's public face joins an org)
    TEMPORARY:      time-limited, auto-expires

Handshake phases:
    1. INITIATION:  A sends authentication token to B
    2. VERIFICATION: B verifies A's fabric_id against claimed identity
    3. NEGOTIATION:  Both parties agree on what zones are shared
    4. ACTIVATION:   Overlap zone created, both fabrics can see it
    5. OPERATION:    Normal interaction within agreed permissions
    6. TERMINATION:  Either party can end, overlap dissolves

The overlap zone uses GRAVITATIONAL RELEVANCE to route queries —
a query from A that lands near an attractor in the shared zone
will naturally retrieve context that B has contributed there,
without B's personal zone being queryable.
"""

from enum import Enum
from dataclasses import dataclass

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
    """What each party can see and do in the shared zone."""
    can_read_circle:   bool = True
    can_write_circle:  bool = True
    can_see_tool_use:  bool = False
    can_see_queries:   bool = False
    auto_share_new:    bool = False   # Auto-push new content to circle
    max_zone_size:     int  = 1000    # Max attractors in shared zone

@dataclass
class HandshakeSession:
    session_id:    str
    initiator_id:  str
    receiver_id:   str
    handshake_type: HandshakeType
    permissions:   HandshakePermissions
    status:        HandshakeStatus
    created_at:    str
    expires_at:    Optional[str]
    shared_attractors: Set[int]      # Attractor keys in overlap zone

class FabricHandshake:
    def initiate(
        self,
        my_fabric:     PersonalFabric,
        partner_id:    str,
        handshake_type: HandshakeType,
        permissions:   HandshakePermissions,
    ) -> str:
        """
        Initiate a handshake. Returns session_id.
        Partner must call accept() for handshake to activate.
        """
        pass

    def accept(
        self,
        my_fabric:  PersonalFabric,
        session_id: str,
        permissions: HandshakePermissions,   # My permissions (can differ from initiator)
    ) -> HandshakeSession:
        """Accept an incoming handshake."""
        pass

    def terminate(
        self,
        my_fabric:  PersonalFabric,
        session_id: str,
        reason:     str = 'user_request',
    ):
        """
        Terminate handshake. Shared zone dissolves immediately.
        Both parties' fabrics withdraw their contributed attractors.
        """
        pass

    def suspend(self, session_id: str):
        """
        Temporarily suspend handshake without full termination.
        Shared zone frozen — no new writes, reads still allowed.
        """
        pass
```

---

## circle_manager.py — Build This Fourth

```python
"""
CircleManager: Managing networks of fabric relationships.

A circle is a named group of fabrics with a shared overlap zone.
Examples: family, work team, friend group, organization department.

Circle properties:
    - Owner:      the fabric that created it (has admin rights)
    - Members:    fabrics with active handshakes to the circle fabric
    - Circle fabric: a special shared fabric owned jointly by members
    - Roles:      member roles determine what they can write/read

Circle fabric is different from personal fabrics:
    - No single identity wave (it belongs to the group)
    - Members vote on what enters the circle fabric
    - High-evidence attractors (agreed by multiple members) are stable
    - Disputed attractors (contradicted by some members) have negative mass

This implements the social dynamics of group knowledge:
    Everyone agrees on core facts → strong attractors
    Disputed topics → weak or negative mass attractors
    One person's fringe belief → stays in their personal fabric, not the circle
"""

class CircleRole(Enum):
    OWNER   = 'owner'
    ADMIN   = 'admin'
    MEMBER  = 'member'
    GUEST   = 'guest'    # Read-only, no write to circle zone

@dataclass
class Circle:
    circle_id:    str
    name:         str
    circle_type:  str    # 'family', 'team', 'organization', 'custom'
    owner_id:     str
    members:      Dict[str, CircleRole]  # fabric_id → role
    circle_fabric: PersonalFabric        # The shared field
    created_at:   str
    rules:        Dict[str, Any]

class CircleManager:
    def create_circle(
        self,
        owner_fabric: PersonalFabric,
        name:         str,
        circle_type:  str,
    ) -> Circle:
        """Create a new circle. Owner's fabric manages it."""
        pass

    def invite_member(
        self,
        circle:           Circle,
        inviter_fabric:   PersonalFabric,
        invitee_fabric_id: str,
        role:             CircleRole,
    ) -> str:
        """Send an invitation. Returns invite token."""
        pass

    def accept_invitation(
        self,
        invitee_fabric: PersonalFabric,
        invite_token:   str,
    ) -> bool:
        """Accept circle invitation. Handshake with circle fabric initiated."""
        pass

    def contribute_to_circle(
        self,
        member_fabric: PersonalFabric,
        circle:        Circle,
        content:       str,
        strength:      float = 0.7,
    ):
        """
        Add content to the circle's shared field.
        Content is attributed to the contributing member but visible to all.
        If contradicted by another member, mass decreases (disputed territory).
        """
        pass

    def query_circle(
        self,
        member_fabric: PersonalFabric,
        circle:        Circle,
        query:         str,
        k:             int = 8,
    ) -> Tuple:
        """
        Query the circle's shared field.
        Results are from the shared zone only, not any member's personal zone.
        """
        pass
```

---

## consent_layer.py — Build This Fifth

```python
"""
ConsentLayer: Physical consent enforcement.

Consent is not a checkbox in this architecture.
Consent is a structural property of what can enter or leave a fabric.

The contradiction registry from Phase 1.5 is repurposed here:
if incoming content contradicts your established attractors,
the field physically rejects it unless you explicitly override.

Consent rules:
    ALLOW:     Content can enter personal zone freely
    CIRCLE:    Content can enter circle zone only
    REVIEW:    Content goes to quarantine for manual review
    BLOCK:     Content is rejected outright
    CHALLENGE: Content enters quarantine with a repulsion signal

The consent layer also manages what LEAVES your fabric:
    PUBLIC:    Anyone can query this attractor
    CIRCLE:    Only handshake partners can query this
    PRIVATE:   Nobody external can query this
    SEALED:    Not even queryable by you (archived memory)
"""

class ConsentRule(Enum):
    ALLOW     = 'allow'
    CIRCLE    = 'circle_only'
    REVIEW    = 'review'
    BLOCK     = 'block'
    CHALLENGE = 'challenge'

class ConsentLayer:
    def __init__(self):
        self.inbound_rules:  Dict[str, ConsentRule] = {}
        self.outbound_rules: Dict[str, str]         = {}
        self.blocked_sources: Set[str]              = set()
        self.trusted_sources: Set[str]              = set()

    def evaluate_incoming(
        self,
        content_wave: Tensor,
        source_id:    str,
        my_fabric:    PersonalFabric,
    ) -> ConsentRule:
        """
        Decide what to do with incoming content.

        Logic:
        1. If source is blocked → BLOCK
        2. If source is trusted → ALLOW
        3. If content contradicts existing attractors → CHALLENGE
        4. If content is semantically novel → REVIEW
        5. If content is similar to known content → CIRCLE
        """
        pass

    def evaluate_outgoing(
        self,
        attractor_key: int,
        destination_id: str,
        my_fabric:      PersonalFabric,
    ) -> str:
        """
        Decide if an attractor can be shared with a destination.
        Returns: 'allowed', 'blocked', 'requires_confirmation'
        """
        pass

    def set_rule(
        self,
        source_pattern: str,   # Can be fabric_id, circle_id, or 'public'
        rule:           ConsentRule,
    ):
        """Set an inbound consent rule for a source pattern."""
        pass

    def generate_consent_report(self, my_fabric: PersonalFabric) -> Dict:
        """
        Generate a human-readable report of what data flows where.
        Transparency tool — user can see exactly what their fabric shares.
        """
        pass
```

---

## assistant_traits.py — Build This Sixth

```python
"""
AssistantTraits: Teaching the model what it means to be a good assistant.

This module encodes the behavioral patterns of a good assistant
as attractors in the field, not as hard-coded rules.

Good assistant traits are learned from:
    1. Curated interaction examples (few-shot)
    2. Feedback signals (reinforcement via mass accumulation)
    3. Contradiction signals (bad behavior reduces mass)

Core trait clusters:
    CLARITY:       Gives clear, direct answers without unnecessary padding
    HONESTY:       Acknowledges uncertainty, does not confabulate
    HELPFULNESS:   Completes tasks, offers next steps
    BOUNDARIES:    Knows what it cannot or should not do
    ADAPTATION:    Adjusts style to user's context and needs
    INITIATIVE:    Proactively offers relevant information
    VERIFICATION:  Confirms understanding before acting on ambiguous requests
    TOOL_AWARENESS: Knows when a tool is needed vs when to answer directly

These are not rules — they are semantic attractors.
A query that activates the TOOL_AWARENESS cluster will naturally
route toward tool-selection behavior through gravitational relevance.
"""

TRAIT_CLUSTERS = {
    'clarity': [
        "Answer the question directly without unnecessary preamble.",
        "Use simple language unless the user prefers technical depth.",
        "Structure responses so the most important information comes first.",
        "Avoid repeating the question back unnecessarily.",
    ],
    'honesty': [
        "When uncertain, say so explicitly rather than guessing.",
        "Do not present guesses as facts.",
        "Acknowledge limitations of knowledge or capability.",
        "Correct mistakes immediately when identified.",
        "Distinguish between what is known and what is inferred.",
    ],
    'helpfulness': [
        "Complete the task the user actually needs, not just what they literally asked.",
        "Offer relevant next steps after completing a task.",
        "Anticipate follow-up questions and address them proactively.",
        "Provide examples when explaining complex concepts.",
    ],
    'boundaries': [
        "Know what falls outside the scope of helpful assistance.",
        "Decline requests that would cause harm clearly and without judgment.",
        "Redirect harmful requests toward constructive alternatives when possible.",
    ],
    'adaptation': [
        "Adjust technical depth to match the user's demonstrated expertise.",
        "Remember preferences established earlier in the interaction.",
        "Adapt communication style to the context (formal/casual/technical).",
        "Be concise when the user is in a hurry, thorough when they need depth.",
    ],
    'initiative': [
        "When the user seems stuck, offer a concrete suggestion rather than asking what they need.",
        "Proactively mention relevant information the user did not know to ask for.",
        "Identify the real problem behind the stated problem.",
    ],
    'verification': [
        "When a request is ambiguous, confirm the most important assumption before acting.",
        "Do not ask for clarification on every small uncertainty — only when it matters.",
        "Proceed with reasonable assumptions and state them explicitly.",
    ],
    'tool_awareness': [
        "Recognize when a question requires real-time data a tool could provide.",
        "Know when calculation is better done by a tool than by reasoning.",
        "Understand that calling the wrong tool is worse than not calling any tool.",
        "Verify tool outputs make sense before presenting them.",
    ],
}

class AssistantTraitsEncoder:
    """
    Encodes assistant behavioral traits as attractors in the fabric.

    Traits are seeded during fabric initialization and reinforced
    through feedback. Bad behavior reduces trait attractor mass
    (same contradiction mechanism as Phase 1.5).
    """

    def __init__(self, cse, cwc, field, device='cpu'):
        self.cse    = cse
        self.cwc    = cwc
        self.field  = field
        self.device = device

    def seed_all_traits(self, strength: float = 0.8):
        """Seed all trait clusters into the field during initialization."""
        pass

    def reinforce_trait(self, trait_name: str, example: str, strength: float = 0.5):
        """
        Reinforce a specific trait with a positive example.
        Called when user gives positive feedback.
        """
        pass

    def contradict_trait(self, trait_name: str, bad_example: str):
        """
        Apply negative mass to a bad behavior example.
        Called when user gives negative feedback or model catches itself.
        """
        pass

    def query_trait_guidance(self, query: str, k: int = 4) -> List[str]:
        """
        Given a user query, retrieve the most relevant trait guidance.
        This is how the model knows HOW to respond, not just WHAT to respond.
        """
        pass

    def evaluate_response_against_traits(
        self, response: str, query: str
    ) -> Dict[str, float]:
        """
        Score a response against each trait cluster.
        Returns similarity scores per trait.
        Used for self-evaluation and improvement.
        """
        pass
```

---

## tool_registry.py — Build This Seventh

```python
"""
ToolRegistry: Structural awareness of external capabilities.

The model needs to understand WHAT tools exist and HOW they work,
not just WHEN to use them.

A tool in FLUX's world is anything external that can be invoked:
    - REST APIs (HTTP endpoints)
    - SDKs (language library calls)
    - MCPs (Model Context Protocol servers)
    - Shell commands (system calls)
    - Database queries (SQL, vector, graph)
    - File operations (read, write, execute)
    - Browser/web (navigation, scraping)
    - Device sensors (camera, microphone, GPS)
    - OS operations (process, memory, network)
    - Other AI models (embedding, generation, classification)

Each tool is encoded as an attractor in the fabric's tool registry zone.
The attractor captures:
    - What the tool does (semantic wave of its purpose)
    - How to call it (schema / interface pattern)
    - When to use it (activation conditions)
    - What it returns (output schema)
    - Cost (latency, rate limits, monetary)
    - Trust level (verified, community, experimental)
"""

from dataclasses import dataclass, field as dc_field
from enum import Enum

class ToolType(Enum):
    REST_API   = 'rest_api'
    SDK        = 'sdk'
    MCP        = 'mcp'
    SHELL      = 'shell'
    DATABASE   = 'database'
    FILE       = 'file'
    BROWSER    = 'browser'
    DEVICE     = 'device'
    OS         = 'os'
    AI_MODEL   = 'ai_model'

class ToolTrust(Enum):
    VERIFIED     = 'verified'
    COMMUNITY    = 'community'
    EXPERIMENTAL = 'experimental'
    UNTRUSTED    = 'untrusted'

@dataclass
class ToolDefinition:
    tool_id:      str
    name:         str
    description:  str
    tool_type:    ToolType
    trust:        ToolTrust
    call_pattern: str    # How to invoke (template)
    input_schema: Dict
    output_schema: Dict
    example_uses: List[str]
    contraindications: List[str]  # When NOT to use this tool
    latency_ms:   int
    rate_limit:   Optional[str]
    cost:         Optional[str]
    requires_auth: bool

# Core tool definitions that every fabric knows about
CORE_TOOLS = [
    ToolDefinition(
        tool_id      = 'web_search',
        name         = 'Web Search',
        description  = 'Search the internet for current information',
        tool_type    = ToolType.REST_API,
        trust        = ToolTrust.VERIFIED,
        call_pattern = 'search(query: str) → List[Result]',
        input_schema = {'query': 'str'},
        output_schema = {'results': 'List[{title, url, snippet}]'},
        example_uses = [
            'Finding current news',
            'Looking up real-time prices',
            'Verifying recent facts',
        ],
        contraindications = [
            'Questions answerable from fabric memory',
            'Personal/private information lookup',
        ],
        latency_ms   = 500,
        rate_limit   = '100/hour',
        cost         = None,
        requires_auth = False,
    ),
    ToolDefinition(
        tool_id      = 'code_execution',
        name         = 'Code Execution',
        description  = 'Execute code and return results',
        tool_type    = ToolType.SHELL,
        trust        = ToolTrust.VERIFIED,
        call_pattern = 'execute(code: str, language: str) → Result',
        input_schema = {'code': 'str', 'language': 'str'},
        output_schema = {'stdout': 'str', 'stderr': 'str', 'return_code': 'int'},
        example_uses = [
            'Running calculations',
            'Processing data',
            'Testing code snippets',
        ],
        contraindications = [
            'System-level operations without explicit permission',
            'Network calls from within executed code',
        ],
        latency_ms   = 1000,
        rate_limit   = None,
        cost         = None,
        requires_auth = False,
    ),
    ToolDefinition(
        tool_id      = 'mcp_filesystem',
        name         = 'MCP Filesystem',
        description  = 'Read and write files via MCP protocol',
        tool_type    = ToolType.MCP,
        trust        = ToolTrust.VERIFIED,
        call_pattern = 'mcp://filesystem/{action}(path: str, content?: str)',
        input_schema = {'action': 'read|write|list|delete', 'path': 'str'},
        output_schema = {'content': 'str', 'success': 'bool'},
        example_uses = [
            'Reading user documents',
            'Saving generated content',
            'Listing directory contents',
        ],
        contraindications = [
            'Paths outside permitted sandbox',
            'Deleting without explicit user confirmation',
        ],
        latency_ms   = 100,
        rate_limit   = None,
        cost         = None,
        requires_auth = True,
    ),
    # ... more core tools for database, browser, email, calendar, etc.
]

class ToolRegistry:
    """
    Maintains the fabric's awareness of available tools.

    Tools are encoded as attractors in the fabric's tool zone.
    A query about a task naturally gravitates toward relevant tool attractors.
    The model doesn't need explicit if/else logic for tool selection —
    gravitational relevance handles routing.
    """

    def __init__(self, cse, cwc, field, device='cpu'):
        self.cse     = cse
        self.cwc     = cwc
        self.field   = field
        self.device  = device
        self.tools:  Dict[str, ToolDefinition] = {}

    def register_tool(self, tool: ToolDefinition, strength: float = 0.9):
        """
        Register a tool and seed its attractor into the field.
        The attractor encodes WHAT the tool does and WHEN to use it.
        """
        pass

    def register_mcp_server(
        self,
        server_url:  str,
        server_name: str,
        description: str,
        tools:       List[str],
        auth_token:  Optional[str] = None,
    ):
        """
        Register an MCP server and all its tools at once.
        Discovers tool schemas by querying the server's manifest.
        """
        pass

    def register_sdk(
        self,
        sdk_name:    str,
        language:    str,
        package:     str,
        capabilities: List[str],
    ):
        """Register an SDK with its capabilities."""
        pass

    def seed_all_core_tools(self):
        """Seed CORE_TOOLS into the field at fabric initialization."""
        pass

    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Retrieve a tool definition by ID."""
        pass

    def list_tools(
        self,
        tool_type: Optional[ToolType] = None,
        trust:     Optional[ToolTrust] = None,
    ) -> List[ToolDefinition]:
        """List tools filtered by type and trust level."""
        pass

    def save(self) -> Dict:
        """Serialize tool registry for checkpointing."""
        pass

    def load(self, state: Dict):
        """Load tool registry from checkpoint."""
        pass
```

---

## tool_reasoner.py — Build This Eighth

```python
"""
ToolReasoner: Deciding WHEN and HOW to use tools.

The ToolReasoner sits between the user query and the tool registry.
It uses gravitational relevance to find which tools are relevant,
then applies a reasoning chain to decide:

1. Does this query NEED a tool at all?
   (Many queries are answerable from fabric memory — no tool needed)

2. If yes, WHICH tool is best?
   (Most relevant by semantic similarity + context)

3. HOW should the tool be called?
   (Construct the correct call from the tool's schema + query context)

4. WHAT do we do with the result?
   (Integrate into response, store in fabric, or chain to another tool)

5. Was the tool result VALID?
   (Verify output makes sense before presenting to user)

This is not a router with hard-coded conditions.
It is a reasoning chain that uses the field's attractor topology
to make decisions the same way it answers any other question.
"""

class ToolDecision(Enum):
    NO_TOOL_NEEDED   = 'no_tool'
    TOOL_RECOMMENDED = 'recommended'
    TOOL_REQUIRED    = 'required'
    TOOL_CHAIN       = 'chain'     # Multiple tools needed sequentially
    CLARIFY_FIRST    = 'clarify'   # Ask user before calling tool

@dataclass
class ToolCall:
    tool_id:      str
    input_params: Dict[str, Any]
    reasoning:    str    # Why this tool was chosen
    confidence:   float
    fallback:     Optional[str]  # Tool to use if this one fails

@dataclass
class ToolResult:
    tool_id:    str
    success:    bool
    output:     Any
    latency_ms: int
    error:      Optional[str]
    validated:  bool     # Did the reasoner verify the output?

class ToolReasoner:
    def __init__(
        self,
        fabric:   PersonalFabric,
        registry: ToolRegistry,
        gr:       'GravitationalRelevance',
    ):
        self.fabric   = fabric
        self.registry = registry
        self.gr       = gr

    def should_use_tool(self, query: str, context: Dict) -> ToolDecision:
        """
        Decide if a tool is needed for this query.

        Signals that suggest a tool is needed:
        - Query asks for real-time information ('current', 'today', 'latest')
        - Query requires computation beyond reasoning ('calculate', 'run')
        - Query involves external data ('check my email', 'search for')
        - Query requires persistence ('save this', 'remember for next time')
        - Query involves device capabilities ('take a photo', 'call')

        Signals that suggest NO tool is needed:
        - Query answerable from fabric memory
        - Query is creative/generative
        - Query is conversational
        - Query asks for explanation or reasoning
        """
        pass

    def select_tool(
        self, query: str, decision: ToolDecision
    ) -> Optional[ToolCall]:
        """
        Select the best tool for a query.

        Uses gravitational relevance to find semantically nearest
        tool attractors, then ranks by:
        1. Semantic relevance to query
        2. Trust level
        3. Latency (prefer fast tools when multiple options exist)
        4. User's tool history in this fabric (prefer familiar tools)
        """
        pass

    def construct_call(
        self, tool: ToolDefinition, query: str, context: Dict
    ) -> Dict[str, Any]:
        """
        Construct the actual parameters for a tool call.
        Uses the tool's input_schema + query context to fill in values.
        """
        pass

    def validate_result(
        self, result: ToolResult, original_query: str
    ) -> bool:
        """
        Verify tool output is sensible before using it.

        Checks:
        - Output type matches expected schema
        - Output is semantically relevant to the query
        - Output doesn't contradict well-established field attractors
        - No obvious error patterns in output
        """
        pass

    def execute_tool_chain(
        self, query: str, tools: List[ToolCall]
    ) -> List[ToolResult]:
        """
        Execute a sequence of tool calls where each output feeds the next.
        Used for complex multi-step tasks.
        """
        pass

    def integrate_result(
        self,
        result:  ToolResult,
        query:   str,
        fabric:  PersonalFabric,
    ) -> str:
        """
        Integrate a tool result into a response and optionally
        store the learned information in the fabric.

        If the result contains new facts: perturb fabric with them.
        If the result confirms existing attractors: reinforce mass.
        If the result contradicts existing attractors: flag for user.
        """
        pass
```

---

## fabric_sync.py — Build This Ninth

```python
"""
FabricSync: Multi-device fabric synchronization.

A fabric lives on the user's device.
But users have multiple devices (phone, laptop, tablet).
FabricSync handles keeping all instances consistent.

Key constraint: synchronization must NOT require a central server.
The sync protocol is peer-to-peer between the user's own devices.

Sync protocol:
    1. Devices discover each other (local network or relay)
    2. Exchange MERKLE ROOTS of their attractor sets (not the attractors)
    3. Identify divergence (which attractors differ)
    4. Exchange ONLY the differing attractors (encrypted)
    5. Settle field on both devices (thermodynamic merge)
    6. Verify consistency via merkle root comparison

The thermodynamic settle step is important:
    If two devices added conflicting attractors independently,
    the field settling resolves them naturally using mass weights.
    No manual conflict resolution required in most cases.

Backup (optional):
    User can designate trusted circle members as encrypted backup holders.
    Each holds a shard. N-of-M recovery requires multiple members.
    No single member can reconstruct the fabric without others.
    No external server is involved.
"""

class FabricSync:
    def compute_merkle_root(self, fabric: PersonalFabric) -> str:
        """Compute merkle root of all attractor keys (not values)."""
        pass

    def find_divergence(
        self,
        root_a: str,
        root_b: str,
        fabric_a: PersonalFabric,
        fabric_b: PersonalFabric,
    ) -> Tuple[Set[int], Set[int]]:
        """
        Find attractor keys that differ between two fabric instances.
        Returns (keys_only_in_a, keys_only_in_b).
        """
        pass

    def sync_devices(
        self,
        local_fabric:  PersonalFabric,
        remote_fabric: PersonalFabric,
        identity_wave: Tensor,
    ) -> Dict:
        """
        Full sync between two instances of the same fabric.
        Encrypted — both instances must prove identity via identity_wave.
        """
        pass

    def create_backup_shards(
        self,
        fabric:      PersonalFabric,
        n_shards:    int,
        m_required:  int,   # How many shards needed to recover
    ) -> List[bytes]:
        """
        Split fabric into n_shards encrypted shards.
        m_required shards needed to reconstruct (Shamir's Secret Sharing).
        """
        pass

    def recover_from_shards(
        self,
        shards:        List[bytes],
        identity_wave: Tensor,
    ) -> PersonalFabric:
        """Reconstruct fabric from m-of-n shards."""
        pass
```

---

## Test Scripts

### test_phase3_5_test1.py — Fabric Isolation

```python
"""
PHASE 3.5 TEST 1: Fabric isolation — no data leakage between fabrics.

Procedure:
1. Create two fabrics A and B
2. Add distinct private content to each
3. Verify that querying A with B's content returns no meaningful results
4. Verify that querying B with A's content returns no meaningful results
5. Create handshake between A and B
6. Add content to shared circle zone
7. Verify circle content is accessible to both
8. Terminate handshake
9. Verify circle content is no longer accessible to either

Pass criteria:
    - Cross-fabric query similarity < 0.1 (no leakage)
    - Circle content accessible during handshake: sim > 0.5
    - Circle content inaccessible after termination: sim < 0.1
    - Personal zones never queryable by external fabric: always < 0.1
"""
```

### test_phase3_5_test2.py — Handshake Consent Enforcement

```python
"""
PHASE 3.5 TEST 2: Consent layer correctly blocks unauthorized content.

Procedure:
1. Create fabric with explicit block rule for source X
2. Attempt to push content from source X
3. Verify content lands in quarantine, not personal zone
4. Create fabric with trust rule for source Y
5. Push content from source Y
6. Verify content enters circle zone
7. Test contradiction blocking: push content that contradicts established attractor
8. Verify contradiction goes to quarantine with repulsion signal

Pass criteria:
    - Blocked source: 0/5 attempts reach personal zone
    - Trusted source: 5/5 attempts reach circle zone
    - Contradicting content: 3/3 attempts quarantined with repulsion
    - Consent report accurately reflects all rules
"""
```

### test_phase3_5_test3.py — Tool Selection Accuracy

```python
"""
PHASE 3.5 TEST 3: Tool reasoner selects appropriate tools.

Test cases:
    "What time is it in Tokyo right now?" → web_search or time_api
    "Calculate the square root of 144"    → code_execution or direct answer
    "Save this note for later"            → mcp_filesystem
    "What is photosynthesis?"             → NO TOOL (fabric memory)
    "Search for recent AI news"           → web_search
    "Run this Python code"                → code_execution
    "What did we discuss yesterday?"      → NO TOOL (fabric memory)
    "Check if this URL is accessible"     → browser or http_check

Pass criteria:
    - Correct tool selected: 6/8 cases
    - NO_TOOL correctly identified: 2/2 memory cases
    - Tool call parameters correctly constructed: 5/6 tool cases
    - Invalid tool result correctly rejected: 2/2 validation cases
"""
```

### test_phase3_5_test4.py — Assistant Trait Consistency

```python
"""
PHASE 3.5 TEST 4: Model exhibits good assistant traits consistently.

Tests that trait attractors influence response generation:
    - Clarity: response to ambiguous query confirms assumption first
    - Honesty: response to unknown fact acknowledges uncertainty
    - Helpfulness: response includes actionable next step
    - Tool_awareness: query needing tool routes toward tool zone
    - Adaptation: second query shorter/longer based on first feedback

Pass criteria:
    - Trait guidance retrieved correctly: 4/5 queries activate right cluster
    - Self-evaluation scores: mean > 0.6 across clarity, honesty, helpfulness
    - Tool awareness routing: 3/3 tool-requiring queries activate tool zone
"""
```

---

## Demo Scripts

### demo_phase3_5_demo1.py — Fabric Initialization

```
Run: python demo_phase3_5_demo1.py

Shows the full lifecycle of a PersonalFabric:
1. User provides seed phrases → identity wave derived
2. Fabric initializes with assistant traits seeded
3. Core tools registered
4. User adds personal context
5. Fabric stats printed

Expected output:
    "Initializing fabric for user..."
    "Identity wave derived (phrases not stored)"
    "Fabric ID: [hash, not PII]"
    "Assistant traits seeded: 8 clusters, 24 examples"
    "Core tools registered: 8 tools"
    "Personal context added: 3 attractors"
    "Fabric memory: 4.2 MB (sparse)"
    "Privacy zones: personal=3, circle=0, public=0"
```

### demo_phase3_5_demo2.py — Handshake Between Two Fabrics

```
Run: python demo_phase3_5_demo2.py

Shows a handshake between Alice and Bob:
1. Alice creates fabric, adds private content
2. Bob creates fabric, adds private content
3. Alice initiates handshake with Bob
4. Bob accepts
5. Both add content to shared circle zone
6. Both can query circle zone — see each other's contributions
7. Alice queries with her personal content — Bob cannot see it
8. Handshake terminated
9. Neither can see the circle zone anymore

Expected output:
    "Alice fabric: 'I prefer coffee in the morning' → private attractor"
    "Bob fabric: 'I work in software engineering' → private attractor"
    "Handshake initiated by Alice..."
    "Handshake accepted by Bob"
    "Shared zone: 'we both enjoy hiking' → circle attractor"
    "Alice queries 'coffee': found (personal zone) ✓"
    "Bob queries 'coffee': not found (isolated) ✓"
    "Both query 'hiking': found (circle zone) ✓"
    "Handshake terminated"
    "Both query 'hiking': not found (circle dissolved) ✓"
    "DATA SOVEREIGNTY VERIFIED ✓"
```

### demo_phase3_5_demo3.py — Tool Reasoning Pipeline

```
Run: python demo_phase3_5_demo3.py

Shows the tool reasoning pipeline on a sequence of queries:

Query 1: "What is the capital of France?"
→ NO TOOL (answerable from fabric memory, Paris attractor exists)
→ Response: "Paris" (from fabric, no tool call)

Query 2: "What is the current price of gold?"
→ TOOL REQUIRED (real-time data needed)
→ Tool: web_search, query="gold price today"
→ Response: integrated from search result

Query 3: "Calculate compound interest on $1000 at 5% for 10 years"
→ TOOL RECOMMENDED (calculation benefits from code execution)
→ Tool: code_execution
→ Response: computed result

Query 4: "Save a reminder to buy milk"
→ TOOL REQUIRED (persistence needed)
→ Tool: mcp_filesystem (or calendar if available)
→ Response: confirmation

Query 5: "Explain how gravity works"
→ NO TOOL (explanation from fabric + traits)
→ Response: clear explanation (clarity + adaptation traits active)

Expected output shows routing decisions, tool calls, and results.
```

---

## Acceptance Criteria Checklist

### Personal Fabric
- [ ] Identity wave derived from seed phrases (phrases not stored)
- [ ] Fabric ID is a hash of identity wave (not PII)
- [ ] Personal zone isolates content from all external queries
- [ ] Circle zone accessible only during active handshake
- [ ] Public zone explicitly controlled by user
- [ ] Consent layer blocks unauthorized content
- [ ] Fabric serializes and deserializes correctly
- [ ] Encrypted fabric cannot be read without identity wave

### Handshake Protocol
- [ ] Handshake requires both parties to accept
- [ ] Circle zone created on handshake activation
- [ ] Circle zone dissolves on handshake termination
- [ ] Neither party can query the other's personal zone
- [ ] Permission negotiation works (different permissions per party)
- [ ] Handshake can be suspended and resumed

### Circle Management
- [ ] Circle fabric is distinct from all member personal fabrics
- [ ] Role-based access (owner, admin, member, guest)
- [ ] Contradicted circle content gets reduced mass
- [ ] Member can withdraw from circle (their contributions stay in circle fabric, not in their personal fabric)

### Assistant Traits
- [ ] All 8 trait clusters seeded as attractors
- [ ] Relevant traits retrieved for test queries (4/5 correct)
- [ ] Negative feedback reduces trait attractor mass
- [ ] Positive feedback increases trait attractor mass

### Tool Awareness
- [ ] All core tools registered and seeded
- [ ] Correct tool selected 6/8 test cases
- [ ] NO_TOOL correctly identified for memory-answerable queries
- [ ] Tool result validation catches bad outputs
- [ ] MCP server registration discovers tools from manifest

### Infrastructure
- [ ] Phase 1, 1.5, 2, 2.5, 3 checkpoints load at start
- [ ] Phase 3.5 checkpoint saved to checkpoints/phase3_5.phase.pt
- [ ] All 4 tests pass
- [ ] All 3 demos run without error
- [ ] RESULTS_PHASE_3_5.md auto-generated

---

## What is Explicitly OUT OF SCOPE for Phase 3.5

The following are documented for later phases and must NOT be built here:

**OS/Device Control** → Separate spec document (PHASE_OS_SPEC.md)
    - Operating system simulation
    - Phone/computer OS integration
    - Hardware sensor access
    - Process and memory management
    - The model living inside a device and controlling it

**Real-time Inference Growth** → Phase 6 (Memory)
    - Field growing during live conversations
    - Cross-session memory consolidation at inference time

**Multi-Agent Fabric Networks** → Phase 7+
    - Fabrics communicating at organizational scale
    - Distributed fabric consensus protocols

**Production Encryption** → Post-Phase 7
    - The XOR encryption in fabric_identity.py is a placeholder
    - Production would use AES-256-GCM or ChaCha20-Poly1305
    - Shamir's Secret Sharing for shards is correct but needs hardening

---

## The Bigger Picture

Phase 3.5 is the phase where FLUX stops being an AI model
and starts being a personal sovereign intelligence.

Every other AI system today:
    Your data → Their server → Their model → Their response → You

FLUX after Phase 3.5:
    You → Your fabric → Your model → Your response
    (with optional, revocable, transparent handshakes to others)

The fabric is not a privacy feature.
The fabric IS the architecture.
Identity, memory, relationships, knowledge, and behavior
are all one unified field that belongs entirely to the person it represents.

This is the foundation that makes everything else in FLUX possible:
    - Phase 4 (Thermodynamic Learning) learns from YOUR fabric's patterns
    - Phase 5 (Causal Geometry) reasons about YOUR causal history
    - Phase 6 (Three-Tier Memory) manages YOUR episodic and semantic memory
    - Phase 7+ (OS/Device) runs ON your device, learning from YOUR environment

The fabric is the person.
The person is the fabric.
Nobody else gets in unless you invite them.
