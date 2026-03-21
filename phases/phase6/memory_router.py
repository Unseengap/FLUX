import torch
from working_memory import WorkingMemory
from episodic_memory import EpisodicMemory
from semantic_memory import SemanticMemory

class MemoryRouter:
    """
    Routes between memory tiers.
    """
    def __init__(self, working: WorkingMemory, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic

    def route_query(self, query_wave: torch.Tensor):
        # 1. Check Working Memory
        context = self.working.get_current_context()

        # 2. Query Episodic
        episodic_facts = self.episodic.search(query_wave, k=3)

        # 3. Semantic memory is always on (part of the field)
        return {
            'context': context,
            'episodic': episodic_facts,
            'semantic': "Field retrieval active"
        }
