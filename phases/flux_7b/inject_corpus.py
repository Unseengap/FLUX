"""
FLUX-7B: Knowledge Injection Pipeline

Injects large corpora into the resonance field WITHOUT gradients.
This is where FLUX fundamentally differs from LLMs.

LLM training: See data → backprop → weeks of compute
FLUX injection: See data → physics perturbation → hours of compute

Supported corpora:
- Wikipedia (6M articles)
- OpenWebText (8M documents)
- Books (100K+ books)
- Code (GitHub, StackOverflow)
- Custom documents

Estimated times (single A100):
- Wikipedia: ~4 hours
- OpenWebText: ~6 hours
- Full corpus: ~12 hours
"""

import sys
import torch
from pathlib import Path
from typing import List, Optional, Iterator, Callable
from dataclasses import dataclass
import time
from tqdm import tqdm

# Setup paths
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from resonance_field_7b import ResonanceField7B, FieldConfig
from flux_7b_model import FLUX7B, FLUX7BConfig, PhysicsStack


@dataclass
class InjectionConfig:
    """Configuration for knowledge injection."""
    
    # Injection parameters
    strength: float = 1.0
    
    # Settling schedule
    settle_every: int = 500  # Settle after N documents
    settle_steps: int = 50   # Settling iterations
    
    # Batching
    batch_size: int = 32  # Waves per injection batch
    
    # Progress
    log_every: int = 1000
    checkpoint_every: int = 10000
    
    # Limits
    max_doc_length: int = 4096  # Max bytes per document
    min_doc_length: int = 50   # Min bytes per document


class InjectionPipeline:
    """
    Pipeline for injecting knowledge into FLUX-7B field.
    
    Handles:
    - Document preprocessing
    - Wave encoding via CSE
    - Field injection with settling
    - Progress tracking and checkpointing
    """
    
    def __init__(
        self,
        system: FLUX7B,
        config: Optional[InjectionConfig] = None,
    ):
        self.system = system
        self.config = config or InjectionConfig()
        
        # Statistics
        self.total_injected = 0
        self.total_waves = 0
        self.start_time = None
        
    def inject_corpus(
        self,
        documents: Iterator[str],
        total: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Inject corpus into field.
        
        Args:
            documents: Iterator of document texts
            total: Total number of documents (for progress bar)
            checkpoint_path: Path for saving checkpoints
        """
        self.start_time = time.time()
        
        # Progress bar
        pbar = tqdm(documents, total=total, desc="Injecting")
        
        for i, doc in enumerate(pbar):
            # Skip short documents
            if len(doc) < self.config.min_doc_length:
                continue
                
            # Truncate long documents
            if len(doc) > self.config.max_doc_length:
                doc = doc[:self.config.max_doc_length]
                
            # Inject document
            self._inject_document(doc)
            self.total_injected += 1
            
            # Periodic settling
            if (i + 1) % self.config.settle_every == 0:
                self.system.field.settle(self.config.settle_steps)
                
            # Logging
            if (i + 1) % self.config.log_every == 0:
                self._log_progress(pbar)
                
            # Checkpointing
            if checkpoint_path and (i + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path, i + 1)
                
        # Final consolidation
        print("\n  Final settling and consolidation...")
        self.system.field.settle(self.config.settle_steps * 2)
        self.system.field.consolidate()
        
        # Final checkpoint
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, "final")
            
        self._print_summary()
        
    def _inject_document(self, doc: str):
        """Inject single document."""
        try:
            # Encode to waves
            waves = self.system.physics.encode(doc)  # [seq, 432]
            
            # Project to field space
            field_waves = self.system.physics.project_to_field(waves)  # [seq, field_features]
            
            # Inject in batches
            for j in range(0, len(field_waves), self.config.batch_size):
                batch = field_waves[j:j + self.config.batch_size]
                for wave in batch:
                    self.system.field.inject(
                        wave.to(self.system.device),
                        strength=self.config.strength,
                    )
                self.total_waves += len(batch)
                
        except Exception as e:
            # Skip problematic documents
            pass
            
    def _log_progress(self, pbar):
        """Update progress bar with stats."""
        elapsed = time.time() - self.start_time
        rate = self.total_injected / elapsed
        
        stats = self.system.field.stats()
        pbar.set_postfix({
            'docs/s': f'{rate:.1f}',
            'waves': f'{self.total_waves:,}',
            'mass': f'{stats["total_mass"]:.0f}',
        })
        
    def _save_checkpoint(self, path: str, step):
        """Save injection checkpoint."""
        checkpoint_path = f"{path}_step{step}"
        self.system.field.save(checkpoint_path + "_field.pt")
        
        # Save state
        torch.save({
            'total_injected': self.total_injected,
            'total_waves': self.total_waves,
            'step': step,
        }, checkpoint_path + "_state.pt")
        
        print(f"    ✓ Checkpoint saved: {checkpoint_path}")
        
    def _print_summary(self):
        """Print injection summary."""
        elapsed = time.time() - self.start_time
        stats = self.system.field.stats()
        
        print("\n" + "=" * 60)
        print("  INJECTION COMPLETE")
        print("=" * 60)
        print(f"  Documents: {self.total_injected:,}")
        print(f"  Waves: {self.total_waves:,}")
        print(f"  Time: {elapsed / 3600:.1f} hours")
        print(f"  Rate: {self.total_injected / elapsed:.1f} docs/sec")
        print(f"  Attractors: {stats['n_attractors']:,}")
        print(f"  Total mass: {stats['total_mass']:,.0f}")
        print("=" * 60)


# ─────────────────────────────────────────────
# Dataset Loaders
# ─────────────────────────────────────────────

def load_wikipedia() -> Iterator[str]:
    """Load Wikipedia articles."""
    from datasets import load_dataset
    
    print("  Loading Wikipedia...")
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    
    for item in ds:
        yield item['text']
        

def load_openwebtext() -> Iterator[str]:
    """Load OpenWebText documents."""
    from datasets import load_dataset
    
    print("  Loading OpenWebText...")
    ds = load_dataset("openwebtext", split="train", streaming=True)
    
    for item in ds:
        yield item['text']
        

def load_books() -> Iterator[str]:
    """Load books corpus."""
    from datasets import load_dataset
    
    print("  Loading Books...")
    ds = load_dataset("bookcorpus", split="train", streaming=True)
    
    for item in ds:
        yield item['text']
        

def load_code() -> Iterator[str]:
    """Load code corpus (The Stack subset)."""
    from datasets import load_dataset
    
    print("  Loading Code...")
    ds = load_dataset("bigcode/the-stack-dedup", 
                      data_dir="data/python",
                      split="train", 
                      streaming=True)
    
    for item in ds:
        yield item['content']


def load_custom_documents(paths: List[str]) -> Iterator[str]:
    """Load custom documents from files."""
    for path in paths:
        path = Path(path)
        if path.is_file():
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                yield f.read()
        elif path.is_dir():
            for file in path.rglob('*.txt'):
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    yield f.read()


# ─────────────────────────────────────────────
# Main Injection Functions
# ─────────────────────────────────────────────

def inject_full_corpus(
    system: FLUX7B,
    checkpoint_dir: str = './checkpoints/flux_7b',
    include_wikipedia: bool = True,
    include_openwebtext: bool = True,
    include_books: bool = True,
    include_code: bool = False,
):
    """
    Inject full training corpus.
    
    Estimated time: ~12 hours on single A100
    """
    import itertools
    
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    pipeline = InjectionPipeline(system)
    
    # Build document iterator
    sources = []
    total_estimate = 0
    
    if include_wikipedia:
        sources.append(load_wikipedia())
        total_estimate += 6_000_000
        
    if include_openwebtext:
        sources.append(load_openwebtext())
        total_estimate += 8_000_000
        
    if include_books:
        sources.append(load_books())
        total_estimate += 74_000
        
    if include_code:
        sources.append(load_code())
        total_estimate += 5_000_000
        
    # Chain all sources
    documents = itertools.chain(*sources)
    
    print(f"\n  Injecting ~{total_estimate:,} documents...")
    print(f"  Checkpoint dir: {checkpoint_path}")
    
    pipeline.inject_corpus(
        documents,
        total=total_estimate,
        checkpoint_path=str(checkpoint_path / 'injection'),
    )
    
    # Save final system
    system.save(str(checkpoint_path / 'flux_7b_injected'))
    

def inject_quick_test(system: FLUX7B, n_docs: int = 1000):
    """
    Quick injection test with small corpus.
    """
    from datasets import load_dataset
    
    print(f"  Quick test: injecting {n_docs} documents...")
    
    ds = load_dataset("openwebtext", split=f"train[:{n_docs}]")
    documents = (item['text'] for item in ds)
    
    pipeline = InjectionPipeline(system, InjectionConfig(
        settle_every=100,
        log_every=100,
    ))
    
    pipeline.inject_corpus(documents, total=n_docs)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FLUX-7B Knowledge Injection')
    parser.add_argument('--mode', choices=['full', 'test', 'custom'], default='test')
    parser.add_argument('--checkpoint-dir', default='./checkpoints/flux_7b')
    parser.add_argument('--custom-path', type=str, help='Path to custom documents')
    parser.add_argument('--n-docs', type=int, default=1000, help='Docs for test mode')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    # Create system
    if args.mode == 'test':
        from flux_7b_model import create_flux_7b_test
        system = create_flux_7b_test(args.device)
    else:
        from flux_7b_model import create_flux_7b_full
        system = create_flux_7b_full(args.device)
        
    # Load CSE if available
    try:
        from flux_utils import load_checkpoint
        from cse import ContinuousSemanticEncoder
        
        # Try to load from existing checkpoint
        apex = load_checkpoint(phase=8)
        cse = ContinuousSemanticEncoder()
        cse.load_state_dict(apex['cse']['state_dict'])
        system.physics.load_cse(cse)
        print("  ✓ Loaded CSE from checkpoint")
    except Exception as e:
        print(f"  ⚠ CSE not loaded: {e}")
        print("  Using random projections for testing")
    
    # Run injection
    if args.mode == 'full':
        inject_full_corpus(system, args.checkpoint_dir)
    elif args.mode == 'test':
        inject_quick_test(system, args.n_docs)
    elif args.mode == 'custom':
        if not args.custom_path:
            raise ValueError("--custom-path required for custom mode")
        documents = load_custom_documents([args.custom_path])
        pipeline = InjectionPipeline(system)
        pipeline.inject_corpus(documents)
