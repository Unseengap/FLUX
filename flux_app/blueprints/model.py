"""
Model Blueprint - Model management routes

Provides:
- /model/load - Load the FLUX model
- /model/unload - Unload to free memory
- /model/info - Model architecture info

All routes include comprehensive logging.
"""

import time
import logging
from flask import Blueprint, jsonify, current_app, request, g

model_bp = Blueprint('model', __name__)


def get_logger():
    """Get the app logger."""
    return logging.getLogger('flux_app')


@model_bp.route('/load', methods=['POST'])
def load_model():
    """Load the FLUX model from checkpoint."""
    logger = get_logger()
    request_id = getattr(g, 'request_id', 'unknown')
    
    service = current_app.flux_service
    
    if service.is_loaded():
        logger.info(f"│ [MODEL] Load requested but model already loaded")
        return jsonify({'status': 'already_loaded', 'message': 'Model already loaded'})
    
    logger.info(f"│ [MODEL] Loading FLUX model...")
    logger.info(f"│ [MODEL] Model path: {current_app.config.get('MODEL_PATH')}")
    logger.info(f"│ [MODEL] Device: {current_app.config.get('DEVICE')}")
    
    try:
        load_start = time.time()
        result = service.load_model()
        load_time = (time.time() - load_start) * 1000
        
        logger.info(f"│ [MODEL] ✓ Model loaded successfully in {load_time:.2f}ms")
        logger.info(f"│ [MODEL] Stats:")
        for k, v in result.items():
            logger.info(f"│   {k}: {v}")
        
        return jsonify({'status': 'loaded', 'stats': result})
    except FileNotFoundError as e:
        logger.error(f"│ [MODEL] ✗ Checkpoint not found: {e}")
        return jsonify({'status': 'error', 'message': f'Checkpoint not found: {e}'}), 404
    except Exception as e:
        logger.error(f"│ [MODEL] ✗ Load failed: {type(e).__name__}: {e}")
        logger.exception("│ [MODEL] Traceback:")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@model_bp.route('/unload', methods=['POST'])
def unload_model():
    """Unload model to free memory."""
    logger = get_logger()
    request_id = getattr(g, 'request_id', 'unknown')
    
    service = current_app.flux_service
    
    was_loaded = service.is_loaded()
    logger.info(f"│ [MODEL] Unload requested (was_loaded={was_loaded})")
    
    try:
        service.unload_model()
        logger.info(f"│ [MODEL] ✓ Model unloaded, memory freed")
        return jsonify({'status': 'unloaded'})
    except Exception as e:
        logger.error(f"│ [MODEL] ✗ Unload failed: {type(e).__name__}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@model_bp.route('/info', methods=['GET'])
def model_info():
    """Get FLUX architecture information."""
    logger = get_logger()
    
    logger.debug(f"│ [MODEL] Architecture info requested")
    
    return jsonify({
        'name': 'FLUX Beta',
        'version': '1.0-beta',
        'architecture': {
            'phase1_cse': {
                'name': 'Continuous Semantic Encoder',
                'description': 'Raw UTF-8 bytes → 432-dim semantic waves',
                'no_tokenizer': True,
            },
            'phase2_field': {
                'name': 'Resonance Field',
                'description': '96³ × 512 energy landscape with attractors',
                'zero_forgetting': True,
            },
            'phase3_gravity': {
                'name': 'Gravitational Relevance',
                'description': 'O(log n) relevance via spatial mass',
                'replaces': 'Attention (O(n²))',
            },
            'phase4_thermo': {
                'name': 'Thermodynamic Learner',
                'description': 'Energy settling for learning without backprop',
                'no_gradients': True,
            },
            'phase5_cgn': {
                'name': 'Causal Geometry Nodes',
                'description': 'Multi-timescale processing with causal tracing',
                'nodes': '32 fast + 16 medium + 8 slow',
            },
            'phase6_memory': {
                'name': 'Three-Tier Memory',
                'description': 'Working + Episodic + Semantic memory',
                'forgetting_score': 0.0000,
            },
            'phase8_decoder': {
                'name': 'WaveDecoder',
                'description': 'Autoregressive byte-level generation',
                'cross_attention': 'Attends to wave sequence',
            },
        },
        'key_properties': [
            'Zero catastrophic forgetting (0.0000 score)',
            'O(log n) scaling vs O(n²) for attention',
            'No tokenizer — works on raw UTF-8 bytes',
            'One-shot learning via episodic memory',
            'Causal traceability — knows WHY it knows',
            'Thermodynamic learning — no backpropagation',
        ],
    })
