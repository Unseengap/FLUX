"""
API Blueprint - REST endpoints for inference

Provides:
- /api/generate - SSE streaming generation with thinking process
- /api/learn - One-shot fact learning
- /api/recall - Memory recall
- /api/stats - Model statistics
- /api/health - Health check

All endpoints include comprehensive logging.
"""

import json
import time
import logging
from flask import Blueprint, request, Response, current_app, jsonify, g

api_bp = Blueprint('api', __name__)

def get_logger():
    """Get the app logger."""
    return logging.getLogger('flux_app')


@api_bp.route('/generate', methods=['POST'])
def generate():
    """
    Generate response with character-by-character SSE streaming.
    
    Shows full thinking process:
    - Phase 1: CSE encoding
    - Phase 2: Field query
    - Phase 3: Gravitational relevance
    - Phase 4: Thermodynamic settling
    - Phase 5: Causal geometry
    - Phase 6: Memory routing
    - Phase 8: WaveDecoder generation (streamed)
    """
    logger = get_logger()
    request_id = getattr(g, 'request_id', 'unknown')
    
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', current_app.config['MAX_LENGTH'])
    temperature = data.get('temperature', current_app.config['TEMPERATURE'])
    learn = data.get('learn', False)
    
    logger.info(f"│ [GENERATE] prompt_len={len(prompt)}, max_length={max_length}, temp={temperature}, learn={learn}")
    
    if not prompt:
        logger.warning(f"│ [GENERATE] Empty prompt rejected")
        return jsonify({'error': 'No prompt provided'}), 400
    
    service = current_app.flux_service
    
    if not service.is_loaded():
        logger.warning(f"│ [GENERATE] Model not loaded - returning 503")
        return jsonify({'error': 'Model not loaded'}), 503
    
    logger.info(f"│ [GENERATE] Starting SSE stream for: '{prompt[:50]}...' " if len(prompt) > 50 else f"│ [GENERATE] Starting SSE stream for: '{prompt}'")
    
    def event_stream():
        """Generate SSE events with thinking process."""
        gen_start = time.time()
        event_count = 0
        char_count = 0
        
        try:
            # Stream thinking process + generation
            for event in service.generate_with_thinking(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                learn=learn,
            ):
                event_count += 1
                
                # Log phase transitions
                if event.get('type') == 'phase':
                    phase = event.get('phase', '?')
                    name = event.get('name', 'unknown')
                    status = event.get('status', 'unknown')
                    logger.debug(f"│ [GENERATE] Phase {phase} ({name}): {status}")
                    if event.get('details'):
                        for k, v in event['details'].items():
                            logger.debug(f"│   {k}: {v}")
                
                # Count characters
                if event.get('type') == 'char':
                    char_count += 1
                
                yield f"data: {json.dumps(event)}\n\n"
            
            gen_time = (time.time() - gen_start) * 1000
            logger.info(f"│ [GENERATE] Stream complete: {char_count} chars, {event_count} events, {gen_time:.2f}ms")
            
            # End of stream
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            logger.error(f"│ [GENERATE] Stream error: {type(e).__name__}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@api_bp.route('/learn', methods=['POST'])
def learn_fact():
    """
    One-shot fact learning via episodic memory.
    
    This is FLUX's key differentiator: instant learning without fine-tuning.
    """
    logger = get_logger()
    request_id = getattr(g, 'request_id', 'unknown')
    
    data = request.get_json()
    fact = data.get('fact', '')
    
    logger.info(f"│ [LEARN] Attempting to learn fact: '{fact[:80]}...' " if len(fact) > 80 else f"│ [LEARN] Attempting to learn fact: '{fact}'")
    
    if not fact:
        logger.warning(f"│ [LEARN] Empty fact rejected")
        return jsonify({'error': 'No fact provided'}), 400
    
    service = current_app.flux_service
    
    if not service.is_loaded():
        logger.warning(f"│ [LEARN] Model not loaded - returning 503")
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        learn_start = time.time()
        result = service.learn_fact(fact)
        learn_time = (time.time() - learn_start) * 1000
        
        logger.info(f"│ [LEARN] ✓ Fact learned in {learn_time:.2f}ms")
        logger.info(f"│ [LEARN] Episodic size: {result.get('episodic_size')}, Learning steps: {result.get('learning_steps')}")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"│ [LEARN] ✗ Failed: {type(e).__name__}: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/recall', methods=['POST'])
def recall():
    """
    Query episodic memory for related facts.
    """
    logger = get_logger()
    request_id = getattr(g, 'request_id', 'unknown')
    
    data = request.get_json()
    query = data.get('query', '')
    k = data.get('k', 5)
    
    logger.info(f"│ [RECALL] Query: '{query[:50]}...', k={k}" if len(query) > 50 else f"│ [RECALL] Query: '{query}', k={k}")
    
    if not query:
        logger.warning(f"│ [RECALL] Empty query rejected")
        return jsonify({'error': 'No query provided'}), 400
    
    service = current_app.flux_service
    
    if not service.is_loaded():
        logger.warning(f"│ [RECALL] Model not loaded - returning 503")
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        recall_start = time.time()
        results = service.recall(query, k=k)
        recall_time = (time.time() - recall_start) * 1000
        
        logger.info(f"│ [RECALL] ✓ Found {len(results)} results in {recall_time:.2f}ms")
        for i, r in enumerate(results[:3]):
            logger.debug(f"│ [RECALL]   {i+1}. sim={r.get('similarity', 0):.4f}: {r.get('fact', '')[:40]}...")
        
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"│ [RECALL] ✗ Failed: {type(e).__name__}: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/stats', methods=['GET'])
def stats():
    """Get model statistics."""
    logger = get_logger()
    
    service = current_app.flux_service
    
    if not service.is_loaded():
        logger.debug(f"│ [STATS] Model not loaded")
        return jsonify({'loaded': False})
    
    stats = service.get_stats()
    logger.info(f"│ [STATS] Model stats retrieved: {stats.get('total_params')} params, {stats.get('episodic_entries')} episodic entries")
    
    return jsonify({
        'loaded': True,
        'stats': stats,
    })


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    logger = get_logger()
    
    model_loaded = current_app.flux_service.is_loaded()
    status = 'healthy' if model_loaded else 'model_not_loaded'
    
    logger.debug(f"│ [HEALTH] Status: {status}")
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
    })
