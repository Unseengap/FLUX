"""
Chat Blueprint - Main UI routes

Serves the dark-themed chat interface with thinking process visualization.
All routes include request logging.
"""

import logging
from flask import Blueprint, render_template, current_app, g

chat_bp = Blueprint('chat', __name__)


def get_logger():
    """Get the app logger."""
    return logging.getLogger('flux_app')


@chat_bp.route('/')
def index():
    """Main chat interface."""
    logger = get_logger()
    logger.info(f"│ [CHAT] Serving main chat interface")
    return render_template('chat.html')


@chat_bp.route('/about')
def about():
    """About FLUX architecture."""
    logger = get_logger()
    logger.info(f"│ [CHAT] Serving about page")
    return render_template('about.html')


@chat_bp.route('/architecture')
def architecture():
    """Interactive FLUX architecture presentation."""
    logger = get_logger()
    logger.info(f"│ [CHAT] Serving architecture page")
    return render_template('architecture.html')


@chat_bp.route('/status')
def status():
    """Model status page."""
    logger = get_logger()
    service = current_app.flux_service
    
    is_loaded = service.is_loaded()
    stats = service.get_stats() if is_loaded else None
    
    logger.info(f"│ [CHAT] Status page - model_loaded={is_loaded}")
    if stats:
        logger.debug(f"│ [CHAT] Stats: params={stats.get('total_params')}, device={stats.get('device')}")
    
    return render_template('status.html', stats=stats, loaded=is_loaded)
