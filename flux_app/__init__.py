"""
FLUX Chat Application Factory

Modular Flask app with Blueprints for:
- Chat UI (dark theme, thinking process visualization)
- API (SSE streaming, character-by-character generation)
- Model (FLUX Beta loading and inference)

Includes comprehensive logging for all request flows.
"""

import logging
import sys
from datetime import datetime
from flask import Flask, request, g
from pathlib import Path


def setup_logging(app: Flask, log_level: str = 'DEBUG') -> logging.Logger:
    """Configure detailed logging for Flask app."""
    
    # Create logger
    logger = logging.getLogger('flux_app')
    logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with detailed formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'flask_app.log'
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s │ %(levelname)-8s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Also configure werkzeug logger
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.INFO)
    
    return logger


def create_app(config_name: str = 'development') -> Flask:
    """Application factory pattern."""
    
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
    )
    
    # Configuration
    app.config.update(
        SECRET_KEY='flux-physics-inspired-ai',
        MODEL_PATH=str(Path(__file__).parent.parent / 'checkpoints' / 'Flux-beta.flx'),
        DEVICE='auto',  # auto-detect: cuda > mps > cpu
        MAX_LENGTH=150,
        TEMPERATURE=0.8,
        LOG_LEVEL='DEBUG',
    )
    
    # Setup logging
    logger = setup_logging(app, app.config.get('LOG_LEVEL', 'DEBUG'))
    app.logger = logger
    
    logger.info("═" * 60)
    logger.info("FLUX Chat Application Starting")
    logger.info(f"Config: {config_name}")
    logger.info(f"Model Path: {app.config['MODEL_PATH']}")
    logger.info(f"Device: {app.config['DEVICE']}")
    logger.info("═" * 60)
    
    # Request lifecycle logging
    @app.before_request
    def log_request_start():
        """Log incoming request details."""
        g.request_start_time = datetime.now()
        g.request_id = f"{g.request_start_time.strftime('%H%M%S')}-{id(request) % 10000:04d}"
        
        logger.info(f"┌─ REQUEST [{g.request_id}] ─────────────────────────────")
        logger.info(f"│ Method: {request.method}")
        logger.info(f"│ Path: {request.path}")
        logger.info(f"│ Remote: {request.remote_addr}")
        
        if request.args:
            logger.debug(f"│ Query: {dict(request.args)}")
        
        if request.is_json:
            try:
                data = request.get_json(silent=True)
                if data:
                    # Truncate long values for logging
                    log_data = {}
                    for k, v in data.items():
                        if isinstance(v, str) and len(v) > 100:
                            log_data[k] = v[:100] + '...'
                        else:
                            log_data[k] = v
                    logger.debug(f"│ JSON Body: {log_data}")
            except Exception:
                pass
    
    @app.after_request
    def log_request_end(response):
        """Log request completion and timing."""
        duration_ms = 0
        if hasattr(g, 'request_start_time'):
            duration = datetime.now() - g.request_start_time
            duration_ms = duration.total_seconds() * 1000
        
        request_id = getattr(g, 'request_id', 'unknown')
        
        # Color-code by status
        status = response.status_code
        if status < 300:
            level = logging.INFO
            status_emoji = "✓"
        elif status < 400:
            level = logging.INFO
            status_emoji = "→"
        elif status < 500:
            level = logging.WARNING
            status_emoji = "⚠"
        else:
            level = logging.ERROR
            status_emoji = "✗"
        
        logger.log(level, f"│ Status: {status} {status_emoji}")
        logger.log(level, f"│ Duration: {duration_ms:.2f}ms")
        logger.log(level, f"│ Size: {response.content_length or 0} bytes")
        logger.log(level, f"└─ END [{request_id}] ─────────────────────────────────")
        
        return response
    
    @app.errorhandler(Exception)
    def log_exception(error):
        """Log unhandled exceptions."""
        request_id = getattr(g, 'request_id', 'unknown')
        logger.error(f"│ EXCEPTION [{request_id}]: {type(error).__name__}")
        logger.error(f"│ Message: {str(error)}")
        logger.exception("│ Traceback:")
        
        from flask import jsonify
        return jsonify({
            'error': str(error),
            'type': type(error).__name__
        }), 500
    
    # Register blueprints
    from flux_app.blueprints.chat import chat_bp
    from flux_app.blueprints.api import api_bp
    from flux_app.blueprints.model import model_bp
    
    app.register_blueprint(chat_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(model_bp, url_prefix='/model')
    
    logger.info("Blueprints registered: chat, api, model")
    
    # Initialize model service (singleton)
    with app.app_context():
        from flux_app.services.flux_service import FluxService
        app.flux_service = FluxService(app.config)
        logger.info("FluxService initialized")
    
    logger.info("Application factory complete - ready to serve")
    
    return app
