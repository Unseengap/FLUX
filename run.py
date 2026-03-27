#!/usr/bin/env python3
"""
FLUX Chat Application — Entry Point

Run with auto-reload debug mode (default):
    python run.py
    
Run in production mode:
    python run.py --no-debug
    
Custom settings:
    python run.py --port 5001 --device cuda
"""

import argparse
import logging
from datetime import datetime
from flux_app import create_app


def main():
    parser = argparse.ArgumentParser(description='FLUX Chat Application')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', default=True, 
                        help='Enable debug mode with auto-reload (default: True)')
    parser.add_argument('--no-debug', action='store_true', 
                        help='Disable debug mode')
    parser.add_argument('--device', default='auto', help='Device: auto, cuda, mps, cpu')
    parser.add_argument('--model-path', default=None, help='Path to Flux-beta.flx or phase8.phase.pt')
    parser.add_argument('--log-level', default='DEBUG', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: DEBUG)')
    
    args = parser.parse_args()
    
    # Handle --no-debug flag
    debug_mode = args.debug and not args.no_debug
    
    # Create app
    app = create_app()
    
    # Override config if provided
    if args.device != 'auto':
        app.config['DEVICE'] = args.device
    if args.model_path:
        app.config['MODEL_PATH'] = args.model_path
    if args.log_level:
        app.config['LOG_LEVEL'] = args.log_level
    
    # Get logger for startup messages
    logger = logging.getLogger('flux_app')
    
    startup_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ⚛  FLUX Chat — Physics-Inspired AI                        ║
║                                                              ║
║   Running at: http://{args.host}:{args.port}                         ║
║   Device: {app.config['DEVICE']:<10}                                    ║
║   Debug Mode: {'ON (auto-reload enabled)' if debug_mode else 'OFF':<36} ║
║   Log Level: {args.log_level:<10}                                  ║
║   Started: {startup_time}                           ║
║                                                              ║
║   Press Ctrl+C to stop                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    logger.info("=" * 60)
    logger.info(f"Flask server starting")
    logger.info(f"URL: http://{args.host}:{args.port}")
    logger.info(f"Debug: {debug_mode}, Reload: {debug_mode}")
    logger.info("=" * 60)
    
    # Use reloader in debug mode for auto-reload on file changes
    app.run(
        host=args.host,
        port=args.port,
        debug=debug_mode,
        use_reloader=debug_mode,  # Auto-reload on code changes
        threaded=True,
    )


if __name__ == '__main__':
    main()
