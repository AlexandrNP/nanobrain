#!/usr/bin/env python3
"""
NanoBrain Web Interface Demo Server

A simple demo server to test the NanoBrain web interface.

Usage:
    python demo_server.py                    # Run with default settings
    python demo_server.py --port 8080        # Run on custom port
    python demo_server.py --config config.yml # Run with custom config
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from nanobrain.library.interfaces.web.web_interface import WebInterface
    from nanobrain.library.interfaces.web.config.web_interface_config import WebInterfaceConfig
    print("✅ Successfully imported NanoBrain web interface")
except ImportError as e:
    print(f"❌ Failed to import web interface: {e}")
    print("   Make sure you're running from the correct directory and dependencies are installed")
    sys.exit(1)


async def run_demo_server(
    port: int = 8000, 
    host: str = "0.0.0.0",
    config_file: str = None
):
    """
    Run the demo server.
    
    Args:
        port: Port to run on
        host: Host to bind to
        config_file: Optional configuration file path
    """
    print("🧠 NanoBrain Web Interface Demo Server")
    print("=" * 50)
    
    try:
        # Create web interface using mandatory from_config pattern
        if config_file:
            print(f"📁 Loading configuration from: {config_file}")
            # Note: from_config_file method may need to be updated to use from_config internally
            interface = WebInterface.from_config_file(config_file)
        else:
            print("⚙️  Using default configuration")
            config = WebInterfaceConfig()
            if hasattr(config, 'server'):
                config.server.host = host
                config.server.port = port
            else:
                # Set attributes directly if server sub-config doesn't exist
                config.host = host
                config.port = port
            
            # Use mandatory from_config pattern
            interface = WebInterface.from_config(config)
        
        print(f"🚀 Starting server on {host}:{port}")
        print(f"📖 API documentation will be available at: http://{host}:{port}/docs")
        print(f"🔍 Health check available at: http://{host}:{port}/api/v1/health")
        print(f"💬 Chat endpoint: http://{host}:{port}/api/v1/chat/")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Start the server (this will block)
        await interface.start_server()
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'interface' in locals():
            await interface.shutdown()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="NanoBrain Web Interface Demo Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_server.py                           # Default: localhost:8000
  python demo_server.py --port 8080               # Custom port
  python demo_server.py --host 0.0.0.0            # Bind to all interfaces
  python demo_server.py --config my_config.yml    # Custom configuration

The server provides a REST API for NanoBrain chat workflows:
  POST /api/v1/chat/           - Send chat messages
  GET  /api/v1/health          - Health check
  GET  /api/v1/status          - Detailed status
  GET  /docs                   - Interactive API documentation
        """
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    # Validate config file if provided
    if args.config and not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Run the server
    try:
        asyncio.run(run_demo_server(
            port=args.port,
            host=args.host,
            config_file=args.config
        ))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 