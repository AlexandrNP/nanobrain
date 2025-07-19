#!/usr/bin/env python3
"""
NanoBrain Chatbot Server

Enhanced chatbot server that integrates with NanoBrain's standardized deployment system.
Supports event-driven workflow communication and configurable deployment modes.

Usage:
    python chatbot_server.py --mode development
    python chatbot_server.py --mode production --config chatbot_production.yml
    python chatbot_server.py --port 8080 --host 0.0.0.0
"""

import asyncio
import argparse
import sys
import os
import signal
from pathlib import Path
from typing import Optional, Dict, Any

# Add nanobrain to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from nanobrain.library.interfaces.web.web_interface import WebInterface
    from nanobrain.library.interfaces.web.config.web_interface_config import WebInterfaceConfig
    from nanobrain.core.logging_system import get_logger
    print("✅ Successfully imported NanoBrain web interface components")
except ImportError as e:
    print(f"❌ Failed to import NanoBrain components: {e}")
    print("   Make sure you're running from the correct directory and NanoBrain is installed")
    sys.exit(1)


class ChatbotServer:
    """
    Enhanced chatbot server with standardized deployment support.
    
    Features:
    - Configuration-driven deployment
    - Event-driven workflow integration
    - Development and production modes
    - Automatic workflow discovery and integration
    """
    
    def __init__(self):
        self.logger = get_logger("ChatbotServer")
        self.web_interface: Optional[WebInterface] = None
        self.workflow = None
    
    async def start(self, 
                   mode: str = "development",
                   config_file: Optional[str] = None,
                   port: Optional[int] = None,
                   host: Optional[str] = None,
                   workflow_config: Optional[str] = None) -> None:
        """Start the chatbot server with specified configuration"""
        
        try:
            self.logger.info(f"🚀 Starting NanoBrain Chatbot Server in {mode} mode")
            
            # Load configuration
            web_config = self._load_configuration(mode, config_file)
            
            # Override configuration if specified
            if port:
                web_config.server.port = port
            if host:
                web_config.server.host = host
            
            # Initialize workflow if specified
            if workflow_config or hasattr(web_config, 'workflow_integration'):
                await self._initialize_workflow(web_config, workflow_config)
            
            # Create web interface using from_config pattern
            self.web_interface = WebInterface.from_config(
                web_config, 
                chat_workflow=self.workflow
            )
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Display startup information
            self._display_startup_info(web_config, mode)
            
            # Start server
            await self.web_interface.start_server()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start chatbot server: {e}")
            raise
    
    def _load_configuration(self, mode: str, config_file: Optional[str]) -> WebInterfaceConfig:
        """Load web interface configuration"""
        
        if config_file:
            if not Path(config_file).exists():
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
            self.logger.info(f"📁 Loading configuration from: {config_file}")
            return WebInterfaceConfig.from_yaml(config_file)
        
        # Use default configuration based on mode
        config_files = {
            "development": script_dir / "config" / "chatbot_development.yml",
            "production": script_dir / "config" / "chatbot_production.yml",
            "testing": script_dir / "config" / "web_interface_config.yml"  # Fallback
        }
        
        default_config = config_files.get(mode)
        if default_config and default_config.exists():
            self.logger.info(f"📁 Loading default {mode} configuration: {default_config}")
            return WebInterfaceConfig.from_yaml(str(default_config))
        
        # Fallback to default configuration
        self.logger.warning(f"⚠️  No configuration found for mode '{mode}', using defaults")
        return WebInterfaceConfig()
    
    async def _initialize_workflow(self, web_config: WebInterfaceConfig, 
                                  workflow_config: Optional[str]) -> None:
        """Initialize workflow for event-driven communication"""
        
        try:
            # Determine workflow configuration
            if workflow_config:
                config_path = workflow_config
            elif hasattr(web_config, 'workflow_integration'):
                integration = web_config.workflow_integration
                config_path = getattr(integration, 'workflow_config', None)
            else:
                self.logger.info("ℹ️  No workflow configuration specified")
                return
            
            if not config_path:
                self.logger.info("ℹ️  No workflow configuration path found")
                return
            
            self.logger.info(f"🔗 Initializing workflow from: {config_path}")
            
            # Try to load chatbot viral workflow
            try:
                from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import ChatbotViralWorkflow
                from nanobrain.core.config.component_factory import load_config_file
                
                # Load workflow configuration
                if Path(config_path).exists():
                    workflow_config_dict = load_config_file(config_path)
                    self.workflow = ChatbotViralWorkflow.from_config(workflow_config_dict)
                    await self.workflow.initialize()
                    self.logger.info("✅ Workflow initialized successfully")
                else:
                    self.logger.warning(f"⚠️  Workflow config not found: {config_path}")
                    
            except ImportError as e:
                self.logger.warning(f"⚠️  Workflow not available: {e}")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize workflow: {e}")
                # Continue without workflow
                
        except Exception as e:
            self.logger.error(f"❌ Workflow initialization error: {e}")
            # Continue without workflow
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, shutting down...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _display_startup_info(self, config: WebInterfaceConfig, mode: str) -> None:
        """Display server startup information"""
        
        host = config.server.host
        port = config.server.port
        
        print(f"\n🧠 NanoBrain Chatbot Server")
        print("=" * 50)
        print(f"📋 Mode:           {mode}")
        print(f"🌐 Server:         http://{host}:{port}")
        print(f"🔧 API:            http://{host}:{port}{config.api.prefix}")
        print(f"💚 Health:         http://{host}:{port}{config.api.prefix}/health")
        
        # Show docs URL in development
        if mode == "development" and config.api.docs_url:
            print(f"📖 Docs:           http://{host}:{port}{config.api.docs_url}")
        
        # Show WebSocket info if enabled
        if hasattr(config, 'websocket') and config.websocket.enable_websocket:
            ws_protocol = "wss" if hasattr(config.security, 'enable_auth') and config.security.enable_auth else "ws"
            ws_host = host if host != "0.0.0.0" else "localhost"
            print(f"🔌 WebSocket:      {ws_protocol}://{ws_host}:{port}{config.api.prefix}/ws")
        
        # Show workflow status
        if self.workflow:
            print(f"🧬 Workflow:       ✅ ChatbotViralWorkflow loaded")
        else:
            print(f"🧬 Workflow:       ⚠️  No workflow loaded")
        
        print("=" * 50)
        print(f"🎯 Features:")
        print(f"   • 💬 Intelligent viral protein analysis chat")
        print(f"   • 🧬 Alphavirus specialization")
        print(f"   • 📊 Real-time health monitoring")
        
        if mode == "development":
            print(f"   • 🔧 Hot reloading and debug features")
            print(f"   • 📖 Interactive API documentation")
        else:
            print(f"   • 🏭 Production optimizations")
            print(f"   • 🔒 Security features enabled")
        
        print(f"\nPress Ctrl+C to stop the server")
        print()
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the server"""
        
        try:
            self.logger.info("🔄 Shutting down chatbot server...")
            
            # Shutdown web interface
            if self.web_interface:
                await self.web_interface.shutdown()
            
            # Shutdown workflow
            if self.workflow:
                await self.workflow.shutdown()
            
            self.logger.info("✅ Chatbot server shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")


async def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="NanoBrain Chatbot Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Deployment Modes:
  development    - Local development with hot reloading and debug features
  production     - Production deployment with optimizations and security
  testing        - Testing configuration with mock features

Examples:
  python chatbot_server.py --mode development
  python chatbot_server.py --mode production --port 8080
  python chatbot_server.py --config custom_config.yml
  python chatbot_server.py --workflow viral_analysis_config.yml

Configuration Files:
  The server will automatically use configuration files based on mode:
  - development: config/chatbot_development.yml
  - production:  config/chatbot_production.yml
  
  You can override with --config to use a custom configuration file.
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["development", "production", "testing"],
        default="development",
        help="Deployment mode (default: development)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Custom configuration file path"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Override server port"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        help="Override server host"
    )
    
    parser.add_argument(
        "--workflow",
        type=str,
        help="Workflow configuration file"
    )
    
    args = parser.parse_args()
    
    # Create and start server
    server = ChatbotServer()
    
    try:
        await server.start(
            mode=args.mode,
            config_file=args.config,
            port=args.port,
            host=args.host,
            workflow_config=args.workflow
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 