#!/usr/bin/env python3
"""
NanoBrain Web Interface Demo

This demo shows how to:
1. Start the NanoBrain web interface server
2. Make API requests to the server
3. Handle responses and errors

Run this script to start a demo server and see examples of API usage.
"""

import asyncio
import aiohttp
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import NanoBrain web interface
try:
    from nanobrain.library.interfaces.web import WebInterface, WebInterfaceConfig
    from nanobrain.library.interfaces.web.config.web_interface_config import (
        ServerConfig, APIConfig, CORSConfig, ChatConfig, LoggingConfig
    )
    WEB_INTERFACE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import web interface: {e}")
    WEB_INTERFACE_AVAILABLE = False


class WebInterfaceDemo:
    """
    Demo class for the NanoBrain Web Interface.
    
    This class demonstrates how to start the web interface server
    and make various API requests to test functionality.
    """
    
    def __init__(self):
        self.server_task = None
        self.interface = None
        self.base_url = "http://localhost:8000"
        self.api_base = f"{self.base_url}/api/v1"
        
    async def create_interface(self) -> WebInterface:
        """Create a web interface with demo configuration."""
        print("🔧 Creating web interface with demo configuration...")
        
        # Create custom configuration for demo
        config = WebInterfaceConfig(
            name="nanobrain_web_demo",
            version="1.0.0",
            description="NanoBrain Web Interface Demo",
            
            # Server settings
            server=ServerConfig(
                host="127.0.0.1",  # Localhost only for demo
                port=8000,
                workers=1,
                reload=False,
                access_log=True
            ),
            
            # API settings
            api=APIConfig(
                prefix="/api/v1",
                title="NanoBrain Demo API",
                description="Demo REST API for NanoBrain Chat System",
                version="1.0.0"
            ),
            
            # CORS settings (permissive for demo)
            cors=CORSConfig(
                allow_origins=["*"],
                allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
                allow_headers=["*"],
                allow_credentials=False
            ),
            
            # Chat settings
            chat=ChatConfig(
                default_temperature=0.7,
                default_max_tokens=1000,
                default_use_rag=False,
                enable_streaming=False
            ),
            
            # Logging settings
            logging=LoggingConfig(
                enable_request_logging=True,
                enable_response_logging=True,
                log_level="INFO",
                log_requests_body=False,
                log_responses_body=False
            )
        )
        
        interface = WebInterface(config)
        await interface.initialize()
        
        print("✅ Web interface created and initialized")
        return interface
    
    async def start_server(self):
        """Start the web interface server in the background."""
        if not WEB_INTERFACE_AVAILABLE:
            print("❌ Web interface not available - skipping server start")
            return
        
        print("🚀 Starting web interface server...")
        
        self.interface = await self.create_interface()
        
        # Start server in background task
        self.server_task = asyncio.create_task(self.interface.start_server())
        
        # Wait a moment for server to start
        await asyncio.sleep(2)
        print(f"✅ Server started at {self.base_url}")
        print(f"📖 API documentation: {self.base_url}/docs")
        print(f"📊 ReDoc documentation: {self.base_url}/redoc")
    
    async def stop_server(self):
        """Stop the web interface server."""
        if self.server_task:
            print("🛑 Stopping web interface server...")
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        if self.interface:
            await self.interface.shutdown()
        
        print("✅ Server stopped")
    
    async def test_health_endpoints(self):
        """Test health check endpoints."""
        print("\n🔍 Testing health endpoints...")
        
        async with aiohttp.ClientSession() as session:
            # Test ping
            try:
                async with session.get(f"{self.api_base}/ping") as response:
                    result = await response.json()
                    print(f"✅ Ping: {result}")
            except Exception as e:
                print(f"❌ Ping failed: {e}")
            
            # Test health
            try:
                async with session.get(f"{self.api_base}/health") as response:
                    result = await response.json()
                    print(f"✅ Health: {result['status']} - Components: {result['components']}")
            except Exception as e:
                print(f"❌ Health check failed: {e}")
            
            # Test status
            try:
                async with session.get(f"{self.api_base}/status") as response:
                    result = await response.json()
                    print(f"✅ Status: {result['api_status']} - Uptime: {result['uptime_seconds']:.1f}s")
            except Exception as e:
                print(f"❌ Status check failed: {e}")
    
    async def test_chat_endpoint(self):
        """Test the main chat endpoint."""
        print("\n💬 Testing chat endpoint...")
        
        # Test cases
        test_queries = [
            {
                "query": "Hello, how are you today?",
                "options": {
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            },
            {
                "query": "What is machine learning?",
                "options": {
                    "temperature": 0.5,
                    "max_tokens": 1000,
                    "use_rag": False
                },
                "user_id": "demo_user"
            },
            {
                "query": "Explain quantum computing in simple terms",
                "options": {
                    "temperature": 0.8,
                    "max_tokens": 800
                }
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            for i, test_case in enumerate(test_queries, 1):
                print(f"\n📤 Test {i}: {test_case['query'][:50]}...")
                
                try:
                    async with session.post(
                        f"{self.api_base}/chat/",
                        json=test_case
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            print(f"✅ Response ({len(result['response'])} chars): {result['response'][:100]}...")
                            print(f"   📊 Metadata: {result['metadata']['processing_time_ms']:.1f}ms")
                            print(f"   🆔 Conversation: {result['conversation_id']}")
                        else:
                            error_text = await response.text()
                            print(f"❌ Error {response.status}: {error_text}")
                            
                except Exception as e:
                    print(f"❌ Request failed: {e}")
    
    async def test_error_handling(self):
        """Test error handling with invalid requests."""
        print("\n🚨 Testing error handling...")
        
        # Test cases that should produce errors
        error_test_cases = [
            {
                "name": "Empty query",
                "data": {"query": ""},
                "expected_status": 422
            },
            {
                "name": "Missing query",
                "data": {"options": {"temperature": 0.7}},
                "expected_status": 422
            },
            {
                "name": "Invalid temperature",
                "data": {
                    "query": "Test",
                    "options": {"temperature": 5.0}  # Too high
                },
                "expected_status": 422
            },
            {
                "name": "Invalid max_tokens",
                "data": {
                    "query": "Test", 
                    "options": {"max_tokens": -100}  # Negative
                },
                "expected_status": 422
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            for test_case in error_test_cases:
                print(f"\n🧪 Testing: {test_case['name']}")
                
                try:
                    async with session.post(
                        f"{self.api_base}/chat/",
                        json=test_case["data"]
                    ) as response:
                        
                        result = await response.json()
                        
                        if response.status == test_case["expected_status"]:
                            print(f"✅ Got expected error {response.status}: {result.get('message', 'No message')}")
                        else:
                            print(f"⚠️  Unexpected status {response.status}, expected {test_case['expected_status']}")
                            
                except Exception as e:
                    print(f"❌ Test failed: {e}")
    
    async def demonstrate_client_usage(self):
        """Demonstrate different ways to use the API from client code."""
        print("\n🔧 Client usage examples...")
        
        # Example 1: Simple request
        print("\n📝 Example 1: Simple chat request")
        await self._simple_chat_request()
        
        # Example 2: Request with options
        print("\n📝 Example 2: Request with custom options")
        await self._chat_with_options()
        
        # Example 3: Conversation flow
        print("\n📝 Example 3: Conversation flow")
        await self._conversation_flow()
    
    async def _simple_chat_request(self):
        """Simple chat request example."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base}/chat/",
                    json={"query": "Hello! Can you help me?"}
                ) as response:
                    result = await response.json()
                    print(f"   💬 Response: {result['response'][:100]}...")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    async def _chat_with_options(self):
        """Chat request with custom options."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.api_base}/chat/",
                    json={
                        "query": "Explain artificial intelligence",
                        "options": {
                            "temperature": 0.3,  # More focused
                            "max_tokens": 200,   # Shorter response
                            "use_rag": False
                        },
                        "user_id": "example_user"
                    }
                ) as response:
                    result = await response.json()
                    print(f"   💬 Response: {result['response'][:100]}...")
                    print(f"   ⏱️  Processing time: {result['metadata']['processing_time_ms']:.1f}ms")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    async def _conversation_flow(self):
        """Demonstrate a conversation flow."""
        conversation_id = None
        
        async with aiohttp.ClientSession() as session:
            # First message
            try:
                async with session.post(
                    f"{self.api_base}/chat/",
                    json={
                        "query": "Hi, I'm interested in learning about Python programming",
                        "options": {"temperature": 0.7}
                    }
                ) as response:
                    result = await response.json()
                    conversation_id = result['conversation_id']
                    print(f"   💬 First: {result['response'][:80]}...")
                    print(f"   🆔 Conversation ID: {conversation_id}")
            except Exception as e:
                print(f"   ❌ First message error: {e}")
                return
            
            # Follow-up message
            try:
                async with session.post(
                    f"{self.api_base}/chat/",
                    json={
                        "query": "What are the best practices for writing clean code?",
                        "options": {
                            "conversation_id": conversation_id,
                            "temperature": 0.6
                        }
                    }
                ) as response:
                    result = await response.json()
                    print(f"   💬 Follow-up: {result['response'][:80]}...")
            except Exception as e:
                print(f"   ❌ Follow-up error: {e}")
    
    async def show_api_documentation_info(self):
        """Show information about API documentation."""
        print("\n📚 API Documentation")
        print("=" * 50)
        print(f"🌐 Interactive API docs (Swagger): {self.base_url}/docs")
        print(f"📖 Alternative docs (ReDoc):      {self.base_url}/redoc")
        print("\nAPI Endpoints:")
        print(f"   POST {self.api_base}/chat/                     - Send chat message")
        print(f"   GET  {self.api_base}/chat/conversations/{{id}} - Get conversation")
        print(f"   GET  {self.api_base}/health                   - Health check")
        print(f"   GET  {self.api_base}/status                   - Detailed status")
        print(f"   GET  {self.api_base}/ping                     - Simple ping")
    
    async def run_full_demo(self):
        """Run the complete demo."""
        print("🧠 NanoBrain Web Interface Demo")
        print("=" * 50)
        
        if not WEB_INTERFACE_AVAILABLE:
            print("❌ Web interface components not available")
            print("   Make sure FastAPI and dependencies are installed")
            return
        
        try:
            # Start server
            await self.start_server()
            
            # Show API documentation info
            await self.show_api_documentation_info()
            
            # Run tests
            await self.test_health_endpoints()
            await self.test_chat_endpoint()
            await self.test_error_handling()
            await self.demonstrate_client_usage()
            
            print("\n🎉 Demo completed successfully!")
            print("\n💡 The server is still running. You can:")
            print(f"   • Open {self.base_url}/docs in your browser")
            print("   • Make your own API requests")
            print("   • Press Ctrl+C to stop the server")
            
            # Keep server running
            try:
                await asyncio.Event().wait()  # Wait indefinitely
            except KeyboardInterrupt:
                print("\n👋 Stopping demo...")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await self.stop_server()


# Standalone test functions
async def test_basic_functionality():
    """Test basic functionality without full demo."""
    print("🧪 Testing basic web interface functionality...")
    
    if not WEB_INTERFACE_AVAILABLE:
        print("❌ Web interface not available")
        return
    
    # Create interface
    interface = WebInterface.create_default()
    
    # Test initialization
    await interface.initialize()
    print("✅ Interface initialization successful")
    
    # Test status
    status = interface.get_status()
    print(f"✅ Status check successful: {status['name']} v{status['version']}")
    
    # Cleanup
    await interface.shutdown()
    print("✅ Cleanup successful")


# Main execution
async def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NanoBrain Web Interface Demo")
    parser.add_argument(
        "--basic", 
        action="store_true", 
        help="Run basic functionality test only"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the demo server on"
    )
    
    args = parser.parse_args()
    
    if args.basic:
        await test_basic_functionality()
    else:
        demo = WebInterfaceDemo()
        demo.base_url = f"http://localhost:{args.port}"
        demo.api_base = f"{demo.base_url}/api/v1"
        await demo.run_full_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 