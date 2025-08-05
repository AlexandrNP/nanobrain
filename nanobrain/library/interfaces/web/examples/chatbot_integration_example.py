#!/usr/bin/env python3
"""
Chatbot Integration Example with Universal NanoBrain Interface
Shows how to replace custom server implementation with universal interface components.

This example demonstrates:
1. Replacing custom ChatRequest models with framework-compliant models
2. Using UniversalNanoBrainServer instead of custom FastAPI implementation
3. Leveraging universal workflow routing and response processing
4. Maintaining backward compatibility with existing frontend

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
from pathlib import Path

# Universal interface imports - replace custom implementations
from nanobrain.library.interfaces.web.servers import UniversalServerFactory
from nanobrain.library.interfaces.web.models import (
    ChatRequest, ChatResponse,  # Framework-compliant models
    WorkflowCapabilities, RequestAnalysis
)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalChatbotServer:
    """
    Universal chatbot server using NanoBrain framework components.
    Replaces custom server implementation with universal interface.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize universal chatbot server.
        
        Args:
            config_path: Path to universal server configuration file
        """
        self.config_path = config_path
        self.server = None
        self.is_running = False
        
    async def initialize(self):
        """Initialize the universal server with chatbot-specific configuration"""
        try:
            logger.info("🚀 Initializing Universal Chatbot Server")
            
            # Create universal server using factory
            self.server = UniversalServerFactory.create_fastapi_server(self.config_path)
            
            # Validate server configuration
            await self.validate_chatbot_requirements()
            
            logger.info("✅ Universal Chatbot Server initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Server initialization failed: {e}")
            raise
    
    async def validate_chatbot_requirements(self):
        """Validate that required workflows are available for chatbot functionality"""
        try:
            # Check that chatbot workflows are discovered
            registry = self.server.workflow_registry
            capabilities = await registry.get_all_capabilities()
            
            # Look for chatbot-compatible workflows
            chatbot_workflows = [
                cap for cap in capabilities 
                if cap.natural_language_input and 
                'conversational' in cap.categories or 'chat' in cap.categories
            ]
            
            if not chatbot_workflows:
                logger.warning("⚠️ No chatbot-compatible workflows found")
            else:
                logger.info(f"✅ Found {len(chatbot_workflows)} chatbot-compatible workflows")
                for workflow in chatbot_workflows:
                    logger.info(f"  - {workflow.workflow_id}: {workflow.description}")
            
        except Exception as e:
            logger.error(f"❌ Chatbot requirements validation failed: {e}")
    
    async def start(self, host: str = "0.0.0.0", port: int = 5001):
        """Start the universal chatbot server"""
        try:
            if not self.server:
                await self.initialize()
            
            logger.info(f"🚀 Starting Universal Chatbot Server on {host}:{port}")
            
            # Start universal server
            await self.server.start(host, port)
            self.is_running = True
            
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the universal chatbot server"""
        try:
            if self.server:
                await self.server.stop()
            self.is_running = False
            logger.info("✅ Universal Chatbot Server stopped")
            
        except Exception as e:
            logger.error(f"❌ Server shutdown failed: {e}")
    
    async def process_chat_request(self, request: ChatRequest) -> ChatResponse:
        """
        Process chat request using universal interface.
        This method shows how to migrate from custom request handling.
        """
        try:
            # The universal server handles request processing automatically
            # through its routing and response processing pipeline
            
            # For custom processing, you can access components directly:
            router = self.server.workflow_router
            response_processor = self.server.response_processor
            
            # Analyze request
            analyzer = self.server.request_analyzer
            analysis = await analyzer.analyze_request(request)
            
            # Route to appropriate workflow
            route = await router.route_request(request, analysis)
            
            # Execute workflow
            universal_response = await router.execute_route(route)
            
            # Process response
            standardized_response = await response_processor.standardize_response(universal_response)
            
            # Convert to ChatResponse format
            chat_response = ChatResponse(
                response=standardized_response.message,
                request_id=request.request_id,
                conversation_id=request.options.conversation_id,
                success=standardized_response.success,
                metadata=standardized_response.metadata,
                error_message=standardized_response.error_message
            )
            
            return chat_response
            
        except Exception as e:
            logger.error(f"❌ Chat request processing failed: {e}")
            return ChatResponse(
                response="I apologize, but I encountered an error processing your request.",
                request_id=request.request_id,
                conversation_id=request.options.conversation_id if request.options else None,
                success=False,
                error_message=str(e)
            )


def create_chatbot_config():
    """
    Create chatbot-specific configuration.
    This shows how to customize universal interface for chatbot use case.
    """
    return {
        "server_type": "fastapi",
        "host": "0.0.0.0", 
        "port": 5001,
        "debug": True,
        
        "fastapi_config": {
            "title": "NanoBrain Chatbot Server",
            "description": "Universal chatbot interface powered by NanoBrain framework",
            "version": "1.0.0",
            "docs_url": "/docs",
            "redoc_url": "/redoc"
        },
        
        "components": {
            "workflow_registry": {
                "class": "nanobrain.library.interfaces.web.routing.workflow_registry.WorkflowRegistry",
                "config": {
                    "auto_discovery": True,
                    "discovery_paths": [
                        "nanobrain.library.workflows.chatbot_viral_integration",
                        "nanobrain.library.workflows.viral_protein_analysis"
                    ],
                    "require_natural_language_input": True,
                    "minimum_compliance_score": 0.6  # Lower for chatbot flexibility
                }
            },
            
            "request_analyzer": {
                "class": "nanobrain.library.interfaces.web.analysis.request_analyzer.UniversalRequestAnalyzer",
                "config": {
                    "intent_classifier": {
                        "class": "nanobrain.library.interfaces.web.analysis.intent_classifier.IntentClassifier",
                        "config": {
                            "method": "hybrid",
                            "confidence_thresholds": {
                                "high_confidence": 0.7,  # Adjusted for conversational AI
                                "medium_confidence": 0.5,
                                "low_confidence": 0.2
                            }
                        }
                    }
                }
            },
            
            "workflow_router": {
                "class": "nanobrain.library.interfaces.web.routing.workflow_router.WorkflowRouter", 
                "config": {
                    "routing_strategy": "best_match",
                    "fallback_strategy": "general_conversation",
                    "routing_thresholds": {
                        "minimum_confidence": 0.2,  # Liberal for chatbot
                        "fallback_threshold": 0.1
                    },
                    "execution_config": {
                        "max_execution_time": 60.0,  # Faster for chat
                        "enable_caching": True
                    }
                }
            },
            
            "response_processor": {
                "class": "nanobrain.library.interfaces.web.processing.response_processor.UniversalResponseProcessor",
                "config": {
                    "standardization": True,
                    "streaming_support": True,  # Enable for real-time chat
                    "error_handling": {
                        "detailed_error_messages": True,  # Helpful for development
                        "fallback_message": "I'm sorry, I couldn't process that. Could you try rephrasing your question?"
                    }
                }
            }
        },
        
        "endpoints": {
            "chat": "/api/chat",  # Maintain existing endpoint for compatibility
            "universal_chat": "/api/universal-chat",  # New universal endpoint
            "capabilities": "/api/workflows/capabilities",
            "health": "/api/health"
        },
        
        "cors_config": {
            "allow_origins": ["http://localhost:3000"],  # Frontend URL
            "allow_methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["*"],
            "allow_credentials": True
        }
    }


async def main():
    """
    Main function demonstrating universal chatbot server setup.
    This replaces the custom server implementation in Chatbot/server/nanobrain_server.py
    """
    try:
        logger.info("🎯 Starting Universal Chatbot Integration Example")
        
        # Create configuration
        config = create_chatbot_config()
        
        # Initialize universal chatbot server
        chatbot_server = UniversalChatbotServer(config)
        await chatbot_server.initialize()
        
        # Start server
        await chatbot_server.start()
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down server...")
        if 'chatbot_server' in locals():
            await chatbot_server.stop()
    except Exception as e:
        logger.error(f"❌ Example execution failed: {e}")
        raise


class LegacyCompatibilityAdapter:
    """
    Adapter for maintaining compatibility with existing frontend code.
    Shows how to bridge between old custom models and new framework models.
    """
    
    @staticmethod
    def convert_legacy_request(legacy_request: dict) -> ChatRequest:
        """
        Convert legacy request format to framework-compliant ChatRequest.
        
        Legacy format: {"message": str, "conversation_id": str, "options": dict}
        Framework format: ChatRequest with query, options, request_id
        """
        try:
            # Extract data from legacy format
            message = legacy_request.get("message", "")
            conversation_id = legacy_request.get("conversation_id")
            legacy_options = legacy_request.get("options", {})
            
            # Build framework-compliant request
            from nanobrain.library.interfaces.web.models.request_models import ChatOptions
            
            options = ChatOptions(
                conversation_id=conversation_id,
                **legacy_options
            )
            
            chat_request = ChatRequest(
                query=message,  # Framework uses 'query' instead of 'message'
                options=options
            )
            
            return chat_request
            
        except Exception as e:
            logger.error(f"❌ Legacy request conversion failed: {e}")
            raise
    
    @staticmethod
    def convert_framework_response(framework_response: ChatResponse) -> dict:
        """
        Convert framework ChatResponse to legacy format for frontend compatibility.
        
        Framework format: ChatResponse object
        Legacy format: {"response": str, "success": bool, ...}
        """
        try:
            return {
                "response": framework_response.response,
                "success": framework_response.success,
                "conversation_id": framework_response.conversation_id,
                "request_id": framework_response.request_id,
                "metadata": framework_response.metadata,
                "error": framework_response.error_message
            }
            
        except Exception as e:
            logger.error(f"❌ Framework response conversion failed: {e}")
            raise


def create_integration_script():
    """
    Create integration script for updating existing Chatbot application.
    This shows the exact steps to migrate from custom to universal interface.
    """
    integration_steps = """
    # Integration Steps for Existing Chatbot Application
    
    ## 1. Update Server Implementation
    
    # OLD: Custom FastAPI server (Chatbot/server/nanobrain_server.py)
    # NEW: Universal server with framework components
    
    Replace:
    ```python
    # Old custom server
    from fastapi import FastAPI
    app = FastAPI()
    
    class ChatRequest(BaseModel):
        message: str  # Non-framework-compliant
    ```
    
    With:
    ```python
    # New universal server
    from nanobrain.library.interfaces.web.servers import UniversalServerFactory
    from nanobrain.library.interfaces.web.models import ChatRequest  # Framework-compliant
    
    server = UniversalServerFactory.create_fastapi_server(config_path)
    ```
    
    ## 2. Update Request/Response Models
    
    # OLD: Custom models with 'message' field
    # NEW: Framework models with 'query' field
    
    Use LegacyCompatibilityAdapter for gradual migration.
    
    ## 3. Update Frontend Configuration
    
    # Update Chatbot/frontend/src/config/nanobrain-config.js:
    ```javascript
    export const NANOBRAIN_CONFIG = {
        apiUrl: 'http://localhost:5001/api',
        endpoints: {
            chat: '/universal-chat',  // NEW universal endpoint
            capabilities: '/workflows/capabilities',
            health: '/health'
        }
    };
    ```
    
    ## 4. Leverage Universal Features
    
    # Enable workflow auto-discovery
    # Use intelligent request routing  
    # Benefit from universal response processing
    # Access streaming capabilities
    
    ## 5. Testing and Validation
    
    # Test with existing frontend
    # Validate workflow discovery
    # Verify response compatibility
    # Check streaming functionality
    """
    
    return integration_steps


if __name__ == "__main__":
    # Run the universal chatbot integration example
    asyncio.run(main()) 