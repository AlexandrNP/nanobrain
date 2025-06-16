# NanoBrain Web Interface

## Overview

The NanoBrain Web Interface provides a REST API for accessing NanoBrain chat workflows via HTTP. It's built using FastAPI and provides a robust, scalable interface for integrating NanoBrain capabilities into web applications, mobile apps, and other systems.

## Architecture

The web interface consists of several key components:

- **WebInterface**: Main orchestrator class that manages the FastAPI application
- **Configuration**: YAML-based configuration system for all settings
- **Request/Response Models**: Pydantic models for API validation
- **Routers**: FastAPI routers for different endpoint groups
- **Middleware**: CORS, logging, and error handling
- **ChatWorkflow Integration**: Direct integration with existing chat workflows

## Quick Start

### Basic Usage

```python
import asyncio
from nanobrain.library.interfaces.web import WebInterface

async def main():
    # Create web interface with default settings
    interface = WebInterface.create_default()
    
    # Start the server
    await interface.start_server()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Configuration File

```python
from nanobrain.library.interfaces.web import WebInterface

# Create from configuration file
interface = WebInterface.from_config_file("config/web_interface.yml")
await interface.start_server()
```

### Configuration

The web interface uses YAML configuration files. Here's the default structure:

```yaml
web_interface:
  name: "nanobrain_web_interface"
  version: "1.0.0"
  description: "NanoBrain Web Interface for Chat Workflows"
  
  server:
    host: "0.0.0.0"
    port: 8000
    workers: 1
    reload: false
    
  api:
    prefix: "/api/v1"
    title: "NanoBrain Chat API"
    description: "REST API for NanoBrain Chat System"
    
  cors:
    allow_origins: ["*"]
    allow_methods: ["GET", "POST", "OPTIONS"]
    allow_headers: ["*"]
    
  chat:
    default_temperature: 0.7
    default_max_tokens: 2000
    default_use_rag: false
    
  logging:
    enable_request_logging: true
    enable_response_logging: true
    log_level: "INFO"
```

## API Endpoints

### Chat Endpoints

#### POST /api/v1/chat/

Process a chat request.

**Request Body:**
```json
{
  "query": "Hello, how can you help me?",
  "options": {
    "temperature": 0.7,
    "max_tokens": 2000,
    "use_rag": false,
    "conversation_id": null
  },
  "user_id": "user_123"
}
```

**Response:**
```json
{
  "response": "Hello! I'm here to help you with any questions...",
  "status": "success",
  "conversation_id": "conv_123e4567-e89b-12d3-a456-426614174000",
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
  "metadata": {
    "processing_time_ms": 1250.5,
    "token_count": 25,
    "model_used": "gpt-3.5-turbo",
    "rag_enabled": false,
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "warnings": []
}
```

#### GET /api/v1/chat/conversations/{conversation_id}

Get conversation history.

**Parameters:**
- `conversation_id`: ID of the conversation
- `limit`: Maximum number of messages (default: 50)

#### DELETE /api/v1/chat/conversations/{conversation_id}

Delete a conversation.

### Health Endpoints

#### GET /api/v1/health

Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "chat_workflow": "healthy",
    "logging": "healthy",
    "configuration": "healthy"
  }
}
```

#### GET /api/v1/status

Detailed status information.

#### GET /api/v1/ping

Simple ping endpoint.

## Integration Examples

### Python Client

```python
import aiohttp
import asyncio

async def chat_example():
    async with aiohttp.ClientSession() as session:
        # Send chat request
        async with session.post(
            "http://localhost:8000/api/v1/chat/",
            json={
                "query": "What is machine learning?",
                "options": {
                    "temperature": 0.8,
                    "max_tokens": 1000
                }
            }
        ) as response:
            result = await response.json()
            print(f"Response: {result['response']}")

asyncio.run(chat_example())
```

### cURL Examples

```bash
# Basic chat request
curl -X POST "http://localhost:8000/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Hello, world!",
    "options": {
      "temperature": 0.7
    }
  }'

# Health check
curl "http://localhost:8000/api/v1/health"

# Get status
curl "http://localhost:8000/api/v1/status"
```

### JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

async function sendChatRequest(query) {
  const response = await fetch('http://localhost:8000/api/v1/chat/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: query,
      options: {
        temperature: 0.7,
        max_tokens: 2000
      }
    })
  });
  
  const result = await response.json();
  return result.response;
}

// Usage
sendChatRequest("Explain quantum computing")
  .then(response => console.log(response))
  .catch(error => console.error(error));
```

## Advanced Configuration

### Custom Chat Options

```yaml
chat:
  default_temperature: 0.7
  default_max_tokens: 2000
  default_use_rag: true
  enable_streaming: false
  conversation_timeout_seconds: 3600
  max_conversation_length: 100
```

### Security Settings

```yaml
security:
  enable_rate_limiting: true
  rate_limit_per_minute: 60
  enable_auth: false
  cors_enabled: true
```

### Logging Configuration

```yaml
logging:
  enable_request_logging: true
  enable_response_logging: true
  log_level: "INFO"
  log_requests_body: false
  log_responses_body: false
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY nanobrain/ ./nanobrain/
COPY config/ ./config/

EXPOSE 8000

CMD ["python", "-m", "nanobrain.library.interfaces.web.web_interface"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  nanobrain-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
```

### Production Deployment

For production deployment, consider using:

- **Uvicorn with multiple workers**
- **Reverse proxy (nginx)**
- **Load balancer**
- **Container orchestration (Kubernetes)**

```bash
# Multiple workers
uvicorn nanobrain.library.interfaces.web.web_interface:create_app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --access-log
```

## Monitoring and Metrics

The web interface provides several monitoring endpoints:

- `/api/v1/health` - Component health status
- `/api/v1/status` - Detailed status and metrics
- `/api/v1/ping` - Simple connectivity check

All requests are logged using the NanoBrain logging system with structured logging for monitoring and debugging.

## Error Handling

The API provides structured error responses:

```json
{
  "error": "validation_error",
  "message": "The query field is required and cannot be empty",
  "details": {
    "field": "query",
    "rejected_value": ""
  },
  "request_id": "req_123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad Request (validation errors)
- `500` - Internal Server Error
- `422` - Unprocessable Entity (invalid JSON)

## Integration with NanoBrain Workflows

The web interface seamlessly integrates with existing NanoBrain chat workflows:

1. **Direct Integration**: Uses the ChatWorkflow class directly
2. **Configuration Preservation**: Maintains all workflow configuration options
3. **Logging Integration**: Uses NanoBrain's logging system
4. **Error Handling**: Preserves workflow error handling patterns

## Best Practices

1. **Configuration Management**: Use YAML files for configuration
2. **Error Handling**: Always handle and log errors appropriately
3. **Monitoring**: Implement health checks and monitoring
4. **Security**: Configure CORS and authentication as needed
5. **Performance**: Use appropriate server settings for your load
6. **Logging**: Configure logging levels appropriately for production

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the port in configuration
2. **CORS Issues**: Configure CORS settings properly
3. **Import Errors**: Ensure NanoBrain is properly installed
4. **Workflow Initialization**: Check ChatWorkflow configuration

### Debug Mode

Enable debug mode for development:

```yaml
server:
  reload: true
logging:
  log_level: "DEBUG"
  log_requests_body: true
  log_responses_body: true
```

## Contributing

When extending the web interface:

1. Follow NanoBrain architecture patterns
2. Add proper error handling
3. Include comprehensive logging
4. Update documentation
5. Add tests for new functionality

## API Documentation

When running the server, interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 