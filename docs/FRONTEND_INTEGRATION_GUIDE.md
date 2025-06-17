# NANOBRAIN FRONTEND INTEGRATION GUIDE

**Version**: 1.0.0  
**Status**: ✅ **READY FOR FRONTEND INTEGRATION**  
**Last Updated**: December 2024

## 🎯 Overview

This guide provides everything frontend developers need to integrate with the NanoBrain web API. The backend provides a production-ready REST API with WebSocket support, optimized for modern frontend frameworks like React, Vue, Angular, and Svelte.

## 🚀 Quick Start

### Base API Configuration
```javascript
const API_CONFIG = {
  baseURL: 'http://localhost:8000/api/v1',
  websocketURL: 'ws://localhost:8000/api/v1/ws',
  timeout: 30000
};
```

### Basic Chat Integration (Fetch API)
```javascript
// Send a chat message
async function sendMessage(message, conversationId = null) {
  const response = await fetch(`${API_CONFIG.baseURL}/frontend/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      conversation_id: conversationId,
      options: {
        temperature: 0.7,
        enable_streaming: false
      }
    })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
}

// Usage
try {
  const result = await sendMessage("Hello, how can you help me?");
  console.log("Bot response:", result.message);
  console.log("Conversation ID:", result.conversation_id);
} catch (error) {
  console.error("Chat error:", error);
}
```

## 📡 API Endpoints

### Frontend-Optimized Endpoints

#### POST `/api/v1/frontend/chat`
**Simplified chat endpoint optimized for frontend frameworks**

```typescript
interface FrontendChatRequest {
  message: string;                    // User message (1-10000 chars)
  conversation_id?: string;           // Optional conversation ID
  options?: {                         // Optional configuration
    temperature?: number;             // 0.0-2.0 (default: 0.7)
    max_tokens?: number;              // Max response length
    enable_streaming?: boolean;       // Stream response via WebSocket
    [key: string]: any;               // Additional options
  };
}

interface FrontendChatResponse {
  message: string;                    // Bot response
  conversation_id: string;            // Conversation ID
  timestamp: number;                  // Unix timestamp
  processing_time_ms: number;         // Response time
  status: "success" | "error";        // Status indicator
  metadata?: {                        // Optional metadata
    error_type?: string;
    error_message?: string;
    [key: string]: any;
  };
}
```

#### GET `/api/v1/frontend/system/info`
**Get system configuration for frontend**

```typescript
interface SystemInfo {
  api_version: string;
  websocket_url: string;
  supported_features: string[];
  configuration: {
    max_message_length: number;
    streaming_enabled: boolean;
    websocket_enabled: boolean;
    cors_enabled: boolean;
  };
}
```

## 🔌 WebSocket Integration

### Basic WebSocket Client
```javascript
class NanoBrainWebSocket {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.messageHandlers = new Map();
  }
  
  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
    });
  }
  
  sendChat(message, conversationId = null) {
    const request = {
      type: 'chat_request',
      data: {
        query: message,
        conversation_id: conversationId
      }
    };
    this.ws.send(JSON.stringify(request));
  }
  
  on(event, handler) {
    if (!this.messageHandlers.has(event)) {
      this.messageHandlers.set(event, []);
    }
    this.messageHandlers.get(event).push(handler);
  }
  
  handleMessage(message) {
    const handlers = this.messageHandlers.get(message.type) || [];
    handlers.forEach(handler => handler(message.data));
  }
}

// Usage
const ws = new NanoBrainWebSocket('ws://localhost:8000/api/v1/ws');

ws.on('chat_response', (data) => {
  console.log('Response:', data.response);
});

await ws.connect();
ws.sendChat("Hello!");
```

## 🛠 Error Handling

### HTTP Error Codes
- `400` - Bad Request (validation errors)
- `429` - Too Many Requests (rate limiting)
- `500` - Internal Server Error
- `503` - Service Unavailable

### Robust API Calls
```javascript
async function robustApiCall(endpoint, options) {
  const maxRetries = 3;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(endpoint, options);
      
      if (response.ok) {
        return await response.json();
      }
      
      if (response.status === 429) {
        // Rate limited - wait and retry
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        continue;
      }
      
      throw new Error(`HTTP ${response.status}`);
      
    } catch (error) {
      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, 2000 * attempt));
        continue;
      }
      throw error;
    }
  }
}
```

## 📚 Testing Endpoints

```bash
# Health check
curl http://localhost:8000/api/v1/health

# System information
curl http://localhost:8000/api/v1/frontend/system/info

# Frontend configuration
curl http://localhost:8000/api/v1/frontend/config/frontend
```

## ✅ Ready for Integration!

The NanoBrain web API is production-ready with:
- ✅ RESTful API with optimized frontend endpoints
- ✅ Real-time WebSocket communication
- ✅ Comprehensive error handling
- ✅ CORS configuration
- ✅ Health monitoring
- ✅ Complete documentation

**Start building your frontend application with confidence!** 🚀
