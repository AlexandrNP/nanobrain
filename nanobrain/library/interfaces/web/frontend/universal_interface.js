/**
 * Universal NanoBrain Interface for Frontend
 * Dynamic frontend adaptation system for any NanoBrain workflow with natural language input.
 * 
 * Author: NanoBrain Development Team
 * Date: January 2025
 * Version: 1.0.0
 */

import { DynamicComponentSystem } from './dynamic_components.js';
import { UniversalRequestBuilder } from './request_builder.js';
import { UniversalResponseParser } from './response_parser.js';

/**
 * Configuration for Universal NanoBrain Interface
 */
class UniversalInterfaceConfig {
    constructor(config = {}) {
        // Server configuration
        this.server = {
            baseUrl: config.server?.baseUrl || 'http://localhost:5001',
            apiPrefix: config.server?.apiPrefix || '/api',
            timeout: config.server?.timeout || 30000,
            retryAttempts: config.server?.retryAttempts || 3
        };

        // Interface behavior configuration
        this.interface = {
            enableAutoAdaptation: config.interface?.enableAutoAdaptation ?? true,
            enableStreamingSupport: config.interface?.enableStreamingSupport ?? true,
            enableProgressTracking: config.interface?.enableProgressTracking ?? true,
            adaptationStrategy: config.interface?.adaptationStrategy || 'automatic'
        };

        // Component rendering configuration
        this.rendering = {
            theme: config.rendering?.theme || 'default',
            responsiveDesign: config.rendering?.responsiveDesign ?? true,
            animationDuration: config.rendering?.animationDuration || 300,
            maxResponseHeight: config.rendering?.maxResponseHeight || '500px'
        };

        // Request handling configuration
        this.requests = {
            validateInput: config.requests?.validateInput ?? true,
            includeMetadata: config.requests?.includeMetadata ?? true,
            enableCaching: config.requests?.enableCaching ?? true,
            cacheExpiry: config.requests?.cacheExpiry || 300000 // 5 minutes
        };

        // Error handling configuration
        this.errorHandling = {
            showDetailedErrors: config.errorHandling?.showDetailedErrors ?? false,
            enableRetry: config.errorHandling?.enableRetry ?? true,
            fallbackMessage: config.errorHandling?.fallbackMessage ||
                'I apologize, but I encountered an issue processing your request.'
        };
    }
}

/**
 * Universal NanoBrain Interface Class
 * Provides dynamic frontend adaptation for any NanoBrain workflow
 */
export class UniversalNanoBrainInterface {
    constructor(config = {}) {
        this.config = new UniversalInterfaceConfig(config);
        this.componentSystem = new DynamicComponentSystem(this.config);
        this.requestBuilder = new UniversalRequestBuilder(this.config);
        this.responseParser = new UniversalResponseParser(this.config);

        // State management
        this.state = {
            isInitialized: false,
            currentWorkflow: null,
            availableWorkflows: [],
            conversationHistory: [],
            isProcessing: false,
            lastError: null,
            streamingSessions: new Map()
        };

        // Event handlers
        this.eventHandlers = new Map();

        // Request cache
        this.requestCache = new Map();

        // Initialize interface
        this.initialize();
    }

    /**
     * Initialize the universal interface
     */
    async initialize() {
        try {
            console.log('🚀 Initializing Universal NanoBrain Interface');

            // Discover available workflows
            await this.discoverWorkflows();

            // Setup component system
            await this.componentSystem.initialize();

            // Setup event listeners
            this.setupEventHandlers();

            // Validate server connection
            await this.validateServerConnection();

            this.state.isInitialized = true;
            this.emit('initialized', { success: true });

            console.log('✅ Universal NanoBrain Interface initialized successfully');

        } catch (error) {
            console.error('❌ Interface initialization failed:', error);
            this.state.lastError = error;
            this.emit('initialized', { success: false, error });
            throw error;
        }
    }

    /**
     * Discover available workflows from the server
     */
    async discoverWorkflows() {
        try {
            const response = await this.makeRequest('/workflows/capabilities', {
                method: 'GET'
            });

            if (response.capabilities) {
                this.state.availableWorkflows = response.capabilities;
                console.log(`✅ Discovered ${response.capabilities.length} workflows`);
            }

        } catch (error) {
            console.warn('⚠️ Workflow discovery failed:', error);
            // Continue with empty workflow list - server might not be available yet
            this.state.availableWorkflows = [];
        }
    }

    /**
     * Validate server connection
     */
    async validateServerConnection() {
        try {
            await this.makeRequest('/health', { method: 'GET' });
            console.log('✅ Server connection validated');
        } catch (error) {
            console.warn('⚠️ Server connection validation failed:', error);
            throw new Error('Unable to connect to NanoBrain server');
        }
    }

    /**
     * Setup event handlers
     */
    setupEventHandlers() {
        // Handle component events
        this.componentSystem.on('componentMounted', (data) => {
            this.emit('componentMounted', data);
        });

        this.componentSystem.on('componentError', (data) => {
            this.handleComponentError(data);
        });

        // Handle response parsing events
        this.responseParser.on('parseComplete', (data) => {
            this.emit('responseParseComplete', data);
        });
    }

    /**
     * Process a natural language request
     */
    async processRequest(query, options = {}) {
        try {
            if (!query || typeof query !== 'string') {
                throw new Error('Invalid query: must be a non-empty string');
            }

            this.state.isProcessing = true;
            this.state.lastError = null;
            this.emit('processingStarted', { query, options });

            // Build universal request
            const request = await this.requestBuilder.buildChatRequest(query, options);

            // Check cache if enabled
            if (this.config.requests.enableCaching) {
                const cachedResponse = this.getCachedResponse(request);
                if (cachedResponse) {
                    console.log('✅ Using cached response');
                    return await this.handleResponse(cachedResponse, request);
                }
            }

            // Send request to universal chat endpoint
            const response = await this.makeRequest('/universal-chat', {
                method: 'POST',
                body: JSON.stringify(request)
            });

            // Cache response if successful
            if (this.config.requests.enableCaching && response.success) {
                this.cacheResponse(request, response);
            }

            // Handle response
            return await this.handleResponse(response, request);

        } catch (error) {
            console.error('❌ Request processing failed:', error);
            this.state.lastError = error;
            this.emit('processingError', { error, query, options });
            return this.createErrorResponse(error.message);

        } finally {
            this.state.isProcessing = false;
            this.emit('processingComplete');
        }
    }

    /**
     * Handle server response
     */
    async handleResponse(response, originalRequest) {
        try {
            // Parse response using universal parser
            const parsedResponse = await this.responseParser.parseResponse(response);

            // Add to conversation history
            this.state.conversationHistory.push({
                request: originalRequest,
                response: parsedResponse,
                timestamp: new Date().toISOString()
            });

            // Emit response event
            this.emit('responseReceived', {
                parsed: parsedResponse,
                original: response,
                request: originalRequest
            });

            // Adapt interface based on response if enabled
            if (this.config.interface.enableAutoAdaptation) {
                await this.adaptInterface(parsedResponse);
            }

            return parsedResponse;

        } catch (error) {
            console.error('❌ Response handling failed:', error);
            this.emit('responseError', { error, response, originalRequest });
            return this.createErrorResponse('Failed to process server response');
        }
    }

    /**
     * Adapt interface based on response characteristics
     */
    async adaptInterface(parsedResponse) {
        try {
            const adaptationConfig = {
                responseType: parsedResponse.response_format,
                dataStructure: parsedResponse.data,
                frontendHints: parsedResponse.frontend_hints || {},
                workflowType: parsedResponse.metadata?.workflow_type
            };

            await this.componentSystem.adaptComponents(adaptationConfig);
            this.emit('interfaceAdapted', adaptationConfig);

        } catch (error) {
            console.warn('⚠️ Interface adaptation failed:', error);
            // Continue without adaptation - not critical
        }
    }

    /**
     * Create error response in standard format
     */
    createErrorResponse(errorMessage) {
        return {
            success: false,
            message: this.config.errorHandling.fallbackMessage,
            error_message: errorMessage,
            response_format: 'error',
            data: { error: true },
            metadata: {
                error_type: 'interface_error',
                timestamp: new Date().toISOString()
            }
        };
    }

    /**
     * Make HTTP request to server
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.config.server.baseUrl}${this.config.server.apiPrefix}${endpoint}`;

        const requestOptions = {
            method: options.method || 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.server.timeout);
        requestOptions.signal = controller.signal;

        try {
            const response = await fetch(url, requestOptions);
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();

        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new Error('Request timeout');
            }

            throw error;
        }
    }

    /**
     * Cache response for future use
     */
    cacheResponse(request, response) {
        try {
            const cacheKey = this.generateCacheKey(request);
            const cacheEntry = {
                response,
                timestamp: Date.now(),
                expiry: Date.now() + this.config.requests.cacheExpiry
            };

            this.requestCache.set(cacheKey, cacheEntry);

            // Cleanup expired entries
            this.cleanupCache();

        } catch (error) {
            console.warn('⚠️ Response caching failed:', error);
        }
    }

    /**
     * Get cached response if available and valid
     */
    getCachedResponse(request) {
        try {
            const cacheKey = this.generateCacheKey(request);
            const cacheEntry = this.requestCache.get(cacheKey);

            if (cacheEntry && cacheEntry.expiry > Date.now()) {
                return cacheEntry.response;
            }

            return null;

        } catch (error) {
            console.warn('⚠️ Cache retrieval failed:', error);
            return null;
        }
    }

    /**
     * Generate cache key for request
     */
    generateCacheKey(request) {
        // Simple cache key based on query and options
        const keyData = {
            query: request.query,
            options: request.options
        };
        return btoa(JSON.stringify(keyData));
    }

    /**
     * Cleanup expired cache entries
     */
    cleanupCache() {
        const now = Date.now();
        for (const [key, entry] of this.requestCache.entries()) {
            if (entry.expiry <= now) {
                this.requestCache.delete(key);
            }
        }
    }

    /**
     * Handle component errors
     */
    handleComponentError(errorData) {
        console.error('❌ Component error:', errorData);
        this.state.lastError = errorData.error;
        this.emit('componentError', errorData);
    }

    /**
     * Get interface state
     */
    getState() {
        return { ...this.state };
    }

    /**
     * Get conversation history
     */
    getConversationHistory() {
        return [...this.state.conversationHistory];
    }

    /**
     * Clear conversation history
     */
    clearConversationHistory() {
        this.state.conversationHistory = [];
        this.emit('conversationCleared');
    }

    /**
     * Get available workflows
     */
    getAvailableWorkflows() {
        return [...this.state.availableWorkflows];
    }

    /**
     * Set current workflow (for context)
     */
    setCurrentWorkflow(workflowId) {
        const workflow = this.state.availableWorkflows.find(w => w.workflow_id === workflowId);
        if (workflow) {
            this.state.currentWorkflow = workflow;
            this.emit('workflowChanged', workflow);
        }
    }

    /**
     * Event system methods
     */
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index !== -1) {
                handlers.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`❌ Event handler error for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Enable streaming for real-time responses
     */
    async enableStreaming(streamConfig = {}) {
        try {
            if (!this.config.interface.enableStreamingSupport) {
                throw new Error('Streaming support is disabled');
            }

            // Implementation would setup WebSocket or SSE connection
            console.log('🌊 Streaming enabled with config:', streamConfig);
            this.emit('streamingEnabled', streamConfig);

        } catch (error) {
            console.error('❌ Failed to enable streaming:', error);
            throw error;
        }
    }

    /**
     * Disable streaming
     */
    async disableStreaming() {
        try {
            // Implementation would close streaming connections
            this.state.streamingSessions.clear();
            console.log('🛑 Streaming disabled');
            this.emit('streamingDisabled');

        } catch (error) {
            console.error('❌ Failed to disable streaming:', error);
        }
    }

    /**
     * Get interface health status
     */
    getHealthStatus() {
        return {
            isInitialized: this.state.isInitialized,
            serverConnected: this.state.availableWorkflows.length > 0,
            isProcessing: this.state.isProcessing,
            hasErrors: this.state.lastError !== null,
            cacheSize: this.requestCache.size,
            conversationLength: this.state.conversationHistory.length,
            streamingSessions: this.state.streamingSessions.size
        };
    }

    /**
     * Destroy interface and cleanup resources
     */
    destroy() {
        try {
            // Clear event handlers
            this.eventHandlers.clear();

            // Clear cache
            this.requestCache.clear();

            // Clear streaming sessions
            this.state.streamingSessions.clear();

            // Destroy component system
            if (this.componentSystem) {
                this.componentSystem.destroy();
            }

            console.log('✅ Universal NanoBrain Interface destroyed');

        } catch (error) {
            console.error('❌ Interface destruction failed:', error);
        }
    }
} 