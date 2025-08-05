/**
 * Universal Request Builder for NanoBrain Framework
 * Framework-compliant request building with validation and metadata generation.
 * 
 * Author: NanoBrain Development Team
 * Date: January 2025
 * Version: 1.0.0
 */

/**
 * Request validation schemas
 */
const ValidationSchemas = {
    ChatRequest: {
        required: ['query'],
        properties: {
            query: { type: 'string', minLength: 1, maxLength: 10000 },
            conversation_id: { type: 'string', pattern: '^[a-zA-Z0-9_-]+$' },
            options: { type: 'object' }
        }
    },

    ChatOptions: {
        properties: {
            conversation_id: { type: 'string' },
            user_id: { type: 'string' },
            context: { type: 'object' },
            preferences: { type: 'object' },
            metadata: { type: 'object' }
        }
    }
};

/**
 * Universal Request Builder Class
 * Builds framework-compliant requests with validation and metadata
 */
export class UniversalRequestBuilder {
    constructor(config = {}) {
        this.config = config;
        this.requestId = 0;
        this.defaultOptions = {
            conversation_id: null,
            user_id: 'anonymous',
            context: {},
            preferences: {
                response_format: 'auto',
                max_response_length: 5000,
                include_metadata: true
            },
            metadata: {}
        };

        console.log('🔧 Universal Request Builder initialized');
    }

    /**
     * Build a chat request following NanoBrain protocol
     */
    async buildChatRequest(message, customOptions = {}, userId = null) {
        try {
            if (!message || typeof message !== 'string') {
                throw new Error('Message must be a non-empty string');
            }

            // Generate unique request ID
            const requestId = this.generateRequestId();

            // Build options object
            const options = this.buildChatOptions(customOptions, userId);

            // Create base request structure
            const chatRequest = {
                query: message.trim(),
                options: options,
                request_id: requestId
            };

            // Add client metadata if configured
            if (this.config.requests?.includeMetadata) {
                chatRequest.client_metadata = this.generateClientMetadata();
            }

            // Validate request
            if (this.config.requests?.validateInput) {
                this.validateChatRequest(chatRequest);
            }

            console.log(`✅ Built chat request: ${requestId}`);
            return chatRequest;

        } catch (error) {
            console.error('❌ Chat request building failed:', error);
            throw error;
        }
    }

    /**
     * Build chat options object
     */
    buildChatOptions(customOptions = {}, userId = null) {
        const options = {
            ...this.defaultOptions,
            ...customOptions
        };

        // Set user ID if provided
        if (userId) {
            options.user_id = userId;
        }

        // Ensure conversation ID exists
        if (!options.conversation_id) {
            options.conversation_id = this.generateConversationId();
        }

        // Add timestamp
        options.timestamp = new Date().toISOString();

        // Add framework version
        options.framework_version = '1.0.0';

        return options;
    }

    /**
     * Generate unique request ID
     */
    generateRequestId() {
        this.requestId++;
        const timestamp = Date.now();
        const random = Math.random().toString(36).substr(2, 9);
        return `req_${timestamp}_${this.requestId}_${random}`;
    }

    /**
     * Generate conversation ID
     */
    generateConversationId() {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substr(2, 9);
        return `conv_${timestamp}_${random}`;
    }

    /**
     * Generate client metadata
     */
    generateClientMetadata() {
        return {
            user_agent: navigator.userAgent,
            platform: navigator.platform,
            language: navigator.language,
            screen_resolution: `${screen.width}x${screen.height}`,
            timestamp: new Date().toISOString(),
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            client_type: 'web_browser',
            framework_client: 'nanobrain-universal-interface',
            client_version: '1.0.0'
        };
    }

    /**
     * Validate chat request against schema
     */
    validateChatRequest(request) {
        try {
            // Validate required fields
            const schema = ValidationSchemas.ChatRequest;

            for (const field of schema.required) {
                if (!(field in request)) {
                    throw new Error(`Required field missing: ${field}`);
                }
            }

            // Validate query
            const query = request.query;
            if (typeof query !== 'string') {
                throw new Error('Query must be a string');
            }

            if (query.length < 1) {
                throw new Error('Query cannot be empty');
            }

            if (query.length > 10000) {
                throw new Error('Query exceeds maximum length (10000 characters)');
            }

            // Validate conversation ID format if present
            if (request.options?.conversation_id) {
                const convId = request.options.conversation_id;
                if (typeof convId !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(convId)) {
                    throw new Error('Invalid conversation ID format');
                }
            }

            console.log('✅ Request validation passed');

        } catch (error) {
            console.error('❌ Request validation failed:', error);
            throw error;
        }
    }

    /**
     * Build streaming request
     */
    async buildStreamingRequest(message, streamConfig = {}, customOptions = {}) {
        try {
            const baseRequest = await this.buildChatRequest(message, customOptions);

            // Add streaming-specific configuration
            baseRequest.streaming = {
                enabled: true,
                stream_type: streamConfig.stream_type || 'real_time_results',
                buffer_size: streamConfig.buffer_size || 8192,
                progress_updates: streamConfig.progress_updates ?? true,
                ...streamConfig
            };

            console.log('✅ Built streaming request');
            return baseRequest;

        } catch (error) {
            console.error('❌ Streaming request building failed:', error);
            throw error;
        }
    }

    /**
     * Build workflow-specific request
     */
    async buildWorkflowRequest(message, workflowId, workflowConfig = {}, customOptions = {}) {
        try {
            const baseRequest = await this.buildChatRequest(message, customOptions);

            // Add workflow-specific configuration
            baseRequest.workflow = {
                workflow_id: workflowId,
                workflow_config: workflowConfig,
                force_workflow: true
            };

            console.log(`✅ Built workflow-specific request for: ${workflowId}`);
            return baseRequest;

        } catch (error) {
            console.error('❌ Workflow request building failed:', error);
            throw error;
        }
    }

    /**
     * Build analysis request for request routing insights
     */
    async buildAnalysisRequest(message, customOptions = {}) {
        try {
            const baseRequest = await this.buildChatRequest(message, customOptions);

            // Add analysis-specific configuration
            baseRequest.analysis = {
                include_routing_analysis: true,
                include_intent_classification: true,
                include_domain_classification: true,
                include_workflow_recommendations: true
            };

            console.log('✅ Built analysis request');
            return baseRequest;

        } catch (error) {
            console.error('❌ Analysis request building failed:', error);
            throw error;
        }
    }

    /**
     * Build file upload request
     */
    async buildFileUploadRequest(files, message = '', customOptions = {}) {
        try {
            const baseRequest = await this.buildChatRequest(message || 'File upload request', customOptions);

            // Process files
            const processedFiles = await this.processFiles(files);

            // Add file-specific configuration
            baseRequest.files = processedFiles;
            baseRequest.upload = {
                file_count: processedFiles.length,
                total_size: processedFiles.reduce((sum, file) => sum + file.size, 0),
                file_types: [...new Set(processedFiles.map(file => file.type))]
            };

            console.log(`✅ Built file upload request with ${processedFiles.length} files`);
            return baseRequest;

        } catch (error) {
            console.error('❌ File upload request building failed:', error);
            throw error;
        }
    }

    /**
     * Process files for upload
     */
    async processFiles(files) {
        const processedFiles = [];

        for (const file of files) {
            try {
                const fileInfo = {
                    name: file.name,
                    size: file.size,
                    type: file.type,
                    last_modified: file.lastModified
                };

                // Read file content based on type
                if (file.type.startsWith('text/') || file.type === 'application/json') {
                    fileInfo.content = await this.readFileAsText(file);
                } else {
                    fileInfo.content = await this.readFileAsBase64(file);
                    fileInfo.encoding = 'base64';
                }

                processedFiles.push(fileInfo);

            } catch (error) {
                console.warn(`⚠️ Failed to process file ${file.name}:`, error);
                // Skip problematic files but continue with others
            }
        }

        return processedFiles;
    }

    /**
     * Read file as text
     */
    readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e.target.error);
            reader.readAsText(file);
        });
    }

    /**
     * Read file as base64
     */
    readFileAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                // Remove data URL prefix to get just the base64 content
                const base64 = e.target.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = (e) => reject(e.target.error);
            reader.readAsDataURL(file);
        });
    }

    /**
     * Build context-aware request with conversation history
     */
    async buildContextualRequest(message, conversationHistory = [], customOptions = {}) {
        try {
            const baseRequest = await this.buildChatRequest(message, customOptions);

            // Add conversation context
            baseRequest.context = {
                conversation_history: this.summarizeConversationHistory(conversationHistory),
                context_window: conversationHistory.length,
                has_context: conversationHistory.length > 0
            };

            console.log(`✅ Built contextual request with ${conversationHistory.length} history items`);
            return baseRequest;

        } catch (error) {
            console.error('❌ Contextual request building failed:', error);
            throw error;
        }
    }

    /**
     * Summarize conversation history for context
     */
    summarizeConversationHistory(history) {
        // Keep last 5 exchanges to avoid overwhelming the context
        const recentHistory = history.slice(-5);

        return recentHistory.map(exchange => ({
            query: exchange.request?.query || '',
            response: exchange.response?.message || '',
            timestamp: exchange.timestamp,
            success: exchange.response?.success ?? true
        }));
    }

    /**
     * Build batch request for multiple queries
     */
    async buildBatchRequest(messages, customOptions = {}) {
        try {
            if (!Array.isArray(messages) || messages.length === 0) {
                throw new Error('Messages must be a non-empty array');
            }

            const batchId = this.generateBatchId();
            const requests = [];

            for (let i = 0; i < messages.length; i++) {
                const message = messages[i];
                const request = await this.buildChatRequest(message, {
                    ...customOptions,
                    batch_info: {
                        batch_id: batchId,
                        item_index: i,
                        total_items: messages.length
                    }
                });
                requests.push(request);
            }

            const batchRequest = {
                batch_id: batchId,
                batch_type: 'chat_requests',
                requests: requests,
                batch_options: {
                    parallel_processing: customOptions.parallel_processing ?? true,
                    fail_fast: customOptions.fail_fast ?? false,
                    aggregate_results: customOptions.aggregate_results ?? true
                }
            };

            console.log(`✅ Built batch request with ${requests.length} items`);
            return batchRequest;

        } catch (error) {
            console.error('❌ Batch request building failed:', error);
            throw error;
        }
    }

    /**
     * Generate batch ID
     */
    generateBatchId() {
        const timestamp = Date.now();
        const random = Math.random().toString(36).substr(2, 9);
        return `batch_${timestamp}_${random}`;
    }

    /**
     * Validate request before sending
     */
    validateRequest(request) {
        try {
            // Basic structure validation
            if (!request || typeof request !== 'object') {
                throw new Error('Request must be an object');
            }

            // Check for required fields based on request type
            if (request.query) {
                this.validateChatRequest(request);
            } else if (request.batch_id) {
                this.validateBatchRequest(request);
            } else {
                throw new Error('Unknown request type');
            }

            return true;

        } catch (error) {
            console.error('❌ Request validation failed:', error);
            throw error;
        }
    }

    /**
     * Validate batch request
     */
    validateBatchRequest(request) {
        if (!request.batch_id) {
            throw new Error('Batch request missing batch_id');
        }

        if (!Array.isArray(request.requests) || request.requests.length === 0) {
            throw new Error('Batch request must contain non-empty requests array');
        }

        // Validate each individual request
        for (let i = 0; i < request.requests.length; i++) {
            try {
                this.validateChatRequest(request.requests[i]);
            } catch (error) {
                throw new Error(`Batch item ${i} validation failed: ${error.message}`);
            }
        }
    }

    /**
     * Get request statistics
     */
    getRequestStatistics() {
        return {
            total_requests_built: this.requestId,
            default_options: this.defaultOptions,
            validation_enabled: this.config.requests?.validateInput ?? true,
            metadata_enabled: this.config.requests?.includeMetadata ?? true
        };
    }

    /**
     * Update default options
     */
    updateDefaultOptions(newOptions) {
        this.defaultOptions = {
            ...this.defaultOptions,
            ...newOptions
        };
        console.log('✅ Default options updated');
    }

    /**
     * Reset request builder state
     */
    reset() {
        this.requestId = 0;
        console.log('✅ Request builder state reset');
    }
} 