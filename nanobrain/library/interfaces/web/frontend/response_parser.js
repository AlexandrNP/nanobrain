/**
 * Universal Response Parser for NanoBrain Framework
 * Parses and processes server responses with format detection and validation.
 * 
 * Author: NanoBrain Development Team
 * Date: January 2025
 * Version: 1.0.0
 */

/**
 * Response format types
 */
export const ResponseFormats = {
    STRUCTURED_DATA: 'structured_data',
    TEXT: 'text',
    TABLE: 'table',
    ERROR: 'error',
    STREAMING: 'streaming',
    MIXED: 'mixed',
    VISUALIZATION: 'visualization',
    FILE: 'file'
};

/**
 * Data processing strategies
 */
export const ProcessingStrategies = {
    DIRECT: 'direct',
    TRANSFORM: 'transform',
    AGGREGATE: 'aggregate',
    FILTER: 'filter',
    VALIDATE: 'validate'
};

/**
 * Universal Response Parser Class
 * Handles parsing and processing of diverse server response formats
 */
export class UniversalResponseParser {
    constructor(config = {}) {
        this.config = config;
        this.formatParsers = new Map();
        this.dataProcessors = new Map();
        this.eventHandlers = new Map();

        // Initialize parsers and processors
        this.initializeFormatParsers();
        this.initializeDataProcessors();

        console.log('📥 Universal Response Parser initialized');
    }

    /**
     * Initialize format-specific parsers
     */
    initializeFormatParsers() {
        // Structured data parser
        this.formatParsers.set(ResponseFormats.STRUCTURED_DATA, (response) => {
            return this.parseStructuredData(response);
        });

        // Text parser
        this.formatParsers.set(ResponseFormats.TEXT, (response) => {
            return this.parseTextData(response);
        });

        // Table parser
        this.formatParsers.set(ResponseFormats.TABLE, (response) => {
            return this.parseTableData(response);
        });

        // Error parser
        this.formatParsers.set(ResponseFormats.ERROR, (response) => {
            return this.parseErrorData(response);
        });

        // Streaming parser
        this.formatParsers.set(ResponseFormats.STREAMING, (response) => {
            return this.parseStreamingData(response);
        });

        // Mixed format parser
        this.formatParsers.set(ResponseFormats.MIXED, (response) => {
            return this.parseMixedData(response);
        });
    }

    /**
     * Initialize data processors
     */
    initializeDataProcessors() {
        // Direct processor (no transformation)
        this.dataProcessors.set(ProcessingStrategies.DIRECT, (data) => {
            return data;
        });

        // Transform processor
        this.dataProcessors.set(ProcessingStrategies.TRANSFORM, (data, options) => {
            return this.transformData(data, options);
        });

        // Aggregate processor
        this.dataProcessors.set(ProcessingStrategies.AGGREGATE, (data, options) => {
            return this.aggregateData(data, options);
        });

        // Filter processor
        this.dataProcessors.set(ProcessingStrategies.FILTER, (data, options) => {
            return this.filterData(data, options);
        });

        // Validate processor
        this.dataProcessors.set(ProcessingStrategies.VALIDATE, (data, options) => {
            return this.validateData(data, options);
        });
    }

    /**
     * Parse server response
     */
    async parseResponse(response) {
        try {
            console.log('📥 Parsing server response');

            // Validate response structure
            const validationResult = this.validateResponseStructure(response);
            if (!validationResult.valid) {
                throw new Error(`Invalid response structure: ${validationResult.errors.join(', ')}`);
            }

            // Detect response format
            const detectedFormat = this.detectResponseFormat(response);

            // Parse using appropriate format parser
            const parsedData = await this.parseByFormat(response, detectedFormat);

            // Process data based on frontend hints
            const processedData = await this.processResponseData(parsedData, response.frontend_hints);

            // Create final parsed response
            const parsedResponse = {
                ...response,
                parsed_data: processedData,
                detected_format: detectedFormat,
                parsing_metadata: {
                    parser_version: '1.0.0',
                    parsed_at: new Date().toISOString(),
                    processing_strategy: processedData.processing_strategy || 'direct'
                }
            };

            this.emit('parseComplete', parsedResponse);
            console.log('✅ Response parsing completed');

            return parsedResponse;

        } catch (error) {
            console.error('❌ Response parsing failed:', error);
            this.emit('parseError', { error, response });
            return this.createErrorResponse(response, error.message);
        }
    }

    /**
     * Validate response structure
     */
    validateResponseStructure(response) {
        const errors = [];

        if (!response || typeof response !== 'object') {
            errors.push('Response must be an object');
        } else {
            // Check for required fields
            if (!('success' in response)) {
                errors.push('Missing success field');
            }

            if (!response.message && !response.data) {
                errors.push('Response must contain either message or data');
            }

            if (response.success === false && !response.error_message) {
                errors.push('Error responses must include error_message');
            }
        }

        return {
            valid: errors.length === 0,
            errors: errors
        };
    }

    /**
     * Detect response format
     */
    detectResponseFormat(response) {
        try {
            // Check explicit response_format field first
            if (response.response_format) {
                if (Object.values(ResponseFormats).includes(response.response_format)) {
                    return response.response_format;
                }
            }

            // Check frontend hints
            if (response.frontend_hints?.render_suggestion) {
                switch (response.frontend_hints.render_suggestion) {
                    case 'json_tree':
                        return ResponseFormats.STRUCTURED_DATA;
                    case 'data_table':
                        return ResponseFormats.TABLE;
                    case 'formatted_text':
                        return ResponseFormats.TEXT;
                    case 'error_display':
                        return ResponseFormats.ERROR;
                    default:
                        break;
                }
            }

            // Auto-detect based on data structure
            if (!response.success) {
                return ResponseFormats.ERROR;
            }

            if (response.data) {
                const data = response.data;

                // Check for table structure
                if (data.headers && data.rows) {
                    return ResponseFormats.TABLE;
                }

                // Check for structured data
                if (typeof data === 'object' && !Array.isArray(data)) {
                    return ResponseFormats.STRUCTURED_DATA;
                }

                // Check for array data
                if (Array.isArray(data)) {
                    return ResponseFormats.STRUCTURED_DATA;
                }
            }

            // Default to text if only message
            if (response.message && !response.data) {
                return ResponseFormats.TEXT;
            }

            // Default fallback
            return ResponseFormats.MIXED;

        } catch (error) {
            console.warn('⚠️ Format detection failed, using mixed format:', error);
            return ResponseFormats.MIXED;
        }
    }

    /**
     * Parse response by detected format
     */
    async parseByFormat(response, format) {
        try {
            const parser = this.formatParsers.get(format);
            if (!parser) {
                console.warn(`⚠️ No parser for format: ${format}, using default`);
                return this.parseStructuredData(response);
            }

            return await parser(response);

        } catch (error) {
            console.error(`❌ Format parsing failed for ${format}:`, error);
            // Fallback to basic parsing
            return this.parseStructuredData(response);
        }
    }

    /**
     * Parse structured data format
     */
    parseStructuredData(response) {
        try {
            const data = response.data || {};

            return {
                format: ResponseFormats.STRUCTURED_DATA,
                content: data,
                metadata: {
                    original_format: 'structured',
                    data_type: Array.isArray(data) ? 'array' : typeof data,
                    item_count: Array.isArray(data) ? data.length : Object.keys(data).length
                },
                display_options: {
                    collapsible: true,
                    searchable: true,
                    copyable: true
                }
            };

        } catch (error) {
            console.error('❌ Structured data parsing failed:', error);
            throw error;
        }
    }

    /**
     * Parse text data format
     */
    parseTextData(response) {
        try {
            const content = response.message || response.data || '';
            const textContent = typeof content === 'string' ? content : String(content);

            return {
                format: ResponseFormats.TEXT,
                content: textContent,
                metadata: {
                    original_format: 'text',
                    character_count: textContent.length,
                    line_count: textContent.split('\n').length,
                    word_count: textContent.split(/\s+/).filter(word => word.length > 0).length
                },
                display_options: {
                    preserve_formatting: true,
                    enable_markdown: this.detectMarkdown(textContent),
                    enable_copy: true
                }
            };

        } catch (error) {
            console.error('❌ Text data parsing failed:', error);
            throw error;
        }
    }

    /**
     * Parse table data format
     */
    parseTableData(response) {
        try {
            const data = response.data || {};

            if (!data.headers || !data.rows) {
                throw new Error('Table data must contain headers and rows');
            }

            return {
                format: ResponseFormats.TABLE,
                content: {
                    headers: data.headers,
                    rows: data.rows,
                    metadata: data.metadata || {}
                },
                metadata: {
                    original_format: 'table',
                    column_count: data.headers.length,
                    row_count: data.rows.length,
                    has_pagination: response.frontend_hints?.paginated || false
                },
                display_options: {
                    sortable: response.frontend_hints?.sortable ?? true,
                    filterable: true,
                    exportable: true,
                    pagination: response.frontend_hints?.paginated || data.rows.length > 50
                }
            };

        } catch (error) {
            console.error('❌ Table data parsing failed:', error);
            throw error;
        }
    }

    /**
     * Parse error data format
     */
    parseErrorData(response) {
        try {
            const errorMessage = response.error_message || response.message || 'Unknown error';
            const errorDetails = response.metadata || {};

            return {
                format: ResponseFormats.ERROR,
                content: {
                    message: errorMessage,
                    details: errorDetails,
                    error_type: errorDetails.error_type || 'unknown',
                    timestamp: errorDetails.timestamp || new Date().toISOString()
                },
                metadata: {
                    original_format: 'error',
                    error_severity: this.determineErrorSeverity(errorMessage),
                    has_details: Object.keys(errorDetails).length > 0
                },
                display_options: {
                    show_retry: response.frontend_hints?.show_retry_option ?? true,
                    show_details: this.config.errorHandling?.showDetailedErrors ?? false,
                    collapsible_details: true
                }
            };

        } catch (error) {
            console.error('❌ Error data parsing failed:', error);
            throw error;
        }
    }

    /**
     * Parse streaming data format
     */
    parseStreamingData(response) {
        try {
            const streamData = response.data || {};

            return {
                format: ResponseFormats.STREAMING,
                content: streamData,
                metadata: {
                    original_format: 'streaming',
                    stream_id: streamData.stream_id,
                    chunk_count: streamData.chunk_count || 1,
                    is_final: streamData.is_final || false,
                    progress: streamData.progress || 0
                },
                display_options: {
                    auto_scroll: true,
                    show_progress: true,
                    show_timestamps: true,
                    buffer_chunks: true
                }
            };

        } catch (error) {
            console.error('❌ Streaming data parsing failed:', error);
            throw error;
        }
    }

    /**
     * Parse mixed data format
     */
    parseMixedData(response) {
        try {
            const content = {
                message: response.message || '',
                data: response.data || null
            };

            return {
                format: ResponseFormats.MIXED,
                content: content,
                metadata: {
                    original_format: 'mixed',
                    has_message: !!response.message,
                    has_data: !!response.data,
                    data_type: response.data ? typeof response.data : null
                },
                display_options: {
                    split_view: true,
                    collapsible_sections: true,
                    adaptive_layout: true
                }
            };

        } catch (error) {
            console.error('❌ Mixed data parsing failed:', error);
            throw error;
        }
    }

    /**
     * Process response data based on hints and configuration
     */
    async processResponseData(parsedData, frontendHints = {}) {
        try {
            let processedData = { ...parsedData };

            // Apply data processing based on hints
            if (frontendHints.large_content) {
                processedData = await this.processLargeContent(processedData);
            }

            if (frontendHints.paginated) {
                processedData = await this.processPaginatedData(processedData);
            }

            if (frontendHints.enable_filtering) {
                processedData = await this.enableDataFiltering(processedData);
            }

            // Apply processing strategy if specified
            const strategy = frontendHints.processing_strategy || ProcessingStrategies.DIRECT;
            processedData = await this.applyProcessingStrategy(processedData, strategy, frontendHints);

            return processedData;

        } catch (error) {
            console.error('❌ Data processing failed:', error);
            return parsedData; // Return original data on processing failure
        }
    }

    /**
     * Apply processing strategy
     */
    async applyProcessingStrategy(data, strategy, options = {}) {
        try {
            const processor = this.dataProcessors.get(strategy);
            if (!processor) {
                console.warn(`⚠️ Unknown processing strategy: ${strategy}`);
                return data;
            }

            const processedData = await processor(data, options);
            processedData.processing_strategy = strategy;

            return processedData;

        } catch (error) {
            console.error(`❌ Processing strategy ${strategy} failed:`, error);
            return data;
        }
    }

    /**
     * Process large content
     */
    async processLargeContent(data) {
        try {
            const processed = { ...data };

            // Add lazy loading options
            processed.display_options = {
                ...processed.display_options,
                lazy_loading: true,
                virtual_scrolling: true,
                chunk_loading: true,
                max_initial_items: 50
            };

            // Add content summary for large text
            if (data.format === ResponseFormats.TEXT && data.content.length > 5000) {
                processed.content_summary = data.content.substring(0, 500) + '...';
                processed.metadata.truncated = true;
            }

            return processed;

        } catch (error) {
            console.error('❌ Large content processing failed:', error);
            return data;
        }
    }

    /**
     * Process paginated data
     */
    async processPaginatedData(data) {
        try {
            const processed = { ...data };

            processed.pagination = {
                enabled: true,
                page_size: 25,
                current_page: 1,
                total_items: this.calculateTotalItems(data),
                show_size_selector: true
            };

            return processed;

        } catch (error) {
            console.error('❌ Pagination processing failed:', error);
            return data;
        }
    }

    /**
     * Enable data filtering
     */
    async enableDataFiltering(data) {
        try {
            const processed = { ...data };

            processed.filtering = {
                enabled: true,
                filter_types: this.detectFilterTypes(data),
                quick_filters: this.generateQuickFilters(data)
            };

            return processed;

        } catch (error) {
            console.error('❌ Filter enabling failed:', error);
            return data;
        }
    }

    /**
     * Transform data based on options
     */
    transformData(data, options = {}) {
        try {
            let transformed = { ...data };

            // Apply transformations based on options
            if (options.normalize) {
                transformed = this.normalizeData(transformed);
            }

            if (options.sort) {
                transformed = this.sortData(transformed, options.sort);
            }

            if (options.group) {
                transformed = this.groupData(transformed, options.group);
            }

            return transformed;

        } catch (error) {
            console.error('❌ Data transformation failed:', error);
            return data;
        }
    }

    /**
     * Aggregate data based on options
     */
    aggregateData(data, options = {}) {
        try {
            // Implementation for data aggregation
            return data; // Placeholder

        } catch (error) {
            console.error('❌ Data aggregation failed:', error);
            return data;
        }
    }

    /**
     * Filter data based on options
     */
    filterData(data, options = {}) {
        try {
            // Implementation for data filtering
            return data; // Placeholder

        } catch (error) {
            console.error('❌ Data filtering failed:', error);
            return data;
        }
    }

    /**
     * Validate data based on options
     */
    validateData(data, options = {}) {
        try {
            // Implementation for data validation
            return data; // Placeholder

        } catch (error) {
            console.error('❌ Data validation failed:', error);
            return data;
        }
    }

    /**
     * Utility methods
     */
    detectMarkdown(text) {
        const markdownPatterns = [
            /^#{1,6}\s/m,       // Headers
            /\*\*.*\*\*/,       // Bold
            /\*.*\*/,           // Italic
            /\[.*\]\(.*\)/,     // Links
            /```[\s\S]*```/,    // Code blocks
            /`.*`/              // Inline code
        ];

        return markdownPatterns.some(pattern => pattern.test(text));
    }

    determineErrorSeverity(errorMessage) {
        const errorMessage_lower = errorMessage.toLowerCase();

        if (errorMessage_lower.includes('critical') || errorMessage_lower.includes('fatal')) {
            return 'critical';
        }
        if (errorMessage_lower.includes('warning') || errorMessage_lower.includes('warn')) {
            return 'warning';
        }
        if (errorMessage_lower.includes('info') || errorMessage_lower.includes('notice')) {
            return 'info';
        }

        return 'error';
    }

    calculateTotalItems(data) {
        if (data.format === ResponseFormats.TABLE && data.content.rows) {
            return data.content.rows.length;
        }
        if (Array.isArray(data.content)) {
            return data.content.length;
        }
        return 1;
    }

    detectFilterTypes(data) {
        // Placeholder for filter type detection
        return ['text', 'date', 'number'];
    }

    generateQuickFilters(data) {
        // Placeholder for quick filter generation
        return [];
    }

    normalizeData(data) {
        // Placeholder for data normalization
        return data;
    }

    sortData(data, sortOptions) {
        // Placeholder for data sorting
        return data;
    }

    groupData(data, groupOptions) {
        // Placeholder for data grouping
        return data;
    }

    /**
     * Create error response
     */
    createErrorResponse(originalResponse, errorMessage) {
        return {
            success: false,
            message: 'Response parsing failed',
            error_message: errorMessage,
            response_format: ResponseFormats.ERROR,
            data: { error: true, original_response: originalResponse },
            metadata: {
                parser_error: true,
                timestamp: new Date().toISOString()
            },
            parsed_data: {
                format: ResponseFormats.ERROR,
                content: {
                    message: errorMessage,
                    details: { parsing_failed: true }
                },
                metadata: {
                    original_format: 'error',
                    error_severity: 'error'
                }
            }
        };
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

    emit(event, data) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`❌ Parser event handler error for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Get parser statistics
     */
    getStatistics() {
        return {
            supported_formats: Object.values(ResponseFormats),
            processing_strategies: Object.values(ProcessingStrategies),
            parsers_registered: this.formatParsers.size,
            processors_registered: this.dataProcessors.size,
            parser_version: '1.0.0'
        };
    }
} 