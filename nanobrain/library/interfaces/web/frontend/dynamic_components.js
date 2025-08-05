/**
 * Dynamic Component System for NanoBrain Framework
 * Workflow-specific UI component adaptation and dynamic rendering system.
 * 
 * Author: NanoBrain Development Team
 * Date: January 2025
 * Version: 1.0.0
 */

/**
 * Component types supported by the dynamic system
 */
export const ComponentTypes = {
    CHAT_INTERFACE: 'chat_interface',
    DATA_TABLE: 'data_table',
    JSON_VIEWER: 'json_viewer',
    TEXT_DISPLAY: 'text_display',
    PROGRESS_BAR: 'progress_bar',
    ERROR_DISPLAY: 'error_display',
    STREAMING_DISPLAY: 'streaming_display',
    VISUALIZATION: 'visualization',
    FILE_DISPLAY: 'file_display',
    FORM_INPUT: 'form_input'
};

/**
 * Rendering strategies for different response formats
 */
export const RenderingStrategies = {
    STRUCTURED_DATA: 'structured_data',
    TEXT: 'text',
    TABLE: 'table',
    ERROR: 'error',
    STREAMING: 'streaming',
    MIXED: 'mixed',
    VISUALIZATION: 'visualization'
};

/**
 * Dynamic Component System Class
 * Manages workflow-specific UI component adaptation
 */
export class DynamicComponentSystem {
    constructor(config = {}) {
        this.config = config;
        this.componentRegistry = new Map();
        this.activeComponents = new Map();
        this.componentFactories = new Map();
        this.renderingStrategies = new Map();
        this.eventHandlers = new Map();

        // Initialize component system
        this.initializeComponentFactories();
        this.initializeRenderingStrategies();

        console.log('🎨 Dynamic Component System initialized');
    }

    /**
     * Initialize the component system
     */
    async initialize() {
        try {
            // Register default component types
            this.registerDefaultComponents();

            // Setup event handling
            this.setupEventHandling();

            console.log('✅ Component system initialized successfully');

        } catch (error) {
            console.error('❌ Component system initialization failed:', error);
            throw error;
        }
    }

    /**
     * Initialize component factories
     */
    initializeComponentFactories() {
        // Chat interface factory
        this.componentFactories.set(ComponentTypes.CHAT_INTERFACE, (config) => {
            return this.createChatInterface(config);
        });

        // Data table factory
        this.componentFactories.set(ComponentTypes.DATA_TABLE, (config) => {
            return this.createDataTable(config);
        });

        // JSON viewer factory
        this.componentFactories.set(ComponentTypes.JSON_VIEWER, (config) => {
            return this.createJsonViewer(config);
        });

        // Text display factory
        this.componentFactories.set(ComponentTypes.TEXT_DISPLAY, (config) => {
            return this.createTextDisplay(config);
        });

        // Progress bar factory
        this.componentFactories.set(ComponentTypes.PROGRESS_BAR, (config) => {
            return this.createProgressBar(config);
        });

        // Error display factory
        this.componentFactories.set(ComponentTypes.ERROR_DISPLAY, (config) => {
            return this.createErrorDisplay(config);
        });

        // Streaming display factory
        this.componentFactories.set(ComponentTypes.STREAMING_DISPLAY, (config) => {
            return this.createStreamingDisplay(config);
        });
    }

    /**
     * Initialize rendering strategies
     */
    initializeRenderingStrategies() {
        // Structured data strategy
        this.renderingStrategies.set(RenderingStrategies.STRUCTURED_DATA, (data, hints) => {
            return this.renderStructuredData(data, hints);
        });

        // Text strategy
        this.renderingStrategies.set(RenderingStrategies.TEXT, (data, hints) => {
            return this.renderText(data, hints);
        });

        // Table strategy
        this.renderingStrategies.set(RenderingStrategies.TABLE, (data, hints) => {
            return this.renderTable(data, hints);
        });

        // Error strategy
        this.renderingStrategies.set(RenderingStrategies.ERROR, (data, hints) => {
            return this.renderError(data, hints);
        });

        // Streaming strategy
        this.renderingStrategies.set(RenderingStrategies.STREAMING, (data, hints) => {
            return this.renderStreaming(data, hints);
        });
    }

    /**
     * Register default components
     */
    registerDefaultComponents() {
        // Register standard component configurations
        this.registerComponent('default_chat', {
            type: ComponentTypes.CHAT_INTERFACE,
            layout: 'vertical',
            features: ['input', 'history', 'typing_indicator']
        });

        this.registerComponent('default_table', {
            type: ComponentTypes.DATA_TABLE,
            features: ['sorting', 'pagination', 'search']
        });

        this.registerComponent('default_json', {
            type: ComponentTypes.JSON_VIEWER,
            features: ['collapsible', 'syntax_highlighting', 'copy']
        });
    }

    /**
     * Setup event handling
     */
    setupEventHandling() {
        // Component lifecycle events
        this.on('componentCreated', (data) => {
            console.log(`✅ Component created: ${data.componentId}`);
        });

        this.on('componentDestroyed', (data) => {
            console.log(`🗑️ Component destroyed: ${data.componentId}`);
        });

        this.on('adaptationComplete', (data) => {
            console.log(`🔄 Adaptation complete: ${data.strategy}`);
        });
    }

    /**
     * Adapt components based on response configuration
     */
    async adaptComponents(adaptationConfig) {
        try {
            const { responseType, dataStructure, frontendHints, workflowType } = adaptationConfig;

            console.log('🔄 Adapting components for:', responseType);

            // Determine optimal rendering strategy
            const strategy = this.determineRenderingStrategy(responseType, frontendHints);

            // Apply rendering strategy
            const adaptedComponents = await this.applyRenderingStrategy(
                strategy,
                dataStructure,
                frontendHints
            );

            // Update active components
            this.updateActiveComponents(adaptedComponents);

            this.emit('adaptationComplete', {
                strategy,
                componentCount: adaptedComponents.length,
                adaptationConfig
            });

            return adaptedComponents;

        } catch (error) {
            console.error('❌ Component adaptation failed:', error);
            this.emit('adaptationError', { error, adaptationConfig });
            throw error;
        }
    }

    /**
     * Determine optimal rendering strategy
     */
    determineRenderingStrategy(responseType, frontendHints = {}) {
        // Check explicit frontend hints first
        if (frontendHints.render_suggestion) {
            switch (frontendHints.render_suggestion) {
                case 'json_tree':
                    return RenderingStrategies.STRUCTURED_DATA;
                case 'data_table':
                    return RenderingStrategies.TABLE;
                case 'formatted_text':
                    return RenderingStrategies.TEXT;
                case 'error_display':
                    return RenderingStrategies.ERROR;
                default:
                    break;
            }
        }

        // Determine strategy based on response type
        switch (responseType) {
            case 'structured_data':
                return RenderingStrategies.STRUCTURED_DATA;
            case 'table':
                return RenderingStrategies.TABLE;
            case 'text':
            case 'markdown':
                return RenderingStrategies.TEXT;
            case 'error':
                return RenderingStrategies.ERROR;
            case 'streaming':
                return RenderingStrategies.STREAMING;
            default:
                return RenderingStrategies.STRUCTURED_DATA;
        }
    }

    /**
     * Apply rendering strategy
     */
    async applyRenderingStrategy(strategy, data, hints) {
        try {
            const strategyFunction = this.renderingStrategies.get(strategy);
            if (!strategyFunction) {
                throw new Error(`Unknown rendering strategy: ${strategy}`);
            }

            const components = await strategyFunction(data, hints);
            return Array.isArray(components) ? components : [components];

        } catch (error) {
            console.error('❌ Rendering strategy application failed:', error);
            // Fallback to basic text rendering
            return [await this.renderText(data, hints)];
        }
    }

    /**
     * Render structured data
     */
    async renderStructuredData(data, hints) {
        const componentConfig = {
            type: ComponentTypes.JSON_VIEWER,
            data: data,
            options: {
                collapsible: hints.expandable ?? true,
                maxHeight: hints.large_content ? '400px' : '300px',
                showCopyButton: true,
                syntaxHighlighting: true
            }
        };

        return await this.createComponent(componentConfig);
    }

    /**
     * Render text content
     */
    async renderText(data, hints) {
        const componentConfig = {
            type: ComponentTypes.TEXT_DISPLAY,
            data: data,
            options: {
                preserveFormatting: hints.preserveFormatting ?? true,
                maxHeight: hints.large_content ? '400px' : 'auto',
                enableMarkdown: hints.response_format === 'markdown',
                showLineNumbers: false
            }
        };

        return await this.createComponent(componentConfig);
    }

    /**
     * Render table data
     */
    async renderTable(data, hints) {
        const componentConfig = {
            type: ComponentTypes.DATA_TABLE,
            data: data,
            options: {
                sortable: hints.sortable ?? true,
                pagination: hints.paginated ?? false,
                pageSize: 10,
                searchable: data.headers && data.headers.length > 0,
                responsive: true
            }
        };

        return await this.createComponent(componentConfig);
    }

    /**
     * Render error display
     */
    async renderError(data, hints) {
        const componentConfig = {
            type: ComponentTypes.ERROR_DISPLAY,
            data: data,
            options: {
                showRetryButton: hints.show_retry_option ?? true,
                showDetails: this.config.errorHandling?.showDetailedErrors ?? false,
                errorLevel: 'error'
            }
        };

        return await this.createComponent(componentConfig);
    }

    /**
     * Render streaming display
     */
    async renderStreaming(data, hints) {
        const componentConfig = {
            type: ComponentTypes.STREAMING_DISPLAY,
            data: data,
            options: {
                showProgress: true,
                showStatus: true,
                autoScroll: true,
                bufferSize: 100
            }
        };

        return await this.createComponent(componentConfig);
    }

    /**
     * Create a component using factory
     */
    async createComponent(config) {
        try {
            const componentId = `component_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

            const factory = this.componentFactories.get(config.type);
            if (!factory) {
                throw new Error(`No factory found for component type: ${config.type}`);
            }

            const component = await factory({
                ...config,
                componentId
            });

            // Register active component
            this.activeComponents.set(componentId, component);

            this.emit('componentCreated', { componentId, type: config.type });

            return component;

        } catch (error) {
            console.error('❌ Component creation failed:', error);
            this.emit('componentError', { error, config });
            throw error;
        }
    }

    /**
     * Create chat interface component
     */
    createChatInterface(config) {
        return {
            componentId: config.componentId,
            type: ComponentTypes.CHAT_INTERFACE,
            element: this.createChatInterfaceElement(config),
            data: config.data,
            options: config.options || {},
            destroy: () => this.destroyComponent(config.componentId)
        };
    }

    /**
     * Create data table component
     */
    createDataTable(config) {
        return {
            componentId: config.componentId,
            type: ComponentTypes.DATA_TABLE,
            element: this.createDataTableElement(config),
            data: config.data,
            options: config.options || {},
            destroy: () => this.destroyComponent(config.componentId)
        };
    }

    /**
     * Create JSON viewer component
     */
    createJsonViewer(config) {
        return {
            componentId: config.componentId,
            type: ComponentTypes.JSON_VIEWER,
            element: this.createJsonViewerElement(config),
            data: config.data,
            options: config.options || {},
            destroy: () => this.destroyComponent(config.componentId)
        };
    }

    /**
     * Create text display component
     */
    createTextDisplay(config) {
        return {
            componentId: config.componentId,
            type: ComponentTypes.TEXT_DISPLAY,
            element: this.createTextDisplayElement(config),
            data: config.data,
            options: config.options || {},
            destroy: () => this.destroyComponent(config.componentId)
        };
    }

    /**
     * Create progress bar component
     */
    createProgressBar(config) {
        return {
            componentId: config.componentId,
            type: ComponentTypes.PROGRESS_BAR,
            element: this.createProgressBarElement(config),
            data: config.data,
            options: config.options || {},
            destroy: () => this.destroyComponent(config.componentId),
            updateProgress: (progress) => this.updateProgressBar(config.componentId, progress)
        };
    }

    /**
     * Create error display component
     */
    createErrorDisplay(config) {
        return {
            componentId: config.componentId,
            type: ComponentTypes.ERROR_DISPLAY,
            element: this.createErrorDisplayElement(config),
            data: config.data,
            options: config.options || {},
            destroy: () => this.destroyComponent(config.componentId)
        };
    }

    /**
     * Create streaming display component
     */
    createStreamingDisplay(config) {
        return {
            componentId: config.componentId,
            type: ComponentTypes.STREAMING_DISPLAY,
            element: this.createStreamingDisplayElement(config),
            data: config.data,
            options: config.options || {},
            destroy: () => this.destroyComponent(config.componentId),
            addChunk: (chunk) => this.addStreamingChunk(config.componentId, chunk)
        };
    }

    /**
     * Create chat interface DOM element
     */
    createChatInterfaceElement(config) {
        const container = document.createElement('div');
        container.className = 'nanobrain-chat-interface';
        container.innerHTML = `
            <div class="chat-messages" id="messages-${config.componentId}"></div>
            <div class="chat-input">
                <input type="text" placeholder="Type your message..." />
                <button type="button">Send</button>
            </div>
        `;
        return container;
    }

    /**
     * Create data table DOM element
     */
    createDataTableElement(config) {
        const container = document.createElement('div');
        container.className = 'nanobrain-data-table';

        const data = config.data;
        let tableHTML = '<table class="data-table">';

        // Add headers if available
        if (data.headers) {
            tableHTML += '<thead><tr>';
            data.headers.forEach(header => {
                tableHTML += `<th>${this.escapeHtml(header)}</th>`;
            });
            tableHTML += '</tr></thead>';
        }

        // Add rows if available
        if (data.rows) {
            tableHTML += '<tbody>';
            data.rows.forEach(row => {
                tableHTML += '<tr>';
                row.forEach(cell => {
                    tableHTML += `<td>${this.escapeHtml(String(cell))}</td>`;
                });
                tableHTML += '</tr>';
            });
            tableHTML += '</tbody>';
        }

        tableHTML += '</table>';
        container.innerHTML = tableHTML;

        return container;
    }

    /**
     * Create JSON viewer DOM element
     */
    createJsonViewerElement(config) {
        const container = document.createElement('div');
        container.className = 'nanobrain-json-viewer';

        const jsonData = typeof config.data === 'string' ? config.data : JSON.stringify(config.data, null, 2);

        container.innerHTML = `
            <div class="json-controls">
                <button type="button" class="copy-btn">Copy</button>
                ${config.options.collapsible ? '<button type="button" class="collapse-btn">Collapse</button>' : ''}
            </div>
            <pre class="json-content"><code>${this.escapeHtml(jsonData)}</code></pre>
        `;

        return container;
    }

    /**
     * Create text display DOM element
     */
    createTextDisplayElement(config) {
        const container = document.createElement('div');
        container.className = 'nanobrain-text-display';

        const textData = typeof config.data === 'string' ? config.data : String(config.data);

        if (config.options.enableMarkdown) {
            container.innerHTML = `<div class="markdown-content">${this.escapeHtml(textData)}</div>`;
        } else {
            container.innerHTML = `<div class="text-content">${this.escapeHtml(textData)}</div>`;
        }

        return container;
    }

    /**
     * Create progress bar DOM element
     */
    createProgressBarElement(config) {
        const container = document.createElement('div');
        container.className = 'nanobrain-progress-bar';

        const progress = config.data.progress || 0;

        container.innerHTML = `
            <div class="progress-label">${config.data.label || 'Processing...'}</div>
            <div class="progress-container">
                <div class="progress-fill" style="width: ${progress * 100}%"></div>
            </div>
            <div class="progress-text">${Math.round(progress * 100)}%</div>
        `;

        return container;
    }

    /**
     * Create error display DOM element
     */
    createErrorDisplayElement(config) {
        const container = document.createElement('div');
        container.className = 'nanobrain-error-display';

        const errorData = config.data;

        container.innerHTML = `
            <div class="error-icon">⚠️</div>
            <div class="error-message">${this.escapeHtml(errorData.error_message || 'An error occurred')}</div>
            ${config.options.showRetryButton ? '<button type="button" class="retry-btn">Try Again</button>' : ''}
            ${config.options.showDetails && errorData.details ?
                `<details class="error-details">
                    <summary>Error Details</summary>
                    <pre>${this.escapeHtml(JSON.stringify(errorData.details, null, 2))}</pre>
                </details>` : ''}
        `;

        return container;
    }

    /**
     * Create streaming display DOM element
     */
    createStreamingDisplayElement(config) {
        const container = document.createElement('div');
        container.className = 'nanobrain-streaming-display';

        container.innerHTML = `
            <div class="streaming-header">
                <div class="streaming-status">Streaming...</div>
                <div class="streaming-controls">
                    <button type="button" class="pause-btn">Pause</button>
                    <button type="button" class="clear-btn">Clear</button>
                </div>
            </div>
            <div class="streaming-content" id="streaming-${config.componentId}"></div>
        `;

        return container;
    }

    /**
     * Utility function to escape HTML
     */
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, (m) => map[m]);
    }

    /**
     * Update active components
     */
    updateActiveComponents(newComponents) {
        newComponents.forEach(component => {
            this.activeComponents.set(component.componentId, component);
        });
    }

    /**
     * Register a component configuration
     */
    registerComponent(name, config) {
        this.componentRegistry.set(name, config);
    }

    /**
     * Destroy a component
     */
    destroyComponent(componentId) {
        try {
            const component = this.activeComponents.get(componentId);
            if (component) {
                // Remove from DOM if element exists
                if (component.element && component.element.parentNode) {
                    component.element.parentNode.removeChild(component.element);
                }

                // Remove from active components
                this.activeComponents.delete(componentId);

                this.emit('componentDestroyed', { componentId });
            }
        } catch (error) {
            console.error('❌ Component destruction failed:', error);
        }
    }

    /**
     * Get active component by ID
     */
    getComponent(componentId) {
        return this.activeComponents.get(componentId);
    }

    /**
     * Get all active components
     */
    getAllComponents() {
        return Array.from(this.activeComponents.values());
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
                    console.error(`❌ Component event handler error for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Destroy component system and cleanup
     */
    destroy() {
        try {
            // Destroy all active components
            for (const componentId of this.activeComponents.keys()) {
                this.destroyComponent(componentId);
            }

            // Clear registries
            this.componentRegistry.clear();
            this.componentFactories.clear();
            this.renderingStrategies.clear();
            this.eventHandlers.clear();

            console.log('✅ Dynamic Component System destroyed');

        } catch (error) {
            console.error('❌ Component system destruction failed:', error);
        }
    }
} 