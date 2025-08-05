/**
 * Universal Frontend Components for NanoBrain Framework
 * Dynamic frontend interfaces supporting any NanoBrain workflow with natural language input.
 * 
 * Author: NanoBrain Development Team
 * Date: January 2025
 * Version: 1.0.0
 */

// Universal frontend interface for dynamic workflow adaptation
export { UniversalNanoBrainInterface } from './universal_interface.js';

// Dynamic component system for workflow-specific UI adaptation
export {
    DynamicComponentSystem,
    ComponentTypes,
    RenderingStrategies
} from './dynamic_components.js';

// Universal request builder for framework-compliant requests
export { UniversalRequestBuilder } from './request_builder.js';

// Universal response parser for diverse response formats
export {
    UniversalResponseParser,
    ResponseFormats,
    ProcessingStrategies
} from './response_parser.js';

/**
 * Main factory function for creating universal interface instances
 */
export function createUniversalInterface(config = {}) {
    return new UniversalNanoBrainInterface(config);
}

/**
 * Configuration presets for common use cases
 */
export const ConfigPresets = {
    // Development configuration with detailed debugging
    development: {
        server: {
            baseUrl: 'http://localhost:5001',
            timeout: 30000,
            retryAttempts: 3
        },
        interface: {
            enableAutoAdaptation: true,
            enableStreamingSupport: true,
            enableProgressTracking: true,
            adaptationStrategy: 'automatic'
        },
        rendering: {
            theme: 'debug',
            responsiveDesign: true,
            animationDuration: 200
        },
        requests: {
            validateInput: true,
            includeMetadata: true,
            enableCaching: false // Disable for development
        },
        errorHandling: {
            showDetailedErrors: true,
            enableRetry: true
        }
    },

    // Production configuration optimized for performance
    production: {
        server: {
            baseUrl: process.env.NANOBRAIN_SERVER_URL || 'https://api.nanobrain.ai',
            timeout: 45000,
            retryAttempts: 2
        },
        interface: {
            enableAutoAdaptation: true,
            enableStreamingSupport: true,
            enableProgressTracking: false,
            adaptationStrategy: 'automatic'
        },
        rendering: {
            theme: 'clean',
            responsiveDesign: true,
            animationDuration: 300
        },
        requests: {
            validateInput: true,
            includeMetadata: false,
            enableCaching: true
        },
        errorHandling: {
            showDetailedErrors: false,
            enableRetry: true
        }
    },

    // Minimal configuration for basic usage
    minimal: {
        server: {
            baseUrl: 'http://localhost:5001'
        },
        interface: {
            enableAutoAdaptation: false,
            enableStreamingSupport: false,
            enableProgressTracking: false
        },
        requests: {
            validateInput: false,
            includeMetadata: false,
            enableCaching: false
        },
        errorHandling: {
            showDetailedErrors: false,
            enableRetry: false
        }
    },

    // Research configuration with advanced features
    research: {
        server: {
            baseUrl: 'http://localhost:5001',
            timeout: 120000, // Longer timeout for complex workflows
            retryAttempts: 5
        },
        interface: {
            enableAutoAdaptation: true,
            enableStreamingSupport: true,
            enableProgressTracking: true,
            adaptationStrategy: 'intelligent'
        },
        rendering: {
            theme: 'scientific',
            responsiveDesign: true,
            animationDuration: 400
        },
        requests: {
            validateInput: true,
            includeMetadata: true,
            enableCaching: true
        },
        errorHandling: {
            showDetailedErrors: true,
            enableRetry: true
        }
    }
};

/**
 * Utility functions for common frontend operations
 */
export const FrontendUtils = {
    /**
     * Create a chat interface instance with specified configuration
     */
    createChatInterface: (containerId, config = {}) => {
        const mergedConfig = {
            ...ConfigPresets.development,
            ...config,
            interface: {
                ...ConfigPresets.development.interface,
                ...config.interface
            }
        };

        const nanoBrainInterface = new UniversalNanoBrainInterface(mergedConfig);

        // Mount to specified container
        const container = document.getElementById(containerId);
        if (container) {
            nanoBrainInterface.mountToContainer(container);
        }

        return nanoBrainInterface;
    },

    /**
     * Create a data analysis interface for scientific workflows
     */
    createAnalysisInterface: (containerId, config = {}) => {
        const analysisConfig = {
            ...ConfigPresets.research,
            ...config,
            interface: {
                ...ConfigPresets.research.interface,
                adaptationStrategy: 'analysis_optimized',
                ...config.interface
            }
        };

        return FrontendUtils.createChatInterface(containerId, analysisConfig);
    },

    /**
     * Validate browser compatibility
     */
    validateBrowserCompatibility: () => {
        const features = {
            fetch: typeof fetch !== 'undefined',
            promises: typeof Promise !== 'undefined',
            asyncAwait: (async () => { })().constructor === Promise,
            modules: (() => { try { return typeof eval('import') !== 'undefined'; } catch { return false; } })(),
            classes: (() => { try { class Test { } return true; } catch { return false; } })()
        };

        const isCompatible = Object.values(features).every(feature => feature);

        return {
            compatible: isCompatible,
            features: features,
            recommendations: isCompatible ? [] : [
                'Please use a modern browser that supports ES6+ features',
                'Ensure JavaScript modules are enabled',
                'Update to the latest browser version'
            ]
        };
    },

    /**
     * Get framework information
     */
    getFrameworkInfo: () => {
        return {
            name: 'NanoBrain Universal Interface',
            version: '1.0.0',
            build_date: '2025-01-01',
            supported_browsers: ['Chrome 80+', 'Firefox 75+', 'Safari 13+', 'Edge 80+'],
            features: [
                'Universal workflow adaptation',
                'Dynamic component rendering',
                'Real-time streaming support',
                'Multi-format response handling',
                'Framework-compliant request building'
            ]
        };
    }
};

// Default export for convenience
export default {
    UniversalNanoBrainInterface,
    DynamicComponentSystem,
    UniversalRequestBuilder,
    UniversalResponseParser,
    createUniversalInterface,
    ConfigPresets,
    FrontendUtils,
    ComponentTypes,
    RenderingStrategies,
    ResponseFormats,
    ProcessingStrategies
}; 