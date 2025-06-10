#!/usr/bin/env python3
"""
Enhanced Validation System Demo for NanoBrain Framework

Demonstrates the advanced validation capabilities including:
- Schema validation with field constraints
- Parameter validation for operations
- Custom validator functions
- Enhanced configuration templates
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
current_file = Path(__file__) if '__file__' in globals() else Path.cwd() / 'demo' / 'enhanced_validation_demo.py'
sys.path.insert(0, str(current_file.parent.parent / "src"))

from nanobrain.config import (
    ComponentFactory, SchemaValidator, ConfigSchema,
    FieldSchema, ParameterSchema, FieldType, ConstraintType, FieldConstraint,
    ValidatorFunction, create_component_from_yaml
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_schema_validation():
    """Demonstrate schema validation capabilities."""
    print("\n" + "="*60)
    print("SCHEMA VALIDATION DEMO")
    print("="*60)
    
    # Create a schema for agent configuration
    agent_schema = ConfigSchema(
        name="AgentSchema",
        version="1.0.0",
        description="Schema for agent configuration validation",
        fields=[
            FieldSchema(
                name="model",
                field_type=FieldType.STRING,
                description="Model name to use",
                required=True,
                constraints=[
                    FieldConstraint(
                        constraint_type=ConstraintType.PATTERN,
                        value=r"^(gpt-|claude-|llama-)",
                        message="Model must be a supported LLM"
                    )
                ]
            ),
            FieldSchema(
                name="temperature",
                field_type=FieldType.FLOAT,
                description="Temperature for generation",
                required=False,
                default=0.7,
                constraints=[
                    FieldConstraint(constraint_type=ConstraintType.MIN, value=0.0),
                    FieldConstraint(constraint_type=ConstraintType.MAX, value=1.0)
                ]
            ),
            FieldSchema(
                name="max_tokens",
                field_type=FieldType.INTEGER,
                description="Maximum tokens to generate",
                required=False,
                default=2000,
                constraints=[
                    FieldConstraint(constraint_type=ConstraintType.MIN, value=1),
                    FieldConstraint(constraint_type=ConstraintType.MAX, value=16000)
                ]
            )
        ],
        required_fields=["model"],
        optional_fields=["temperature", "max_tokens"]
    )
    
    # Create validator
    validator = SchemaValidator(agent_schema)
    
    # Test valid configuration
    print("\n1. Testing valid configuration:")
    valid_config = {
        "model": "gpt-4-turbo",
        "temperature": 0.3,
        "max_tokens": 1500
    }
    print(f"Input: {valid_config}")
    
    try:
        validated = validator.validate_config(valid_config)
        print(f"✓ Validation passed: {validated}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Test invalid model
    print("\n2. Testing invalid model:")
    invalid_model_config = {
        "model": "invalid-model",
        "temperature": 0.3
    }
    print(f"Input: {invalid_model_config}")
    
    try:
        validated = validator.validate_config(invalid_model_config)
        print(f"✓ Validation passed: {validated}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Test missing required field
    print("\n3. Testing missing required field:")
    missing_field_config = {
        "temperature": 0.3,
        "max_tokens": 1500
    }
    print(f"Input: {missing_field_config}")
    
    try:
        validated = validator.validate_config(missing_field_config)
        print(f"✓ Validation passed: {validated}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Test out of range values
    print("\n4. Testing out of range temperature:")
    out_of_range_config = {
        "model": "gpt-4",
        "temperature": 1.5  # Too high
    }
    print(f"Input: {out_of_range_config}")
    
    try:
        validated = validator.validate_config(out_of_range_config)
        print(f"✓ Validation passed: {validated}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")


def demo_parameter_validation():
    """Demonstrate parameter validation for operations."""
    print("\n" + "="*60)
    print("PARAMETER VALIDATION DEMO")
    print("="*60)
    
    # Create schema with parameter definitions
    file_writer_schema = ConfigSchema(
        name="FileWriterSchema",
        version="1.0.0",
        description="Schema for file writer operations",
        parameters=[
            ParameterSchema(
                operation="write",
                description="Parameters for writing a file",
                fields=[
                    FieldSchema(
                        name="path",
                        field_type=FieldType.STRING,
                        description="Path to the file to write",
                        required=True,
                        constraints=[
                            FieldConstraint(
                                constraint_type=ConstraintType.PATTERN,
                                value=r"^[^<>:\"|?*]+$",
                                message="Path must not contain invalid characters"
                            )
                        ]
                    ),
                    FieldSchema(
                        name="content",
                        field_type=FieldType.STRING,
                        description="Content to write",
                        required=True,
                        constraints=[
                            FieldConstraint(
                                constraint_type=ConstraintType.MAX_LENGTH,
                                value=10000,
                                message="Content too long"
                            )
                        ]
                    ),
                    FieldSchema(
                        name="encoding",
                        field_type=FieldType.STRING,
                        description="File encoding",
                        required=False,
                        default="utf-8",
                        constraints=[
                            FieldConstraint(
                                constraint_type=ConstraintType.ENUM,
                                value=["utf-8", "ascii", "latin-1"],
                                message="Unsupported encoding"
                            )
                        ]
                    )
                ]
            ),
            ParameterSchema(
                operation="read",
                description="Parameters for reading a file",
                fields=[
                    FieldSchema(
                        name="path",
                        field_type=FieldType.STRING,
                        description="Path to the file to read",
                        required=True
                    ),
                    FieldSchema(
                        name="max_size",
                        field_type=FieldType.INTEGER,
                        description="Maximum file size to read",
                        required=False,
                        default=1048576,  # 1MB
                        constraints=[
                            FieldConstraint(constraint_type=ConstraintType.MIN, value=1),
                            FieldConstraint(constraint_type=ConstraintType.MAX, value=10485760)  # 10MB
                        ]
                    )
                ]
            )
        ]
    )
    
    validator = SchemaValidator(file_writer_schema)
    
    # Test write operation parameters
    print("\n1. Testing write operation parameters:")
    write_params = {
        "path": "output.txt",
        "content": "Hello, World!",
        "encoding": "utf-8"
    }
    print(f"Input: {write_params}")
    
    try:
        validated = validator.validate_parameters("write", write_params)
        print(f"✓ Validation passed: {validated}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Test invalid encoding
    print("\n2. Testing invalid encoding:")
    invalid_encoding_params = {
        "path": "output.txt",
        "content": "Hello, World!",
        "encoding": "invalid-encoding"
    }
    print(f"Input: {invalid_encoding_params}")
    
    try:
        validated = validator.validate_parameters("write", invalid_encoding_params)
        print(f"✓ Validation passed: {validated}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Test read operation with defaults
    print("\n3. Testing read operation with defaults:")
    read_params = {
        "path": "input.txt"
    }
    print(f"Input: {read_params}")
    
    try:
        validated = validator.validate_parameters("read", read_params)
        print(f"✓ Validation passed: {validated}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")


def demo_custom_validators():
    """Demonstrate custom validator functions."""
    print("\n" + "="*60)
    print("CUSTOM VALIDATORS DEMO")
    print("="*60)
    
    # Create schema with custom validators
    code_gen_schema = ConfigSchema(
        name="CodeGenSchema",
        version="1.0.0",
        description="Schema with custom validators for code generation",
        fields=[
            FieldSchema(
                name="language",
                field_type=FieldType.STRING,
                description="Programming language",
                required=True,
                constraints=[
                    FieldConstraint(
                        constraint_type=ConstraintType.ENUM,
                        value=["Python", "JavaScript", "TypeScript", "Java", "Go"],
                        message="Unsupported language"
                    )
                ]
            ),
            FieldSchema(
                name="complexity",
                field_type=FieldType.STRING,
                description="Code complexity level",
                required=False,
                default="intermediate",
                constraints=[
                    FieldConstraint(
                        constraint_type=ConstraintType.ENUM,
                        value=["simple", "intermediate", "advanced", "expert"],
                        message="Invalid complexity level"
                    )
                ]
            ),
            FieldSchema(
                name="context",
                field_type=FieldType.STRING,
                description="Additional context",
                required=False
            )
        ],
        validators=[
            ValidatorFunction(
                name="validate_language_context",
                fields=["language", "context"],
                pre=True,
                description="Validate that context matches the selected language",
                code="""
language = values.get('language', 'Python')
context = values.get('context', '')

if context and language:
    # Check for language-specific keywords in context
    language_keywords = {
        'Python': ['def', 'class', 'import', 'from', '__init__'],
        'JavaScript': ['function', 'const', 'let', 'var', 'class'],
        'TypeScript': ['interface', 'type', 'enum', 'namespace'],
        'Java': ['public', 'private', 'class', 'interface', 'package'],
        'Go': ['func', 'type', 'struct', 'interface', 'package']
    }
    
    if language in language_keywords:
        keywords = language_keywords[language]
        # If context contains code, check if it matches the language
        if any(keyword in context for keyword in keywords):
            # Context seems to match the language
            pass
        elif len(context) > 100 and not any(keyword in context for keyword in keywords):
            # Long context without language keywords might be mismatched
            values['_warning'] = f"Context may not match selected language: {language}"

return values
"""
            ),
            ValidatorFunction(
                name="validate_complexity_consistency",
                fields=["complexity", "context"],
                pre=True,
                description="Ensure complexity level matches context requirements",
                code="""
complexity = values.get('complexity', 'intermediate')
context = values.get('context', '')

if context:
    context_lower = context.lower()
    
    # Check for complexity indicators
    simple_indicators = ['basic', 'simple', 'easy', 'beginner']
    advanced_indicators = ['complex', 'advanced', 'sophisticated', 'enterprise']
    
    has_simple = any(indicator in context_lower for indicator in simple_indicators)
    has_advanced = any(indicator in context_lower for indicator in advanced_indicators)
    
    if has_simple and complexity in ['advanced', 'expert']:
        values['complexity'] = 'intermediate'
        values['_adjustment'] = 'Adjusted complexity from advanced to intermediate based on context'
    elif has_advanced and complexity == 'simple':
        values['complexity'] = 'intermediate'
        values['_adjustment'] = 'Adjusted complexity from simple to intermediate based on context'

return values
"""
            )
        ]
    )
    
    validator = SchemaValidator(schema=code_gen_schema)
    
    # Test matching language and context
    print("\n1. Testing matching language and context:")
    matching_config = {
        "language": "Python",
        "complexity": "intermediate",
        "context": "Create a Python class with def __init__ and import statements"
    }
    print(f"Input: {matching_config}")
    
    try:
        validated = validator.validate_config(matching_config)
        print(f"✓ Validation passed: {validated}")
        if '_warning' in validated:
            print(f"  Warning: {validated['_warning']}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Test mismatched language and context
    print("\n2. Testing mismatched language and context:")
    mismatched_config = {
        "language": "Python",
        "complexity": "intermediate",
        "context": "Create a JavaScript function with const and let variables for handling async operations"
    }
    print(f"Input: {mismatched_config}")
    
    try:
        validated = validator.validate_config(mismatched_config)
        print(f"✓ Validation passed: {validated}")
        if '_warning' in validated:
            print(f"  Warning: {validated['_warning']}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Test complexity adjustment
    print("\n3. Testing complexity adjustment:")
    complexity_config = {
        "language": "Java",
        "complexity": "simple",
        "context": "Create an enterprise-level microservice with advanced caching and sophisticated error handling"
    }
    print(f"Input: {complexity_config}")
    
    try:
        validated = validator.validate_config(complexity_config)
        print(f"✓ Validation passed: {validated}")
        if '_adjustment' in validated:
            print(f"  Adjustment: {validated['_adjustment']}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")


def demo_enhanced_templates():
    """Demonstrate enhanced configuration templates."""
    print("\n" + "="*60)
    print("ENHANCED TEMPLATES DEMO")
    print("="*60)
    
    factory = ComponentFactory()
    
    # Test enhanced coder template
    print("\n1. Testing enhanced coder template:")
    try:
        coder_agent = create_component_from_yaml("step_coder_enhanced.yml", "enhanced_coder")
        print(f"✓ Created enhanced coder agent: {coder_agent}")
        print(f"  Agent type: {type(coder_agent).__name__}")
        if hasattr(coder_agent, 'name'):
            print(f"  Agent name: {coder_agent.name}")
    except Exception as e:
        print(f"✗ Failed to create enhanced coder: {e}")
    
    # Test enhanced file writer template
    print("\n2. Testing enhanced file writer template:")
    try:
        file_writer_agent = create_component_from_yaml("step_file_writer_enhanced.yml", "enhanced_file_writer")
        print(f"✓ Created enhanced file writer agent: {file_writer_agent}")
        print(f"  Agent type: {type(file_writer_agent).__name__}")
        if hasattr(file_writer_agent, 'name'):
            print(f"  Agent name: {file_writer_agent.name}")
    except Exception as e:
        print(f"✗ Failed to create enhanced file writer: {e}")
    
    # Test custom configuration with validation
    print("\n3. Testing custom configuration with validation:")
    custom_config = {
        "name": "CustomAgent",
        "class": "CodeWriterAgent",
        "config": {
            "name": "CustomCodeAgent",
            "model": "gpt-4",
            "temperature": 0.2,
            "max_tokens": 3000,
            "system_prompt": "You are a helpful coding assistant."
        },
        "validation": {
            "required": ["model", "name"],
            "optional": ["temperature", "max_tokens"],
            "constraints": {
                "temperature": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0
                }
            }
        }
    }
    
    try:
        custom_agent = factory.create_component("agent", custom_config, "custom_agent")
        print(f"✓ Created custom agent: {custom_agent}")
        print(f"  Agent type: {type(custom_agent).__name__}")
        if hasattr(custom_agent, 'name'):
            print(f"  Agent name: {custom_agent.name}")
    except Exception as e:
        print(f"✗ Failed to create custom agent: {e}")


def demo_error_handling():
    """Demonstrate comprehensive error handling."""
    print("\n" + "="*60)
    print("ERROR HANDLING DEMO")
    print("="*60)
    
    # Create a strict schema
    strict_schema = ConfigSchema(
        name="StrictSchema",
        version="1.0.0",
        description="Strict schema for error testing",
        fields=[
            FieldSchema(
                name="required_field",
                field_type=FieldType.STRING,
                description="A required field",
                required=True,
                constraints=[
                    FieldConstraint(
                        constraint_type=ConstraintType.MIN_LENGTH,
                        value=5,
                        message="Field must be at least 5 characters"
                    )
                ]
            ),
            FieldSchema(
                name="numeric_field",
                field_type=FieldType.INTEGER,
                description="A numeric field with constraints",
                required=False,
                constraints=[
                    FieldConstraint(constraint_type=ConstraintType.MIN, value=10),
                    FieldConstraint(constraint_type=ConstraintType.MAX, value=100)
                ]
            )
        ],
        required_fields=["required_field"]
    )
    
    validator = SchemaValidator(strict_schema)
    
    # Test various error conditions
    error_cases = [
        {
            "name": "Missing required field",
            "config": {"numeric_field": 50},
            "expected_error": "Required field missing"
        },
        {
            "name": "Field too short",
            "config": {"required_field": "abc", "numeric_field": 50},
            "expected_error": "less than minimum"
        },
        {
            "name": "Numeric value too low",
            "config": {"required_field": "valid_field", "numeric_field": 5},
            "expected_error": "less than minimum"
        },
        {
            "name": "Numeric value too high",
            "config": {"required_field": "valid_field", "numeric_field": 150},
            "expected_error": "greater than maximum"
        },
        {
            "name": "Invalid type conversion",
            "config": {"required_field": "valid_field", "numeric_field": "not_a_number"},
            "expected_error": "invalid literal"
        }
    ]
    
    for i, case in enumerate(error_cases, 1):
        print(f"\n{i}. Testing {case['name']}:")
        print(f"   Input: {case['config']}")
        
        try:
            validated = validator.validate_config(case['config'])
            print(f"   ✗ Expected error but validation passed: {validated}")
        except Exception as e:
            error_msg = str(e).lower()
            expected = case['expected_error'].lower()
            if expected in error_msg:
                print(f"   ✓ Got expected error: {e}")
            else:
                print(f"   ⚠ Got unexpected error: {e}")


def main():
    """Run all validation demos."""
    print("Enhanced Validation System Demo for NanoBrain Framework")
    print("=" * 60)
    
    try:
        demo_schema_validation()
        demo_parameter_validation()
        demo_custom_validators()
        demo_enhanced_templates()
        demo_error_handling()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey features demonstrated:")
        print("✓ Schema validation with field constraints")
        print("✓ Parameter validation for operations")
        print("✓ Custom validator functions with code execution")
        print("✓ Enhanced configuration templates")
        print("✓ Comprehensive error handling and reporting")
        print("✓ Type conversion and validation")
        print("✓ Default value assignment")
        print("✓ Pattern matching and enum validation")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 