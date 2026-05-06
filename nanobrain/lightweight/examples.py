#!/usr/bin/env python3
"""
Nanobrain Lightweight Wrapper Examples
======================================

Comprehensive examples showing how to use the lightweight wrapper
to build real nanobrain workflows.
"""

import sys
from pathlib import Path

# Add lightweight wrapper to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_workflow_builder import EnhancedWorkflowBuilder


def example_1_simple_chat():
    """Example 1: Simple chat workflow with intelligent defaults."""
    
    print("🔥 EXAMPLE 1: Simple Chat Workflow")
    print("=" * 50)
    
    # Create workflow builder
    builder = EnhancedWorkflowBuilder(
        "simple_chat", 
        "Basic chat workflow using framework defaults"
    )
    
    # Add components with intelligent defaults
    builder.add_input("user_message", "DataUnitString")
    builder.add_component("chat_agent", "EnhancedCollaborativeAgent")
    builder.add_output("agent_response", "DataUnitString")
    
    # Connect components
    builder.connect("user_message", "chat_agent")
    builder.connect("chat_agent", "agent_response")
    
    # Show workflow summary
    builder.print_workflow_summary()
    
    # Save workflow
    output_file = "example_1_simple_chat.yml"
    builder.save_workflow(output_file)
    
    print(f"✅ Example 1 complete: {output_file}")
    return output_file


def example_2_custom_parameters():
    """Example 2: Workflow with custom parameters and specific configs."""
    
    print("\n🔥 EXAMPLE 2: Custom Parameters and Config Selection")
    print("=" * 60)
    
    # Create workflow builder
    builder = EnhancedWorkflowBuilder(
        "custom_chat", 
        "Chat workflow with custom parameters and config selection"
    )
    
    # Add components with custom parameters
    builder.add_input("user_query", "DataUnitMemory")
    
    # Use specific config and custom parameters
    builder.add_component(
        "specialized_agent", 
        "EnhancedCollaborativeAgent",
        model="gpt-4",
        temperature=0.3,
        max_tokens=1000,
        system_prompt="You are a helpful AI assistant specialized in scientific queries."
    )
    
    # Add executor with custom settings
    builder.add_component(
        "executor", 
        "LocalExecutor",
        max_workers=2,
        timeout=300
    )
    
    builder.add_output("processed_response", "DataUnitMemory")
    
    # Connect components
    builder.connect("user_query", "specialized_agent")
    builder.connect("specialized_agent", "processed_response")
    
    # Show workflow details
    builder.print_workflow_summary()
    
    # Validate before saving
    validation = builder.validate_workflow()
    print(f"\n🔍 Validation Results:")
    print(f"   Valid: {validation['valid']}")
    print(f"   Errors: {len(validation['errors'])}")
    print(f"   Warnings: {len(validation['warnings'])}")
    
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"   ⚠️  {warning}")
    
    # Save workflow
    output_file = "example_2_custom_chat.yml"
    builder.save_workflow(output_file)
    
    print(f"✅ Example 2 complete: {output_file}")
    return output_file


def example_3_multi_step_workflow():
    """Example 3: Multi-step workflow with different component types."""
    
    print("\n🔥 EXAMPLE 3: Multi-Step Workflow")
    print("=" * 40)
    
    # Create workflow builder
    builder = EnhancedWorkflowBuilder(
        "multi_step_analysis", 
        "Multi-step workflow with query analysis and response generation"
    )
    
    # Input
    builder.add_input("raw_query", "DataUnitString")
    
    # Step 1: Query analysis
    if "QueryAnalysisAgent" in builder.available_classes:
        builder.add_component("query_analyzer", "QueryAnalysisAgent")
    
    # Step 2: Main processing
    builder.add_component(
        "main_processor", 
        "EnhancedCollaborativeAgent",
        model="gpt-4",
        temperature=0.7
    )
    
    # Step 3: Response formatting (if available)
    if "ResponseFormattingStep" in builder.available_classes:
        builder.add_component("response_formatter", "ResponseFormattingStep")
    
    # Output
    builder.add_output("final_response", "DataUnitString")
    
    # Connect components based on what's available
    if "query_analyzer" in builder.components:
        builder.connect("raw_query", "query_analyzer")
        builder.connect("query_analyzer", "main_processor")
    else:
        builder.connect("raw_query", "main_processor")
    
    if "response_formatter" in builder.components:
        builder.connect("main_processor", "response_formatter")
        builder.connect("response_formatter", "final_response")
    else:
        builder.connect("main_processor", "final_response")
    
    # Show workflow
    builder.print_workflow_summary()
    
    # Save workflow
    output_file = "example_3_multi_step.yml"
    builder.save_workflow(output_file)
    
    print(f"✅ Example 3 complete: {output_file}")
    return output_file


def example_4_config_exploration():
    """Example 4: Explore available configs and make informed choices."""
    
    print("\n🔥 EXAMPLE 4: Config Exploration and Selection")
    print("=" * 50)
    
    # Create workflow builder
    builder = EnhancedWorkflowBuilder("config_exploration", "Exploring config options")
    
    # Show available components by category
    print("📂 Available Components by Category:")
    categories = ["Agent", "Step", "Tool", "DataUnit", "Executor"]
    
    for category in categories:
        components = builder.list_available_components(category)
        if components:
            print(f"\n  {category} ({len(components)} available):")
            for component in components[:5]:  # Show first 5
                configs = builder.discovery.get_all_configs_for_class(component)
                print(f"    📁 {component}: {len(configs)} configs")
                
                # Show config options for components with multiple configs
                if len(configs) > 1:
                    for i, config in enumerate(configs[:3]):  # Show first 3
                        priority_labels = ["🥇 CORE", "🥈 COMPONENTS", "🥉 BASE", "🏅 LIBRARY", "🔹 OTHER"]
                        priority_label = priority_labels[min(config["priority"] - 1, 4)]
                        print(f"      {i+1}. {priority_label}: {config['relative_path']}")
                    if len(configs) > 3:
                        print(f"      ... and {len(configs) - 3} more")
            
            if len(components) > 5:
                print(f"    ... and {len(components) - 5} more {category.lower()}s")
    
    # Build a workflow using the exploration results
    print(f"\n🔨 Building workflow with explored components:")
    
    # Use DataUnit with specific config choice
    data_unit_configs = builder.discovery.get_all_configs_for_class("DataUnit")
    if len(data_unit_configs) > 1:
        specific_config = data_unit_configs[1]["relative_path"]  # Use second config
        builder.add_input("input", "DataUnit", config_choice=specific_config)
        print(f"   Used specific DataUnit config: {specific_config}")
    else:
        builder.add_input("input", "DataUnit")
    
    # Use default agent
    builder.add_component("agent", "EnhancedCollaborativeAgent")
    builder.add_output("output", "DataUnit")
    
    # Connect
    builder.connect("input", "agent")
    builder.connect("agent", "output")
    
    # Save
    output_file = "example_4_config_exploration.yml"
    builder.save_workflow(output_file)
    
    print(f"✅ Example 4 complete: {output_file}")
    return output_file


def example_5_error_handling():
    """Example 5: Demonstrate error handling and validation."""
    
    print("\n🔥 EXAMPLE 5: Error Handling and Validation")
    print("=" * 45)
    
    builder = EnhancedWorkflowBuilder("error_demo", "Demonstrating error handling")
    
    print("🧪 Testing various error conditions:")
    
    # Test 1: Invalid class name
    try:
        builder.add_component("invalid", "NonExistentClass")
        print("   ❌ Should have failed for invalid class")
    except ValueError as e:
        print(f"   ✅ Correctly caught invalid class: {e}")
    
    # Test 2: Duplicate component name
    try:
        builder.add_component("test", "DataUnit")
        builder.add_component("test", "DataUnit")  # Duplicate
        print("   ❌ Should have failed for duplicate name")
    except ValueError as e:
        print(f"   ✅ Correctly caught duplicate name: {e}")
    
    # Test 3: Invalid config choice
    try:
        builder.add_component("test2", "DataUnit", config_choice="nonexistent.yml")
        print("   ❌ Should have failed for invalid config")
    except ValueError as e:
        print(f"   ✅ Correctly caught invalid config: {e}")
    
    # Test 4: Connection to non-existent component
    try:
        builder.connect("nonexistent", "test")
        print("   ❌ Should have failed for invalid connection")
    except ValueError as e:
        print(f"   ✅ Correctly caught invalid connection: {e}")
    
    # Build a valid workflow for comparison
    builder.add_input("valid_input", "DataUnit")
    builder.add_output("valid_output", "DataUnit")
    builder.connect("valid_input", "test")
    builder.connect("test", "valid_output")
    
    # Show validation results
    validation = builder.validate_workflow()
    print(f"\n🔍 Final validation:")
    print(f"   Valid: {validation['valid']}")
    print(f"   Components: {validation['component_count']}")
    print(f"   Connections: {validation['connection_count']}")
    
    output_file = "example_5_error_handling.yml"
    builder.save_workflow(output_file)
    
    print(f"✅ Example 5 complete: {output_file}")
    return output_file


def main():
    """Run all examples."""
    
    print("🧪 NANOBRAIN LIGHTWEIGHT WRAPPER EXAMPLES")
    print("=" * 60)
    print("Demonstrating real usage of the lightweight wrapper")
    print()
    
    # Run examples
    example_files = []
    
    try:
        example_files.append(example_1_simple_chat())
        example_files.append(example_2_custom_parameters())
        example_files.append(example_3_multi_step_workflow())
        example_files.append(example_4_config_exploration())
        example_files.append(example_5_error_handling())
        
        print(f"\n🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print(f"Generated {len(example_files)} workflow files:")
        for file in example_files:
            print(f"   📄 {file}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Examine the generated YAML files")
        print(f"   2. Use them as templates for your own workflows")
        print(f"   3. Load them into the nanobrain framework for execution")
        
        # Clean up example files
        print(f"\n🧹 Cleaning up example files...")
        for file in example_files:
            Path(file).unlink(missing_ok=True)
        print(f"   ✅ Cleaned up {len(example_files)} files")
        
    except Exception as e:
        print(f"\n💥 Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
