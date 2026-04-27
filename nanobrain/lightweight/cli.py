#!/usr/bin/env python3
"""
Nanobrain Lightweight Wrapper CLI
=================================

Command-line interface for the lightweight wrapper.
Provides easy access to discovery and workflow building functionality.
"""

import sys
import argparse
import json
from pathlib import Path

# Add lightweight wrapper to path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_config_discovery import ComprehensiveConfigDiscovery
from enhanced_workflow_builder import EnhancedWorkflowBuilder


def cmd_discover(args):
    """Run config discovery and show results."""
    
    print("🔍 Running comprehensive config discovery...")
    
    discovery = ComprehensiveConfigDiscovery()
    results = discovery.discover_all_configs()
    
    class_to_configs = results["class_to_configs"]
    
    if args.category:
        # Filter by category
        filtered_classes = []
        for class_name in class_to_configs.keys():
            if args.category.lower() in class_name.lower():
                filtered_classes.append(class_name)
        
        print(f"\n📂 Classes matching '{args.category}' ({len(filtered_classes)}):")
        for class_name in sorted(filtered_classes):
            configs = class_to_configs[class_name]
            print(f"  📁 {class_name}: {len(configs)} configs")
            
            if args.verbose:
                for i, config in enumerate(configs[:3]):
                    priority_labels = ["🥇 CORE", "🥈 COMPONENTS", "🥉 BASE", "🏅 LIBRARY", "🔹 OTHER"]
                    priority_label = priority_labels[min(config["priority"] - 1, 4)]
                    print(f"    {i+1}. {priority_label}: {config['relative_path']}")
                if len(configs) > 3:
                    print(f"    ... and {len(configs) - 3} more")
    else:
        # Show all classes
        print(f"\n📊 Discovery Summary:")
        print(f"  Total classes: {len(class_to_configs)}")
        
        # Group by category
        categories = {}
        for class_name in class_to_configs.keys():
            # Simple categorization based on class name
            if "agent" in class_name.lower():
                category = "Agents"
            elif "step" in class_name.lower():
                category = "Steps"
            elif "tool" in class_name.lower():
                category = "Tools"
            elif "executor" in class_name.lower():
                category = "Executors"
            elif "dataunit" in class_name.lower() or class_name.startswith("DataUnit"):
                category = "Data Units"
            elif "link" in class_name.lower():
                category = "Links"
            else:
                category = "Other"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(class_name)
        
        print(f"\n📂 Classes by Category:")
        for category, classes in sorted(categories.items()):
            print(f"  {category}: {len(classes)}")
            if args.verbose:
                for class_name in sorted(classes):
                    configs = class_to_configs[class_name]
                    print(f"    📁 {class_name}: {len(configs)} configs")
    
    if args.output:
        # Save results to file
        output_data = {
            "discovery_stats": discovery.stats,
            "classes": {
                class_name: [
                    {
                        "relative_path": config["relative_path"],
                        "priority": config["priority"],
                        "class_path": config["class_path"]
                    }
                    for config in configs
                ]
                for class_name, configs in class_to_configs.items()
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Discovery results saved to: {args.output}")


def cmd_list(args):
    """List available components."""
    
    discovery = ComprehensiveConfigDiscovery()
    discovery.discover_all_configs()
    
    available_classes = discovery.list_available_classes()
    
    if args.category:
        # Filter by category
        filtered = [c for c in available_classes if args.category.lower() in c.lower()]
        print(f"📂 {args.category} components ({len(filtered)}):")
        for class_name in sorted(filtered):
            print(f"  📁 {class_name}")
    else:
        print(f"📂 All available components ({len(available_classes)}):")
        for class_name in sorted(available_classes):
            print(f"  📁 {class_name}")


def cmd_build(args):
    """Interactive workflow builder."""
    
    print("🔧 Interactive Workflow Builder")
    print("=" * 40)
    
    # Get workflow details
    name = args.name or input("Workflow name: ")
    description = args.description or input("Description: ")
    
    builder = EnhancedWorkflowBuilder(name, description)
    
    print(f"\n✅ Created workflow: {name}")
    print(f"Available classes: {len(builder.available_classes)}")
    
    # Interactive component addition
    while True:
        print(f"\nCurrent components: {len(builder.components)}")
        
        action = input("\nAction (add/connect/save/quit): ").lower()
        
        if action == "add":
            comp_name = input("Component name: ")
            
            # Show available classes
            print("Available classes:")
            for i, class_name in enumerate(sorted(builder.available_classes)[:10]):
                print(f"  {i+1}. {class_name}")
            print("  ... (use 'list' command to see all)")
            
            class_name = input("Class name: ")
            
            if class_name in builder.available_classes:
                try:
                    builder.add_component(comp_name, class_name)
                    print(f"✅ Added {comp_name}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print(f"❌ Class '{class_name}' not available")
        
        elif action == "connect":
            if len(builder.components) < 2:
                print("❌ Need at least 2 components to connect")
                continue
            
            print("Available components:")
            for comp_name in builder.components:
                print(f"  📁 {comp_name}")
            
            source = input("Source component: ")
            target = input("Target component: ")
            
            try:
                builder.connect(source, target)
                print(f"✅ Connected {source} → {target}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif action == "save":
            output_file = args.output or f"{name}.yml"
            builder.save_workflow(output_file)
            print(f"✅ Saved to {output_file}")
            break
        
        elif action == "quit":
            break
        
        else:
            print("Available actions: add, connect, save, quit")


def cmd_validate(args):
    """Validate a workflow file."""
    
    if not Path(args.file).exists():
        print(f"❌ File not found: {args.file}")
        return 1
    
    print(f"🔍 Validating workflow: {args.file}")
    
    # For now, just check if it's valid YAML
    import yaml
    
    try:
        with open(args.file, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        print("✅ Valid YAML format")
        
        # Basic structure validation
        required_fields = ["name", "steps"]
        missing_fields = [field for field in required_fields if field not in workflow_data]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return 1
        
        print(f"✅ Basic structure valid")
        print(f"   Name: {workflow_data.get('name')}")
        print(f"   Steps: {len(workflow_data.get('steps', {}))}")
        print(f"   Links: {len(workflow_data.get('links', []))}")
        
        return 0
        
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML: {e}")
        return 1
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Nanobrain Lightweight Wrapper CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s discover --category agent --verbose
  %(prog)s list --category tool
  %(prog)s build --name my_workflow
  %(prog)s validate workflow.yml
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Run config discovery")
    discover_parser.add_argument("--category", help="Filter by category")
    discover_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    discover_parser.add_argument("--output", "-o", help="Save results to JSON file")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument("--category", help="Filter by category")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Interactive workflow builder")
    build_parser.add_argument("--name", help="Workflow name")
    build_parser.add_argument("--description", help="Workflow description")
    build_parser.add_argument("--output", "-o", help="Output file")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate workflow file")
    validate_parser.add_argument("file", help="Workflow file to validate")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "discover":
            cmd_discover(args)
        elif args.command == "list":
            cmd_list(args)
        elif args.command == "build":
            cmd_build(args)
        elif args.command == "validate":
            return cmd_validate(args)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
