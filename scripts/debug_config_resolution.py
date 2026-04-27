#!/usr/bin/env python3
"""
Debug Configuration Resolution

Simple diagnostic script to understand exactly what's happening 
during the class+config resolution process.
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.config import WorkflowConfig

def debug_config_resolution():
    """Debug the configuration resolution process step by step"""
    
    print("🔍 Debug Configuration Resolution Process")
    print("=" * 60)
    
    # Step 1: Load the raw YAML
    config_path = "nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml"
    print(f"📁 Loading YAML from: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    print("✅ Raw YAML loaded")
    print(f"📊 Steps structure type: {type(raw_config.get('steps', {}))}")
    print(f"📊 Number of steps: {len(raw_config.get('steps', {}))}")
    
    # Step 2: Examine the steps configuration
    steps = raw_config.get('steps', {})
    for step_name, step_config in steps.items():
        print(f"\n🔧 Step: {step_name}")
        print(f"   📝 Config type: {type(step_config)}")
        print(f"   📝 Has 'class': {'class' in step_config}")
        print(f"   📝 Has 'config': {'config' in step_config}")
        
        if 'class' in step_config and 'config' in step_config:
            class_path = step_config['class']
            config_value = step_config['config']
            print(f"   🏷️  Class: {class_path}")
            print(f"   📄 Config value: {config_value}")
            print(f"   📄 Config value type: {type(config_value)}")
            
            # Test the path resolution
            try:
                from nanobrain.core.config.config_base import ConfigLoadingContext
                from datetime import datetime
                
                context = ConfigLoadingContext(
                    base_path=Path(config_path).parent,
                    resolution_stack=set(),
                    loading_timestamp=datetime.now(),
                    workflow_directory="nanobrain/library/workflows/chatbot_viral_integration",
                    additional_context={}
                )
                
                from nanobrain.core.config.config_base import ConfigBase
                resolved_path = ConfigBase._resolve_config_path(config_value, context)
                print(f"   ✅ Resolved path: {resolved_path}")
                print(f"   ✅ Path exists: {Path(resolved_path).exists()}")
                
                if Path(resolved_path).exists():
                    # Load the step config file
                    with open(resolved_path, 'r') as f:
                        step_file_content = yaml.safe_load(f)
                    print(f"   📄 Step file loaded successfully")
                    print(f"   📄 Step file type: {type(step_file_content)}")
                    print(f"   📄 Step file keys: {list(step_file_content.keys()) if isinstance(step_file_content, dict) else 'Not a dict'}")
                
            except Exception as e:
                print(f"   ❌ Path resolution failed: {e}")
    
    # Step 3: Try the actual WorkflowConfig.from_config process
    print(f"\n🚀 Attempting WorkflowConfig.from_config...")
    try:
        workflow_config = WorkflowConfig.from_config(
            config_path,
            workflow_directory="nanobrain/library/workflows/chatbot_viral_integration"
        )
        print("✅ WorkflowConfig loaded successfully!")
    except Exception as e:
        print(f"❌ WorkflowConfig loading failed: {e}")
        import traceback
        print(f"📋 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_config_resolution() 