#!/usr/bin/env python3
"""
Debug script to examine link configuration structure and test data unit resolution
"""
import asyncio
import sys
import json
from pathlib import Path

# Add nanobrain to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow

async def debug_link_resolution():
    """Debug link configuration and resolution process"""
    
    print("🔍 Debugging Link Resolution")
    
    # Create workflow instance
    config_path = "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
    workflow = AlphavirusWorkflow.from_config(
        config_path,
        workflow_directory="nanobrain/library/workflows/viral_protein_analysis"
    )
    
    print("\n📋 Workflow Configuration Analysis:")
    print(f"  - Has config: {hasattr(workflow, 'config')}")
    print(f"  - Has links: {hasattr(workflow.config, 'links') if hasattr(workflow, 'config') else 'No config'}")
    
    if hasattr(workflow, 'config') and hasattr(workflow.config, 'links'):
        print(f"  - Number of links: {len(workflow.config.links)}")
        
        # Examine link structure
        for link_id, link_config in workflow.config.links.items():
            print(f"\n🔗 Link: {link_id}")
            print(f"  - Type: {type(link_config)}")
            print(f"  - Has config attr: {hasattr(link_config, 'config')}")
            print(f"  - Is dict: {isinstance(link_config, dict)}")
            
            if hasattr(link_config, 'config'):
                config_dict = link_config.config
                print(f"  - Config type: {type(config_dict)}")
                print(f"  - Config content: {config_dict}")
            elif isinstance(link_config, dict):
                print(f"  - Direct dict content: {link_config}")
            
            # Test data unit resolution
            if hasattr(link_config, 'config'):
                config_dict = link_config.config
            elif isinstance(link_config, dict) and 'config' in link_config:
                config_dict = link_config['config']
            else:
                config_dict = link_config
            
            if isinstance(config_dict, dict):
                source_ref = config_dict.get('source')
                target_ref = config_dict.get('target')
                print(f"  - Source reference: {source_ref}")
                print(f"  - Target reference: {target_ref}")
                
                if source_ref:
                    try:
                        source_data_unit = workflow._resolve_data_unit_reference(source_ref)
                        print(f"  - ✅ Source resolved: {source_data_unit}")
                    except Exception as e:
                        print(f"  - ❌ Source resolution failed: {e}")
                
                if target_ref:
                    try:
                        target_data_unit = workflow._resolve_data_unit_reference(target_ref)
                        print(f"  - ✅ Target resolved: {target_data_unit}")
                    except Exception as e:
                        print(f"  - ❌ Target resolution failed: {e}")
    
    print("\n🎯 Child Steps Analysis:")
    print(f"  - Number of child steps: {len(workflow.child_steps)}")
    for step_id, step_instance in workflow.child_steps.items():
        print(f"  - Step {step_id}: {type(step_instance).__name__}")
        
        # Check data units
        print(f"    - Has step_input_data_units: {hasattr(step_instance, 'step_input_data_units')}")
        print(f"    - Has step_output_data_units: {hasattr(step_instance, 'step_output_data_units')}")
        
        if hasattr(step_instance, 'step_input_data_units') and step_instance.step_input_data_units:
            print(f"    - Input data units: {list(step_instance.step_input_data_units.keys())}")
            
        if hasattr(step_instance, 'step_output_data_units') and step_instance.step_output_data_units:
            print(f"    - Output data units: {list(step_instance.step_output_data_units.keys())}")

if __name__ == "__main__":
    asyncio.run(debug_link_resolution()) 