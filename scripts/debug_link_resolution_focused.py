#!/usr/bin/env python3
"""
Focused debug script for link resolution
"""
import asyncio
import sys
import json
from pathlib import Path

# Add nanobrain to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow

async def debug_link_resolution_focused():
    """Debug link resolution in detail"""
    
    print("🔍 === FOCUSED LINK RESOLUTION DEBUG ===")
    
    # Create workflow instance
    config_path = "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
    workflow = AlphavirusWorkflow.from_config(
        config_path,
        workflow_directory="nanobrain/library/workflows/viral_protein_analysis"
    )
    
    print(f"\n📊 === WORKFLOW STATUS ===")
    print(f"Child steps: {len(workflow.child_steps)}")
    print(f"Step links: {len(workflow.step_links)}")
    print(f"Workflow input data unit: {hasattr(workflow, 'input_data_unit')}")
    print(f"Workflow output data unit: {hasattr(workflow, 'output_data_unit')}")
    
    print(f"\n🔗 === STEP LINKS ANALYSIS ===")
    for link_id, link_instance in workflow.step_links.items():
        print(f"\nLink: {link_id}")
        print(f"  Type: {type(link_instance)}")
        print(f"  Has config: {hasattr(link_instance, 'config')}")
        
        if hasattr(link_instance, 'config'):
            config = link_instance.config
            print(f"  Config type: {type(config)}")
            print(f"  Config has source: {hasattr(config, 'source')}")
            print(f"  Config has target: {hasattr(config, 'target')}")
            
            if hasattr(config, 'source') and hasattr(config, 'target'):
                print(f"  Source ref: {config.source}")
                print(f"  Target ref: {config.target}")
                
        print(f"  Current source: {getattr(link_instance, 'source', 'NOT_SET')}")
        print(f"  Current target: {getattr(link_instance, 'target', 'NOT_SET')}")
        
    print(f"\n👥 === CHILD STEPS ANALYSIS ===")
    for step_id, step_instance in workflow.child_steps.items():
        print(f"\nStep: {step_id}")
        print(f"  Type: {type(step_instance)}")
        print(f"  Has step_input_data_units: {hasattr(step_instance, 'step_input_data_units')}")
        print(f"  Has step_output_data_units: {hasattr(step_instance, 'step_output_data_units')}")
        
        if hasattr(step_instance, 'step_input_data_units') and step_instance.step_input_data_units:
            print(f"  Input data units: {list(step_instance.step_input_data_units.keys())}")
            
        if hasattr(step_instance, 'step_output_data_units') and step_instance.step_output_data_units:
            print(f"  Output data units: {list(step_instance.step_output_data_units.keys())}")
    
    print(f"\n🎯 === MANUAL RESOLUTION TEST ===")
    # Test manual resolution
    first_link = list(workflow.step_links.values())[0]
    if hasattr(first_link, 'config') and hasattr(first_link.config, 'source'):
        source_ref = first_link.config.source
        target_ref = first_link.config.target
        
        print(f"Testing resolution of: {source_ref} -> {target_ref}")
        
        # Build workflow context
        workflow_context = {
            'steps': workflow.child_steps,
            'workflow_data_units': {
                'workflow_input': getattr(workflow, 'input_data_unit', None),
                'workflow_output': getattr(workflow, 'output_data_unit', None)
            }
        }
        
        # Filter out None values
        workflow_context['workflow_data_units'] = {
            k: v for k, v in workflow_context['workflow_data_units'].items() 
            if v is not None
        }
        
        print(f"Workflow context steps: {len(workflow_context['steps'])}")
        print(f"Workflow context data units: {list(workflow_context['workflow_data_units'].keys())}")
        
        # Try resolution
        try:
            from nanobrain.core.link import DirectLink
            source_data_unit = DirectLink._resolve_data_unit_reference(source_ref, workflow_context)
            target_data_unit = DirectLink._resolve_data_unit_reference(target_ref, workflow_context)
            
            print(f"✅ Source resolved to: {source_data_unit}")
            print(f"✅ Target resolved to: {target_data_unit}")
            
            # Test step ID mapping  
            source_step_id = workflow._get_step_id_for_data_unit(source_data_unit)
            target_step_id = workflow._get_step_id_for_data_unit(target_data_unit)
            
            print(f"✅ Source step ID: {source_step_id}")
            print(f"✅ Target step ID: {target_step_id}")
            
        except Exception as e:
            print(f"❌ Resolution failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_link_resolution_focused()) 