#!/usr/bin/env python3
"""
Debug script to examine workflow integration process and link resolution
"""
import asyncio
import sys
import json
from pathlib import Path

# Add nanobrain to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow

async def debug_link_integration():
    """Debug workflow link integration process"""
    
    print("🔍 Debugging Link Integration Process")
    
    # Create workflow instance
    config_path = "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
    workflow = AlphavirusWorkflow.from_config(
        config_path,
        workflow_directory="nanobrain/library/workflows/viral_protein_analysis"
    )
    
    print(f"\n📊 Workflow Integration Analysis:")
    print(f"✅ Steps integrated: {len(workflow.child_steps)}")
    print(f"✅ Links integrated: {len(workflow.step_links)}")
    
    print(f"\n📋 Step Details:")
    for step_id, step in workflow.child_steps.items():
        print(f"  - {step_id}: {type(step).__name__}")
        if hasattr(step, 'step_input_data_units'):
            print(f"    Input DUs: {list(step.step_input_data_units.keys()) if step.step_input_data_units else 'None'}")
        if hasattr(step, 'step_output_data_units'):
            print(f"    Output DUs: {list(step.step_output_data_units.keys()) if step.step_output_data_units else 'None'}")
    
    print(f"\n🔗 Link Analysis:")
    for link_id, link in workflow.step_links.items():
        print(f"  - {link_id}: {type(link).__name__}")
        print(f"    Source: {getattr(link.source, 'name', 'None') if link.source else 'None'}")
        print(f"    Target: {getattr(link.target, 'name', 'None') if link.target else 'None'}")
        
        # Check if link has config with string references
        if hasattr(link, 'config'):
            print(f"    Config Source: {getattr(link.config, 'source', 'No source attr')}")
            print(f"    Config Target: {getattr(link.config, 'target', 'No target attr')}")
    
    print(f"\n🔧 Workflow Data Units:")
    print(f"  - Input: {getattr(workflow, 'input_data_unit', 'None')}")
    print(f"  - Output: {getattr(workflow, 'output_data_unit', 'None')}")
    
    # Test manual resolution
    print(f"\n🧪 Testing Manual Resolution:")
    if workflow.step_links:
        link_id, link = next(iter(workflow.step_links.items()))
        print(f"Testing link: {link_id}")
        
        if hasattr(link, 'config') and hasattr(link.config, 'source') and hasattr(link.config, 'target'):
            source_ref = link.config.source
            target_ref = link.config.target
            
            workflow_context = {
                'steps': workflow.child_steps,
                'workflow_data_units': {
                    'workflow_input': getattr(workflow, 'input_data_unit', None),
                    'workflow_output': getattr(workflow, 'output_data_unit', None)
                }
            }
            
            try:
                print(f"  Source ref: {source_ref}")
                print(f"  Target ref: {target_ref}")
                
                if hasattr(link.__class__, '_resolve_data_unit_reference'):
                    resolved_source = link.__class__._resolve_data_unit_reference(source_ref, workflow_context)
                    resolved_target = link.__class__._resolve_data_unit_reference(target_ref, workflow_context)
                    
                    print(f"  ✅ Resolved source: {resolved_source}")
                    print(f"  ✅ Resolved target: {resolved_target}")
                else:
                    print(f"  ❌ Link class doesn't have _resolve_data_unit_reference method")
                    
            except Exception as e:
                print(f"  ❌ Resolution failed: {e}")

if __name__ == "__main__":
    asyncio.run(debug_link_integration()) 