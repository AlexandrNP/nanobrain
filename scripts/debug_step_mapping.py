#!/usr/bin/env python3

import sys
from pathlib import Path

# Add nanobrain to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow

def debug_step_mapping():
    """Debug step ID mapping and data unit resolution"""
    
    print("🔍 === STEP MAPPING DEBUG ===")
    
    config_path = "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
    workflow = AlphavirusWorkflow.from_config(
        config_path,
        workflow_directory="nanobrain/library/workflows/viral_protein_analysis"
    )
    
    print(f"\n📊 === WORKFLOW CHILD STEPS ===")
    for step_id, step_instance in workflow.child_steps.items():
        print(f"Step ID: {step_id}")
        print(f"  Class: {step_instance.__class__.__name__}")
        
        # Check output data units
        if hasattr(step_instance, 'step_output_data_units') and step_instance.step_output_data_units:
            print(f"  Output data units: {list(step_instance.step_output_data_units.keys())}")
            for du_name, du_instance in step_instance.step_output_data_units.items():
                print(f"    - {du_name}: {du_instance} (id: {id(du_instance)})")
        
        # Check input data units  
        if hasattr(step_instance, 'step_input_data_units') and step_instance.step_input_data_units:
            print(f"  Input data units: {list(step_instance.step_input_data_units.keys())}")
            for du_name, du_instance in step_instance.step_input_data_units.items():
                print(f"    - {du_name}: {du_instance} (id: {id(du_instance)})")
        print()
    
    print(f"\n🔗 === LINK ANALYSIS ===")
    for link_id, link_instance in workflow.step_links.items():
        print(f"Link ID: {link_id}")
        print(f"  Source: {link_instance.source} (id: {id(link_instance.source) if link_instance.source else 'None'})")
        print(f"  Target: {link_instance.target} (id: {id(link_instance.target) if link_instance.target else 'None'})")
        print(f"  Link name: {link_instance.name}")
        print()

if __name__ == "__main__":
    debug_step_mapping() 