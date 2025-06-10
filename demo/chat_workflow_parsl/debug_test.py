#!/usr/bin/env python3
"""
Minimal debug test to identify the issue.
"""

print("🔍 Debug test starting...")

try:
    import sys
    from pathlib import Path
    print("✅ Basic imports successful")
    
    # Setup paths
    demo_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / 'demo' / 'chat_workflow_parsl'
    project_root = demo_dir.parent.parent
    sys.path.insert(0, str(project_root))
    print(f"✅ Path setup complete: {project_root}")
    
    # Test config import
    from nanobrain.config import get_config_manager
    print("✅ Config import successful")
    
    config_manager = get_config_manager()
    print("✅ Config manager created")
    
    if config_manager:
        global_config = config_manager.get_config_dict()
        logging_config = global_config.get('logging', {})
        logging_mode = logging_config.get('mode', 'both')
        print(f"✅ Config loaded, logging mode: {logging_mode}")
    
    # Test workflow import
    from nanobrain.library.workflows.chat_workflow_parsl.workflow import create_parsl_chat_workflow
    print("✅ Workflow import successful")
    
    print("🎉 All imports successful - no issues found!")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc() 