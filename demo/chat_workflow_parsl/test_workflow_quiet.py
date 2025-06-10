#!/usr/bin/env python3
"""
Quiet test script for Parsl Chat Workflow

Tests the basic functionality with proper logging configuration to suppress
Parsl debug output when global config is set to file mode.
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Setup paths
demo_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / 'demo' / 'chat_workflow_parsl'
project_root = demo_dir.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging VERY early, before any other imports
def configure_early_logging():
    """Configure logging early to suppress Parsl output when in file mode."""
    try:
        # Try to load global config to check logging mode
        import yaml
        config_path = project_root / 'config' / 'global_config.yml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                global_config = yaml.safe_load(f)
            
            logging_mode = global_config.get('logging', {}).get('mode', 'both')
            
            if logging_mode == 'file':
                # Set environment variable to disable Parsl logging
                os.environ['PARSL_DISABLE_LOGGING'] = '1'
                
                # Configure root logger to be very quiet
                root_logger = logging.getLogger()
                root_logger.setLevel(logging.CRITICAL)
                
                # Remove any existing console handlers
                for handler in root_logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler):
                        if hasattr(handler, 'stream') and handler.stream.name in ['<stdout>', '<stderr>']:
                            root_logger.removeHandler(handler)
                
                # Pre-configure known noisy loggers
                noisy_loggers = [
                    'parsl', 'parsl.dataflow.dflow', 'parsl.executors', 'parsl.providers',
                    'parsl.utils', 'parsl.serialize', 'parsl.process_loggers',
                    'parsl.executors.high_throughput', 'parsl.executors.high_throughput.executor',
                    'parsl.executors.high_throughput.zmq_pipes', 'parsl.monitoring',
                    'parsl.executors.high_throughput.interchange', 'parsl.executors.high_throughput.manager',
                    'parsl.channels.local', 'parsl.launchers.single_node', 'parsl.providers.local.local',
                    'parsl.jobs.strategy', 'parsl.jobs.job_status_poller', 'parsl.executors.status_handling'
                ]
                
                for logger_name in noisy_loggers:
                    logger = logging.getLogger(logger_name)
                    logger.setLevel(logging.CRITICAL)
                    logger.propagate = False
                    # Add null handler to completely suppress output
                    null_handler = logging.NullHandler()
                    logger.handlers = [null_handler]
                
                print("🔇 Configured quiet mode - Parsl debug output suppressed")
                return True
        
        return False
        
    except Exception as e:
        print(f"Warning: Could not configure early logging: {e}")
        return False

# Configure logging before any other imports
quiet_mode = configure_early_logging()

# Import the workflow
sys.path.insert(0, str(project_root / 'library' / 'workflows' / 'chat_workflow_parsl'))
from workflow import ParslChatWorkflow, create_parsl_chat_workflow


async def test_workflow():
    """Test the Parsl chat workflow functionality."""
    print("🧪 Testing NanoBrain Parsl Chat Workflow (Quiet Mode)")
    print("=" * 60)
    
    workflow = None
    try:
        # Create workflow with configuration
        config_path = project_root / 'library' / 'workflows' / 'chat_workflow_parsl' / 'ParslChatWorkflow.yml'
        print(f"📁 Loading config from: {config_path}")
        
        if quiet_mode:
            print("🔇 Parsl debug output suppressed (file-only logging mode)")
        
        workflow = await create_parsl_chat_workflow(str(config_path))
        print("✅ Workflow created successfully")
        
        # Display workflow status
        status = workflow.get_workflow_status()
        print(f"📊 Workflow Status:")
        print(f"   - Name: {status['name']}")
        print(f"   - Initialized: {status['initialized']}")
        print(f"   - Agents: {status['agent_count']}")
        print(f"   - Parsl Executor: {'✅' if status['parsl_executor'] else '❌'}")
        print(f"   - Data Units: {len(status['data_units'])}")
        print(f"   - Parsl Apps: {len(status['parsl_apps'])}")
        
        # Test message processing
        test_message = "Hello! Can you tell me about distributed computing with Parsl?"
        print(f"\n📝 Testing message: {test_message}")
        
        response = await workflow.process_user_input(test_message)
        print(f"🤖 Response: {response}")
        
        # Test performance stats
        stats = await workflow.get_performance_stats()
        print(f"\n📈 Performance Stats: {stats}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if workflow:
            print("\n🧹 Cleaning up...")
            await workflow.shutdown()
            print("✅ Cleanup complete")


if __name__ == "__main__":
    success = asyncio.run(test_workflow())
    sys.exit(0 if success else 1) 