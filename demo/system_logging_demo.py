#!/usr/bin/env python3
"""
System-Level Logging Demo for NanoBrain Framework

This demo showcases the comprehensive system-level logging capabilities
including organized directory structure, component lifecycle tracking,
and session summaries.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'src'))

# Import NanoBrain components
from core.data_unit import DataUnitMemory, DataUnitConfig
from core.trigger import DataUpdatedTrigger, TriggerConfig
from core.link import DirectLink, LinkConfig
from core.step import Step, StepConfig
from core.agent import ConversationalAgent, AgentConfig
from core.executor import LocalExecutor, ExecutorConfig

# Import comprehensive logging system
from core.logging_system import (
    get_logger, get_system_log_manager, set_debug_mode,
    log_component_lifecycle, register_component, log_workflow_event,
    log_data_unit_operation, log_trigger_event, log_link_operation,
    create_session_summary
)

class SystemLoggingDemo:
    """Comprehensive demo of system-level logging capabilities."""
    
    def __init__(self):
        # Initialize system-level logging
        self.system_manager = get_system_log_manager()
        self.logger = get_logger("system_demo", category="workflows")
        
        # Register this demo as a workflow component
        register_component("workflows", "system_demo", self, {
            "description": "Comprehensive system logging demonstration",
            "features": ["data_units", "triggers", "links", "steps", "agents"]
        })
        
        # Components
        self.executor = None
        self.data_units = {}
        self.triggers = {}
        self.links = {}
        self.steps = {}
        self.agents = {}
        
        log_workflow_event("system_demo", "initialize")
    
    async def setup_executor(self):
        """Setup and register executor."""
        log_workflow_event("system_demo", "setup_executor_start")
        
        self.executor = LocalExecutor()
        register_component("executors", "local_executor", self.executor, {
            "type": "LocalExecutor",
            "purpose": "Demo execution"
        })
        
        log_component_lifecycle("executors", "local_executor", "initialize")
        await self.executor.initialize()
        log_component_lifecycle("executors", "local_executor", "initialized")
        
        log_workflow_event("system_demo", "setup_executor_complete")
    
    async def setup_data_units(self):
        """Setup and register data units."""
        log_workflow_event("system_demo", "setup_data_units_start")
        
        # Create input data unit
        input_config = DataUnitConfig(name="demo_input", data_type="memory")
        self.data_units["input"] = DataUnitMemory(config=input_config)
        
        register_component("data_units", "demo_input", self.data_units["input"], {
            "data_type": "memory",
            "purpose": "Input data storage"
        })
        
        # Create output data unit
        output_config = DataUnitConfig(name="demo_output", data_type="memory")
        self.data_units["output"] = DataUnitMemory(config=output_config)
        
        register_component("data_units", "demo_output", self.data_units["output"], {
            "data_type": "memory",
            "purpose": "Output data storage"
        })
        
        # Log some data operations
        log_data_unit_operation("demo_input", "create", None, {"initial_state": "empty"})
        log_data_unit_operation("demo_output", "create", None, {"initial_state": "empty"})
        
        # Write some test data
        await self.data_units["input"].write("Hello, system logging!")
        log_data_unit_operation("demo_input", "write", "Hello, system logging!", {"size": 23})
        
        log_workflow_event("system_demo", "setup_data_units_complete", {
            "data_units_created": len(self.data_units)
        })
    
    async def setup_triggers(self):
        """Setup and register triggers."""
        log_workflow_event("system_demo", "setup_triggers_start")
        
        # Create input trigger
        input_trigger_config = TriggerConfig(
            name="input_trigger",
            trigger_type="data_updated"
        )
        self.triggers["input"] = DataUpdatedTrigger(
            config=input_trigger_config,
            data_unit=self.data_units["input"]
        )
        
        register_component("triggers", "input_trigger", self.triggers["input"], {
            "trigger_type": "data_updated",
            "monitors": "demo_input"
        })
        
        log_trigger_event("input_trigger", "data_updated", "created", {
            "monitors": "demo_input"
        })
        
        log_workflow_event("system_demo", "setup_triggers_complete", {
            "triggers_created": len(self.triggers)
        })
    
    async def setup_agent(self):
        """Setup and register agent."""
        log_workflow_event("system_demo", "setup_agent_start")
        
        agent_config = AgentConfig(
            name="demo_agent",
            description="Demo agent for system logging",
            model="mock",  # Use mock for demo
            system_prompt="You are a demo agent showcasing system logging."
        )
        
        self.agents["demo"] = ConversationalAgent(
            config=agent_config,
            executor=self.executor
        )
        
        register_component("agents", "demo_agent", self.agents["demo"], {
            "model": "mock",
            "purpose": "System logging demonstration"
        })
        
        log_component_lifecycle("agents", "demo_agent", "initialize")
        await self.agents["demo"].initialize()
        log_component_lifecycle("agents", "demo_agent", "initialized")
        
        log_workflow_event("system_demo", "setup_agent_complete")
    
    async def setup_links(self):
        """Setup and register links."""
        log_workflow_event("system_demo", "setup_links_start")
        
        # Create link from input to output
        link_config = LinkConfig(
            name="demo_link",
            source="demo_input",
            destination="demo_output"
        )
        
        self.links["demo"] = DirectLink(
            config=link_config,
            source_data_unit=self.data_units["input"],
            destination_data_unit=self.data_units["output"]
        )
        
        register_component("links", "demo_link", self.links["demo"], {
            "source": "demo_input",
            "destination": "demo_output",
            "link_type": "DirectLink"
        })
        
        log_link_operation("demo_link", "created", "demo_input", "demo_output")
        
        log_workflow_event("system_demo", "setup_links_complete", {
            "links_created": len(self.links)
        })
    
    async def demonstrate_operations(self):
        """Demonstrate various operations with logging."""
        log_workflow_event("system_demo", "demonstrate_operations_start")
        
        # Simulate data transfer
        input_data = await self.data_units["input"].read()
        log_data_unit_operation("demo_input", "read", input_data, {"operation": "transfer"})
        
        # Process data through agent
        if self.agents["demo"]:
            log_component_lifecycle("agents", "demo_agent", "processing_start")
            result = await self.agents["demo"].process("Process this data for logging demo")
            log_component_lifecycle("agents", "demo_agent", "processing_complete", {
                "result_length": len(result) if result else 0
            })
        
        # Write to output
        output_data = f"Processed: {input_data}"
        await self.data_units["output"].write(output_data)
        log_data_unit_operation("demo_output", "write", output_data, {"processed": True})
        
        # Simulate link operation
        log_link_operation("demo_link", "transfer", "demo_input", "demo_output", output_data)
        
        # Simulate trigger activation
        log_trigger_event("input_trigger", "data_updated", "activated", {
            "data_changed": True,
            "new_data": input_data
        })
        
        log_workflow_event("system_demo", "demonstrate_operations_complete")
    
    async def shutdown(self):
        """Shutdown all components with logging."""
        log_workflow_event("system_demo", "shutdown_start")
        
        # Shutdown agents
        for name, agent in self.agents.items():
            log_component_lifecycle("agents", f"demo_agent", "shutdown")
        
        # Shutdown triggers
        for name, trigger in self.triggers.items():
            log_component_lifecycle("triggers", "input_trigger", "shutdown")
        
        # Shutdown links
        for name, link in self.links.items():
            log_component_lifecycle("links", "demo_link", "shutdown")
        
        # Shutdown data units
        for name, data_unit in self.data_units.items():
            log_component_lifecycle("data_units", f"demo_{name}", "shutdown")
        
        # Shutdown executor
        if self.executor:
            log_component_lifecycle("executors", "local_executor", "shutdown")
            await self.executor.shutdown()
            log_component_lifecycle("executors", "local_executor", "shutdown_complete")
        
        log_workflow_event("system_demo", "shutdown_complete")
        
        # Create comprehensive session summary
        summary = create_session_summary()
        if summary:
            print(f"\n📊 Session Summary Created:")
            print(f"   Session ID: {summary['session_id']}")
            print(f"   Components Registered: {summary['components_registered']}")
            print(f"   Log Files Created: {len(summary['log_files'])}")
            print(f"   Session Directory: {summary['log_directory']}")
            
            # Show component breakdown
            component_types = {}
            for comp_info in summary['component_registry'].values():
                comp_type = comp_info['type']
                component_types[comp_type] = component_types.get(comp_type, 0) + 1
            
            print(f"   Component Breakdown:")
            for comp_type, count in component_types.items():
                print(f"     - {comp_type}: {count}")
    
    async def run(self):
        """Run the comprehensive system logging demo."""
        try:
            print("🚀 Starting Comprehensive System Logging Demo")
            print("=" * 60)
            
            await self.setup_executor()
            print("✅ Executor setup complete")
            
            await self.setup_data_units()
            print("✅ Data units setup complete")
            
            await self.setup_triggers()
            print("✅ Triggers setup complete")
            
            await self.setup_agent()
            print("✅ Agent setup complete")
            
            await self.setup_links()
            print("✅ Links setup complete")
            
            await self.demonstrate_operations()
            print("✅ Operations demonstration complete")
            
            print("\n🎯 All system components logged successfully!")
            print("📝 Check the logs directory for comprehensive system-level logs")
            
        except Exception as e:
            log_workflow_event("system_demo", "error", {"error": str(e)})
            print(f"❌ Demo failed: {e}")
            raise
        finally:
            await self.shutdown()

async def main():
    """Main entry point for the system logging demo."""
    
    # Enable debug mode for comprehensive logging
    set_debug_mode(True)
    
    print("🔧 Initializing System Logging Demo...")
    
    try:
        demo = SystemLoggingDemo()
        await demo.run()
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ System Logging Demo completed!")
    print("📁 Check the logs/session_YYYYMMDD_HHMMSS directory for:")
    print("   - workflows/     (workflow orchestration logs)")
    print("   - agents/        (agent lifecycle and operations)")
    print("   - executors/     (executor initialization and shutdown)")
    print("   - data_units/    (data operations and transfers)")
    print("   - triggers/      (trigger activations and events)")
    print("   - links/         (link operations and data flow)")
    print("   - session_summary.json (comprehensive session overview)")

if __name__ == "__main__":
    asyncio.run(main()) 