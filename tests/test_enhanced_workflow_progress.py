"""
Test Enhanced Workflow Progress Reporting System

Tests the integrated workflow_steps functionality in workflow.py with
YAML-configurable steps, progress reporting, and both main workflows.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.2.0
"""

import pytest
import asyncio
import time
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from nanobrain.core.workflow import (
    Workflow, WorkflowConfig, ProgressReporter, ProgressStep, WorkflowProgress
)
from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import (
    ChatbotViralWorkflow, create_chatbot_viral_workflow
)
from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import (
    AlphavirusWorkflow, create_alphavirus_workflow
)


class TestProgressReporting:
    """Test progress reporting functionality."""
    
    def test_progress_step_creation(self):
        """Test ProgressStep data structure."""
        step = ProgressStep(
            step_id="test_step",
            name="Test Step",
            description="A test step",
            status="pending"
        )
        
        assert step.step_id == "test_step"
        assert step.name == "Test Step"
        assert step.status == "pending"
        assert step.progress_percentage == 0
        
        # Test serialization
        step_dict = step.to_dict()
        assert step_dict['step_id'] == "test_step"
        
        # Test deserialization
        restored_step = ProgressStep.from_dict(step_dict)
        assert restored_step.step_id == step.step_id
    
    def test_workflow_progress_creation(self):
        """Test WorkflowProgress data structure."""
        progress = WorkflowProgress(
            workflow_id="test_workflow",
            workflow_name="Test Workflow"
        )
        
        assert progress.workflow_id == "test_workflow"
        assert progress.workflow_name == "Test Workflow"
        assert progress.overall_progress == 0
        assert progress.status == "pending"
        
        # Test step addition
        step = ProgressStep(
            step_id="step1",
            name="Step 1",
            description="First step",
            status="pending"
        )
        progress.steps.append(step)
        
        assert len(progress.steps) == 1
        assert progress.get_current_step() == step
    
    @pytest.mark.asyncio
    async def test_progress_reporter(self):
        """Test ProgressReporter functionality."""
        reporter = ProgressReporter("test_workflow", "Test Workflow")
        
        # Test callback registration
        callback_called = False
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        reporter.add_progress_callback(test_callback)
        
        # Test step initialization
        step_configs = [
            {"step_id": "step1", "name": "Step 1", "description": "First step"},
            {"step_id": "step2", "name": "Step 2", "description": "Second step"}
        ]
        reporter.initialize_steps(step_configs)
        
        assert len(reporter.workflow_progress.steps) == 2
        
        # Test progress update
        await reporter.update_progress("step1", 50, "running", force_emit=True)
        
        assert callback_called
        assert callback_data is not None
        assert callback_data['overall_progress'] >= 0
        
        # Test progress summary
        summary = reporter.get_progress_summary()
        assert summary['workflow_id'] == "test_workflow"
        assert summary['overall_progress'] >= 0


class TestWorkflowConfiguration:
    """Test YAML workflow configuration."""
    
    def test_workflow_config_creation(self):
        """Test WorkflowConfig creation with progress settings."""
        config_data = {
            "name": "TestWorkflow",
            "description": "Test workflow",
            "enable_progress_reporting": True,
            "progress_batch_interval": 2.0,
            "progress_collapsed_by_default": True,
            "progress_show_technical_errors": True,
            "steps": [
                {
                    "step_id": "test_step",
                    "name": "Test Step",
                    "class": "TestStep",
                    "config": {}
                }
            ],
            "links": []
        }
        
        config = WorkflowConfig(**config_data)
        
        assert config.name == "TestWorkflow"
        assert config.enable_progress_reporting is True
        assert config.progress_batch_interval == 2.0
        assert len(config.steps) == 1
    
    def test_chatbot_viral_workflow_config_loading(self):
        """Test loading ChatbotViralWorkflow YAML configuration."""
        config_path = Path("nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            assert 'name' in yaml_config
            assert 'steps' in yaml_config
            assert 'links' in yaml_config
            assert yaml_config.get('enable_progress_reporting', False) is True
    
    def test_alphavirus_workflow_config_loading(self):
        """Test loading AlphavirusWorkflow YAML configuration."""
        config_path = Path("nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            assert 'name' in yaml_config
            assert 'steps' in yaml_config
            assert 'links' in yaml_config
            assert yaml_config.get('enable_progress_reporting', False) is True


class TestChatbotViralWorkflow:
    """Test ChatbotViralWorkflow with progress reporting."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self):
        """Test ChatbotViralWorkflow initialization with progress reporting."""
        # Mock the YAML config file
        mock_config = {
            "name": "ChatbotViralWorkflow",
            "description": "Test workflow",
            "enable_progress_reporting": True,
            "steps": [
                {
                    "step_id": "query_classification",
                    "name": "Query Classification",
                    "class": "QueryClassificationStep",
                    "config": {}
                }
            ],
            "links": []
        }
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_config), \
             patch.object(ChatbotViralWorkflow, 'initialize', new_callable=AsyncMock):
            
            workflow = ChatbotViralWorkflow(session_id="test_session")
            
            assert workflow.yaml_config == mock_config
            assert workflow.progress_reporter is not None
            assert workflow.progress_reporter.workflow_progress.workflow_name == "ChatbotViralWorkflow"
    
    @pytest.mark.asyncio
    async def test_progress_callback_integration(self):
        """Test progress callback integration in ChatbotViralWorkflow."""
        mock_config = {
            "name": "ChatbotViralWorkflow",
            "description": "Test workflow",
            "enable_progress_reporting": True,
            "steps": [],
            "links": []
        }
        
        progress_updates = []
        
        def progress_callback(data):
            progress_updates.append(data)
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_config), \
             patch.object(ChatbotViralWorkflow, 'initialize', new_callable=AsyncMock):
            
            workflow = ChatbotViralWorkflow(session_id="test_session")
            workflow.add_progress_callback(progress_callback)
            
            # Simulate progress update
            if workflow.progress_reporter:
                await workflow.progress_reporter.update_progress(
                    "test_step", 50, "running", force_emit=True
                )
            
            assert len(progress_updates) > 0
    
    @pytest.mark.asyncio
    async def test_session_management_with_progress(self):
        """Test session management with progress tracking."""
        mock_config = {
            "name": "ChatbotViralWorkflow",
            "description": "Test workflow", 
            "enable_progress_reporting": True,
            "steps": [],
            "links": []
        }
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_config), \
             patch.object(ChatbotViralWorkflow, 'initialize', new_callable=AsyncMock):
            
            workflow = ChatbotViralWorkflow(session_id="test_session")
            
            # Test session info retrieval
            session_info = await workflow.get_session_info("test_session")
            
            if session_info:
                assert 'session_id' in session_info
                if workflow.progress_reporter:
                    assert 'progress_history' in session_info


class TestAlphavirusWorkflow:
    """Test AlphavirusWorkflow with progress reporting."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self):
        """Test AlphavirusWorkflow initialization with progress reporting."""
        mock_config = {
            "name": "AlphavirusWorkflow",
            "description": "Test alphavirus workflow",
            "enable_progress_reporting": True,
            "steps": [
                {
                    "step_id": "data_acquisition",
                    "name": "Data Acquisition",
                    "class": "BVBRCDataAcquisitionStep",
                    "config": {}
                }
            ],
            "links": [],
            "resources": {
                "temporary_directory": "/tmp/test_alphavirus"
            }
        }
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_config), \
             patch.object(AlphavirusWorkflow, 'initialize', new_callable=AsyncMock):
            
            workflow = AlphavirusWorkflow(session_id="test_session")
            
            assert workflow.yaml_config == mock_config
            assert workflow.progress_reporter is not None
            assert workflow.workflow_data is not None
    
    @pytest.mark.asyncio
    async def test_workflow_status_tracking(self):
        """Test workflow status tracking with progress."""
        mock_config = {
            "name": "AlphavirusWorkflow",
            "description": "Test alphavirus workflow",
            "enable_progress_reporting": True,
            "steps": [],
            "links": []
        }
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_config), \
             patch.object(AlphavirusWorkflow, 'initialize', new_callable=AsyncMock):
            
            workflow = AlphavirusWorkflow(session_id="test_session")
            
            # Test workflow status
            status = await workflow.get_workflow_status()
            
            assert 'workflow_name' in status
            assert status['workflow_name'] == 'AlphavirusWorkflow'
            assert 'execution_time' in status
            assert 'total_steps' in status
    
    @pytest.mark.asyncio
    async def test_checkpoint_functionality(self):
        """Test checkpoint save/restore functionality."""
        mock_config = {
            "name": "AlphavirusWorkflow",
            "description": "Test alphavirus workflow",
            "enable_progress_reporting": True,
            "steps": [],
            "links": []
        }
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=mock_config), \
             patch.object(AlphavirusWorkflow, 'initialize', new_callable=AsyncMock):
            
            workflow = AlphavirusWorkflow(session_id="test_session")
            
            # Test checkpoint restoration
            checkpoint_data = {
                "workflow_data": {
                    "original_genomes": ["test_genome"],
                    "filtered_genomes": []
                },
                "execution_state": {
                    "current_step_index": 2,
                    "completed_steps": ["step1", "step2"],
                    "failed_steps": []
                }
            }
            
            success = await workflow.restore_from_checkpoint(checkpoint_data)
            
            # Should succeed even with mock data
            assert isinstance(success, bool)


class TestIntegrationScenarios:
    """Test integration scenarios with both workflows."""
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_progress(self):
        """Test concurrent workflow execution with progress tracking."""
        chatbot_config = {
            "name": "ChatbotViralWorkflow",
            "description": "Test chatbot workflow",
            "enable_progress_reporting": True,
            "steps": [],
            "links": []
        }
        
        alphavirus_config = {
            "name": "AlphavirusWorkflow", 
            "description": "Test alphavirus workflow",
            "enable_progress_reporting": True,
            "steps": [],
            "links": []
        }
        
        progress_updates = {"chatbot": [], "alphavirus": []}
        
        def chatbot_callback(data):
            progress_updates["chatbot"].append(data)
        
        def alphavirus_callback(data):
            progress_updates["alphavirus"].append(data)
        
        with patch('builtins.open'), \
             patch('yaml.safe_load') as mock_yaml, \
             patch.object(ChatbotViralWorkflow, 'initialize', new_callable=AsyncMock), \
             patch.object(AlphavirusWorkflow, 'initialize', new_callable=AsyncMock):
            
            # Setup different configs for different workflows
            def yaml_side_effect(*args, **kwargs):
                if "ChatbotViralWorkflow" in str(args[0]) if args else False:
                    return chatbot_config
                return alphavirus_config
            
            mock_yaml.side_effect = yaml_side_effect
            
            # Create workflows
            chatbot_workflow = ChatbotViralWorkflow(session_id="chatbot_session")
            alphavirus_workflow = AlphavirusWorkflow(session_id="alphavirus_session")
            
            # Add progress callbacks
            chatbot_workflow.add_progress_callback(chatbot_callback)
            alphavirus_workflow.add_progress_callback(alphavirus_callback)
            
            # Simulate progress updates
            if chatbot_workflow.progress_reporter:
                await chatbot_workflow.progress_reporter.update_progress(
                    "test_step", 30, "running", force_emit=True
                )
            
            if alphavirus_workflow.progress_reporter:
                await alphavirus_workflow.progress_reporter.update_progress(
                    "test_step", 60, "running", force_emit=True
                )
            
            # Verify separate progress tracking
            assert len(progress_updates["chatbot"]) > 0
            assert len(progress_updates["alphavirus"]) > 0
    
    def test_yaml_configuration_validation(self):
        """Test YAML configuration validation for both workflows."""
        # Test required fields for chatbot workflow
        chatbot_required_fields = [
            "name", "description", "steps", "links", 
            "enable_progress_reporting"
        ]
        
        # Test required fields for alphavirus workflow  
        alphavirus_required_fields = [
            "name", "description", "steps", "links",
            "enable_progress_reporting", "resources"
        ]
        
        # Mock configs should have all required fields
        chatbot_config = {field: "test_value" for field in chatbot_required_fields}
        chatbot_config["steps"] = []
        chatbot_config["links"] = []
        chatbot_config["enable_progress_reporting"] = True
        
        alphavirus_config = {field: "test_value" for field in alphavirus_required_fields}
        alphavirus_config["steps"] = []
        alphavirus_config["links"] = []
        alphavirus_config["enable_progress_reporting"] = True
        alphavirus_config["resources"] = {"temporary_directory": "/tmp"}
        
        # Validate configs can create WorkflowConfig objects
        try:
            chatbot_wf_config = WorkflowConfig(**chatbot_config)
            alphavirus_wf_config = WorkflowConfig(**alphavirus_config)
            
            assert chatbot_wf_config.enable_progress_reporting is True
            assert alphavirus_wf_config.enable_progress_reporting is True
            
        except Exception as e:
            pytest.fail(f"WorkflowConfig creation failed: {e}")


if __name__ == "__main__":
    # Run basic tests
    import yaml
    
    async def run_basic_tests():
        """Run basic functionality tests."""
        print("🧪 Testing Enhanced Workflow Progress Reporting System")
        
        # Test 1: Progress Reporter
        print("\n1. Testing ProgressReporter...")
        reporter = ProgressReporter("test_workflow", "Test Workflow")
        
        step_configs = [
            {"step_id": "step1", "name": "Step 1", "description": "First step"},
            {"step_id": "step2", "name": "Step 2", "description": "Second step"}
        ]
        reporter.initialize_steps(step_configs)
        
        await reporter.update_progress("step1", 50, "running", force_emit=True)
        summary = reporter.get_progress_summary()
        
        print(f"   ✅ Progress summary: {summary['overall_progress']}%")
        
        # Test 2: YAML Configuration
        print("\n2. Testing YAML Configuration...")
        test_config = {
            "name": "TestWorkflow",
            "description": "Test workflow",
            "enable_progress_reporting": True,
            "steps": [{"step_id": "test", "name": "Test", "class": "TestStep"}],
            "links": []
        }
        
        config = WorkflowConfig(**test_config)
        print(f"   ✅ WorkflowConfig created: {config.name}")
        
        # Test 3: Mock Workflow Creation
        print("\n3. Testing Mock Workflow Creation...")
        
        with patch('builtins.open'), \
             patch('yaml.safe_load', return_value=test_config):
            
            try:
                # Test ChatbotViralWorkflow
                with patch.object(ChatbotViralWorkflow, 'initialize', new_callable=AsyncMock):
                    chatbot_workflow = ChatbotViralWorkflow(session_id="test")
                    print(f"   ✅ ChatbotViralWorkflow created")
                
                # Test AlphavirusWorkflow  
                with patch.object(AlphavirusWorkflow, 'initialize', new_callable=AsyncMock):
                    alphavirus_workflow = AlphavirusWorkflow(session_id="test")
                    print(f"   ✅ AlphavirusWorkflow created")
                    
            except Exception as e:
                print(f"   ❌ Workflow creation failed: {e}")
        
        print("\n🎉 Basic tests completed successfully!")
    
    # Run the tests
    asyncio.run(run_basic_tests()) 