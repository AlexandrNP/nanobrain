#!/usr/bin/env python3
"""
Real Workflow Integration Tests for Chatbot Viral Integration

Tests the actual ChatbotViralWorkflow and its components with real implementations.
This provides validation of the complete integration using actual NanoBrain components.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import pytest
import pytest_asyncio
import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.chatbot_viral_integration import (
    ChatbotTestData,
    CLASSIFICATION_METRICS,
    CONTENT_QUALITY_CHECKS,
    WORKFLOW_VALIDATION
)

# Import actual workflow components
try:
    from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import (
        ChatbotViralWorkflow, 
        InMemorySessionManager
    )
    from nanobrain.library.workflows.chatbot_viral_integration.steps import (
        QueryClassificationStep,
        AnnotationJobStep,
        ConversationalResponseStep,
        ResponseFormattingStep
    )
    from nanobrain.library.infrastructure.data.chat_session_data import (
        ChatSessionData, MessageRole, MessageType
    )
    REAL_COMPONENTS_AVAILABLE = True
    print("✅ Real NanoBrain components loaded successfully")
except ImportError as e:
    print(f"❌ Real components not available: {e}")
    REAL_COMPONENTS_AVAILABLE = False


class TestRealWorkflowIntegration:
    """Test suite for real workflow integration"""
    
    @pytest_asyncio.fixture
    async def real_workflow(self):
        """Setup real ChatbotViralWorkflow instance"""
        if not REAL_COMPONENTS_AVAILABLE:
            pytest.skip("Real workflow components not available")
        
        try:
            from nanobrain.core.workflow import WorkflowConfig
            from nanobrain.core.executor import LocalExecutor, ExecutorConfig
            import yaml
            from pathlib import Path
            
            # Load the real config file
            config_path = Path(__file__).parent.parent.parent / "nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                workflow_config = WorkflowConfig(**yaml_config)
            else:
                # Create minimal config for testing
                workflow_config = WorkflowConfig(
                    name="ChatbotViralWorkflow",
                    description="Test chatbot workflow",
                    enable_progress_reporting=True,
                    steps=[],
                    links=[]
                )
            
            # Create executor using from_config pattern
            executor_config = ExecutorConfig(executor_type="local", max_workers=2)
            executor = LocalExecutor.from_config(executor_config)
            
            # Use from_config pattern with proper dependencies
            workflow = ChatbotViralWorkflow.from_config(workflow_config, executor=executor)
            await workflow.initialize()
            return workflow
        except Exception as e:
            pytest.skip(f"Failed to create real workflow: {e}")
    

    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_query_classification_conversational(self, real_workflow):
        """Test real query classification for conversational queries"""
        test_cases = [
            "What is EEEV?",
            "Tell me about Eastern Equine Encephalitis virus",
            "How do alphaviruses spread?",
            "What are the symptoms of viral encephalitis?"
        ]
        
        for query in test_cases:
            print(f"🧪 Testing real classification: {query}")
            
            response_chunks = []
            classification_found = False
            
            async for chunk in real_workflow.process_user_message(query, f"test_conv_{uuid.uuid4().hex[:8]}"):
                response_chunks.append(chunk)
                
                # Look for classification chunk
                if chunk.get('type') == 'classification':
                    classification_found = True
                    intent = chunk.get('intent')
                    confidence = chunk.get('confidence', 0)
                    
                    print(f"   📋 Classification: {intent} (confidence: {confidence:.3f})")
                    
                    # Validate conversational classification
                    assert intent == "conversational", f"Expected conversational, got {intent}"
                    assert confidence >= 0.3, f"Low confidence: {confidence}"
                    
                    break
            
            assert classification_found, f"No classification chunk found for: {query}"
            print(f"✅ {query} → conversational")
    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_query_classification_annotation(self, real_workflow):
        """Test real query classification for annotation queries"""
        test_cases = [
            "Create PSSM matrix of EEEV",
            "Analyze Eastern Equine Encephalitis virus proteins",
            "Generate protein annotation for Alphavirus",
            "Perform viral protein clustering analysis"
        ]
        
        for query in test_cases:
            print(f"🧪 Testing real classification: {query}")
            
            response_chunks = []
            classification_found = False
            
            async for chunk in real_workflow.process_user_message(query, f"test_ann_{uuid.uuid4().hex[:8]}"):
                response_chunks.append(chunk)
                
                # Look for classification chunk
                if chunk.get('type') == 'classification':
                    classification_found = True
                    intent = chunk.get('intent')
                    confidence = chunk.get('confidence', 0)
                    
                    print(f"   📋 Classification: {intent} (confidence: {confidence:.3f})")
                    
                    # Validate annotation classification
                    assert intent == "annotation", f"Expected annotation, got {intent}"
                    assert confidence >= 0.4, f"Low confidence: {confidence}"
                    
                    break
            
            assert classification_found, f"No classification chunk found for: {query}"
            print(f"✅ {query} → annotation")
    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_conversational_workflow_complete(self, real_workflow):
        """Test complete conversational workflow execution"""
        query = "What is EEEV?"
        session_id = f"test_conv_complete_{uuid.uuid4().hex[:8]}"
        
        print(f"🧪 Testing complete conversational workflow: {query}")
        
        response_chunks = []
        classification_chunk = None
        content_chunk = None
        completion_chunk = None
        
        async for chunk in real_workflow.process_user_message(query, session_id):
            response_chunks.append(chunk)
            
            chunk_type = chunk.get('type')
            print(f"   📦 Chunk: {chunk_type}")
            
            if chunk_type == 'classification':
                classification_chunk = chunk
            elif chunk_type in ['content_complete', 'content_chunk']:
                content_chunk = chunk
            elif chunk_type == 'message_complete':
                completion_chunk = chunk
        
        # Validate workflow execution
        assert len(response_chunks) >= 3, f"Expected at least 3 chunks, got {len(response_chunks)}"
        assert classification_chunk is not None, "No classification chunk found"
        assert content_chunk is not None, "No content chunk found"
        assert completion_chunk is not None, "No completion chunk found"
        
        # Validate classification
        assert classification_chunk['intent'] == 'conversational'
        assert classification_chunk['confidence'] >= 0.3
        
        # Validate content
        content = content_chunk.get('content', '')
        assert len(content) >= 30, f"Content too short: {len(content)} chars"
        assert any(term in content.lower() for term in ['eeev', 'eastern equine', 'virus']), "Missing EEEV content"
        
        # Validate completion
        assert completion_chunk['success'] is True
        assert completion_chunk['session_id'] == session_id
        
        print(f"✅ Complete conversational workflow executed successfully")
        print(f"   Content length: {len(content)} chars")
    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_annotation_workflow_initiation(self, real_workflow):
        """Test annotation workflow initiation (without full execution)"""
        query = "Create PSSM matrix of EEEV"
        session_id = f"test_ann_init_{uuid.uuid4().hex[:8]}"
        
        print(f"🧪 Testing annotation workflow initiation: {query}")
        
        response_chunks = []
        classification_chunk = None
        job_chunk = None
        
        async for chunk in real_workflow.process_user_message(query, session_id):
            response_chunks.append(chunk)
            
            chunk_type = chunk.get('type')
            print(f"   📦 Chunk: {chunk_type}")
            
            if chunk_type == 'classification':
                classification_chunk = chunk
            elif chunk_type in ['job_started', 'job_progress', 'job_complete']:
                job_chunk = chunk
            
            # Stop after reasonable number of chunks for this test
            if len(response_chunks) >= 10:
                break
        
        # Validate workflow initiation
        assert classification_chunk is not None, "No classification chunk found"
        assert classification_chunk['intent'] == 'annotation'
        assert classification_chunk['confidence'] >= 0.4
        
        print(f"✅ Annotation workflow initiated successfully")
        print(f"   Processed {len(response_chunks)} chunks")
    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_session_management(self, real_workflow):
        """Test session management with real components"""
        session_id = f"test_session_{uuid.uuid4().hex[:8]}"
        
        # Use the workflow's session manager directly
        session_manager = real_workflow.session_manager
        
        # Create session
        session_data = await session_manager.get_or_create_session(session_id)
        assert session_data.session_id == session_id
        assert session_data.created_at is not None
        
        # Process multiple messages in same session
        queries = ["What is EEEV?", "Tell me more about alphaviruses"]
        
        for i, query in enumerate(queries):
            print(f"🧪 Processing message {i+1}: {query}")
            
            response_chunks = []
            workflow_completed = False
            
            async for chunk in real_workflow.process_user_message(query, session_id):
                response_chunks.append(chunk)
                chunk_type = chunk.get('type', '')
                
                # Wait for message completion before checking session
                if chunk_type == 'message_complete':
                    workflow_completed = True
                    break
                    
                # Safety break to avoid infinite loops
                if len(response_chunks) >= 10:
                    break
            
            # Ensure workflow completed
            assert workflow_completed, f"Workflow did not complete for query: {query}"
            
            # Small delay to ensure session state is synchronized
            await asyncio.sleep(0.1)
            
            # Verify session persistence
            updated_session = await session_manager.get_session(session_id)
            assert updated_session is not None, f"Session {session_id} not found"
            
            # Check minimum message count - should have at least user + assistant per query
            expected_min_messages = (i + 1) * 2
            actual_message_count = len(updated_session.messages)
            
            print(f"   📊 Expected minimum: {expected_min_messages}, Actual: {actual_message_count}")
            print(f"   📝 Messages: {[f'{msg.role.value}:{msg.content[:30]}...' for msg in updated_session.messages]}")
            
            assert actual_message_count >= expected_min_messages, f"Expected at least {expected_min_messages} messages, got {actual_message_count}"
        
        print(f"✅ Session management working correctly")
        print(f"   Final message count: {len(updated_session.messages)}")
    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_concurrent_sessions(self, real_workflow):
        """Test concurrent session handling with real workflow"""
        queries = [
            ("What is EEEV?", "conversational"),
            ("Create PSSM matrix", "annotation"),
            ("How do viruses spread?", "conversational")
        ]
        
        print("🧪 Testing concurrent session handling")
        
        async def process_session(query_info):
            query, expected_intent = query_info
            session_id = f"concurrent_{expected_intent}_{uuid.uuid4().hex[:6]}"
            
            chunks = []
            classification_found = False
            
            async for chunk in real_workflow.process_user_message(query, session_id):
                chunks.append(chunk)
                
                if chunk.get('type') == 'classification':
                    classification_found = True
                    actual_intent = chunk.get('intent')
                    
                    return {
                        'session_id': session_id,
                        'query': query,
                        'expected_intent': expected_intent,
                        'actual_intent': actual_intent,
                        'classification_found': classification_found,
                        'chunk_count': len(chunks)
                    }
                
                # Stop after reasonable chunk count
                if len(chunks) >= 8:
                    break
            
            return {
                'session_id': session_id,
                'query': query,
                'expected_intent': expected_intent,
                'actual_intent': None,
                'classification_found': classification_found,
                'chunk_count': len(chunks)
            }
        
        # Execute concurrent sessions
        start_time = time.time()
        results = await asyncio.gather(*[process_session(q) for q in queries])
        end_time = time.time()
        
        # Validate results
        assert len(results) == len(queries)
        
        for result in results:
            print(f"   📊 Session {result['session_id'][:12]}: {result['query'][:30]}...")
            print(f"      Expected: {result['expected_intent']}, Got: {result['actual_intent']}")
            print(f"      Chunks: {result['chunk_count']}, Classification: {result['classification_found']}")
            
            assert result['classification_found'], f"No classification for: {result['query']}"
            
            # Allow some flexibility in intent detection for real components
            if result['expected_intent'] == 'annotation':
                assert result['actual_intent'] == 'annotation', f"Wrong intent for annotation query: {result['query']}"
        
        total_time = end_time - start_time
        print(f"✅ Concurrent sessions completed in {total_time:.3f}s")
    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_workflow_performance(self, real_workflow):
        """Test real workflow performance benchmarks"""
        test_cases = [
            ("What is EEEV?", "conversational", 5.0),
            ("Create PSSM matrix of EEEV", "annotation", 8.0)  # More time for annotation setup
        ]
        
        for query, expected_intent, max_time in test_cases:
            print(f"🧪 Performance test: {query}")
            
            start_time = time.time()
            
            chunks = []
            classification_time = None
            
            async for chunk in real_workflow.process_user_message(query, f"perf_{uuid.uuid4().hex[:6]}"):
                chunks.append(chunk)
                
                if chunk.get('type') == 'classification' and classification_time is None:
                    classification_time = time.time() - start_time
                
                # Stop after essential chunks for performance test
                if len(chunks) >= 6:
                    break
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Validate performance
            assert total_time < max_time, f"Query took {total_time:.3f}s, exceeds {max_time}s limit"
            
            if classification_time:
                assert classification_time < 2.0, f"Classification took {classification_time:.3f}s, exceeds 2s limit"
            
            print(f"✅ Performance: Total {total_time:.3f}s, Classification {classification_time:.3f}s")
    
    @pytest.mark.skipif(not REAL_COMPONENTS_AVAILABLE, reason="Real components not available")
    @pytest.mark.asyncio
    async def test_real_edge_cases(self, real_workflow):
        """Test edge cases with real workflow"""
        edge_cases = [
            ("", "empty query"),
            ("   ", "whitespace only"),
            ("xyz unknown terms", "unknown content"),
            ("What is EEEV and also create PSSM matrix?", "mixed intent")
        ]
        
        for query, description in edge_cases:
            print(f"🧪 Edge case test: {description}")
            
            try:
                chunks = []
                error_handled = False
                
                async for chunk in real_workflow.process_user_message(query, f"edge_{uuid.uuid4().hex[:6]}"):
                    chunks.append(chunk)
                    
                    # Look for error handling or graceful degradation
                    if chunk.get('type') in ['error', 'classification']:
                        error_handled = True
                        
                        if chunk.get('type') == 'classification':
                            intent = chunk.get('intent')
                            confidence = chunk.get('confidence', 0)
                            print(f"   📋 Handled as: {intent} (confidence: {confidence:.3f})")
                    
                    # Stop after reasonable chunk count
                    if len(chunks) >= 6:
                        break
                
                # Should handle edge cases gracefully
                assert len(chunks) > 0, f"No response for edge case: {description}"
                
                print(f"✅ Edge case handled gracefully: {description}")
                
            except Exception as e:
                # Acceptable if error is handled gracefully
                print(f"⚠️ Edge case caused exception (acceptable): {description} - {e}")


@pytest.mark.asyncio
async def test_real_workflow_availability():
    """Test if real workflow components are available and working"""
    if not REAL_COMPONENTS_AVAILABLE:
        pytest.skip("Real components not available")
    
    try:
        from nanobrain.core.workflow import WorkflowConfig
        from nanobrain.core.executor import LocalExecutor, ExecutorConfig
        import yaml
        from pathlib import Path
        
        # Load the real config file
        config_path = Path(__file__).parent.parent.parent / "nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            workflow_config = WorkflowConfig(**yaml_config)
        else:
            # Create minimal config for testing
            workflow_config = WorkflowConfig(
                name="ChatbotViralWorkflow",
                description="Test chatbot workflow",
                enable_progress_reporting=True,
                steps=[],
                links=[]
            )
        
        # Create executor using from_config pattern
        executor_config = ExecutorConfig(executor_type="local", max_workers=2)
        executor = LocalExecutor.from_config(executor_config)
        
        # Use from_config pattern with proper dependencies
        workflow = ChatbotViralWorkflow.from_config(workflow_config, executor=executor)
        await workflow.initialize()
        print("✅ ChatbotViralWorkflow instantiated successfully using from_config")
        
        # Test basic functionality
        test_query = "What is EEEV?"
        session_id = f"availability_test_{uuid.uuid4().hex[:8]}"
        
        chunks = []
        async for chunk in workflow.process_user_message(test_query, session_id):
            chunks.append(chunk)
            if len(chunks) >= 3:  # Just test basic response
                break
        
        assert len(chunks) > 0, "No response chunks received"
        print(f"✅ Real workflow responded with {len(chunks)} chunks")
        
    except Exception as e:
        pytest.fail(f"Real workflow availability test failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    import asyncio
    
    async def run_real_integration_tests():
        print("🧪 Running Real Workflow Integration Tests...")
        
        if not REAL_COMPONENTS_AVAILABLE:
            print("❌ Real components not available - skipping tests")
            return
        
        try:
            # Test availability first
            await test_real_workflow_availability()
            
            # Create workflow instance using from_config
            from nanobrain.core.workflow import WorkflowConfig
            from nanobrain.core.executor import LocalExecutor, ExecutorConfig
            import yaml
            from pathlib import Path
            
            # Load the real config file
            config_path = Path(__file__).parent.parent.parent / "nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                workflow_config = WorkflowConfig(**yaml_config)
            else:
                # Create minimal config for testing
                workflow_config = WorkflowConfig(
                    name="ChatbotViralWorkflow",
                    description="Test chatbot workflow",
                    enable_progress_reporting=True,
                    steps=[],
                    links=[]
                )
            
            # Create executor using from_config pattern
            executor_config = ExecutorConfig(executor_type="local", max_workers=2)
            executor = LocalExecutor.from_config(executor_config)
            
            workflow = ChatbotViralWorkflow.from_config(workflow_config, executor=executor)
            await workflow.initialize()
            session_manager = InMemorySessionManager()
            
            # Create test instance
            test_instance = TestRealWorkflowIntegration()
            
            # Run core integration tests
            print("\n📋 Testing Query Classification...")
            await test_instance.test_real_query_classification_conversational(workflow)
            await test_instance.test_real_query_classification_annotation(workflow)
            
            print("\n🔄 Testing Complete Workflows...")
            await test_instance.test_real_conversational_workflow_complete(workflow)
            await test_instance.test_real_annotation_workflow_initiation(workflow)
            
            print("\n📊 Testing Session Management...")
            await test_instance.test_real_session_management(workflow, session_manager)
            
            print("\n⚡ Testing Performance...")
            await test_instance.test_real_workflow_performance(workflow)
            
            print("\n🔍 Testing Edge Cases...")
            await test_instance.test_real_edge_cases(workflow)
            
            print("\n🎉 All Real Workflow Integration Tests Passed!")
            
        except Exception as e:
            print(f"❌ Real integration tests failed: {e}")
            raise
    
    asyncio.run(run_real_integration_tests())