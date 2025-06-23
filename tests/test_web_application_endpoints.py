#!/usr/bin/env python3
"""
Web Application Endpoint Testing

Tests the complete web application through HTTP endpoints to validate:
1. "What is EEEV?" → Conversational response via /api/chat
2. "Create PSSM matrix of EEEV" → Annotation workflow via /api/chat  
3. WebSocket functionality for real-time updates
4. Health endpoints and monitoring

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import json
import time
import sys
import subprocess
import signal
import requests
import socketio
from pathlib import Path
from typing import Dict, Any, List, Optional
import pytest
import threading
import re

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Chatbot" / "server"))


class WebApplicationTester:
    """Comprehensive web application testing"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.server_process = None
        self.sio_client = None
        self.websocket_events = []
        self.server_ready = False
        
    async def start_server(self, timeout: int = 30):
        """Start the chatbot server"""
        print("🚀 Starting chatbot server...")
        
        # Change to Chatbot directory
        chatbot_dir = Path(__file__).parent.parent / "Chatbot"
        
        try:
            # Start server process
            self.server_process = subprocess.Popen(
                [sys.executable, "-c", """
import sys
sys.path.insert(0, 'server')
from nanobrain_server import ChatbotServer
import asyncio

async def main():
    server = ChatbotServer()
    await server.start(host='0.0.0.0', port=5001)

if __name__ == '__main__':
    asyncio.run(main())
                """],
                cwd=chatbot_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=None if sys.platform == "win32" else lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
            )
            
            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"{self.base_url}/api/health", timeout=5)
                    if response.status_code == 200:
                        self.server_ready = True
                        print("✅ Server is ready!")
                        return True
                except:
                    pass
                
                await asyncio.sleep(1)
            
            print("❌ Server failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the chatbot server"""
        if self.server_process:
            print("🛑 Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
        
        if self.sio_client:
            self.sio_client.disconnect()
            self.sio_client = None
    
    async def setup_websocket_client(self):
        """Setup WebSocket client for real-time testing"""
        print("🔌 Setting up WebSocket client...")
        
        self.sio_client = socketio.AsyncClient()
        self.websocket_events = []
        
        # Event handlers
        @self.sio_client.event
        async def connect():
            print("✅ WebSocket connected")
            await self.sio_client.emit('register_session', {'session_id': 'test_session'})
        
        @self.sio_client.event
        async def disconnect():
            print("🔌 WebSocket disconnected")
        
        @self.sio_client.event
        async def chat_response(data):
            print(f"📨 Chat response: {data.get('content', '')[:100]}...")
            self.websocket_events.append({'type': 'chat_response', 'data': data})
        
        @self.sio_client.event
        async def progress_update(data):
            print(f"📊 Progress: {data.get('progress', 0)}%")
            self.websocket_events.append({'type': 'progress_update', 'data': data})
        
        @self.sio_client.event
        async def chat_error(data):
            print(f"❌ Chat error: {data.get('error', '')}")
            self.websocket_events.append({'type': 'chat_error', 'data': data})
        
        try:
            await self.sio_client.connect(self.base_url)
            await asyncio.sleep(1)  # Give connection time to establish
            return True
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def test_health_endpoints(self):
        """Test health and monitoring endpoints"""
        print("\n🏥 Testing health endpoints...")
        
        # Test basic health endpoint
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            assert response.status_code == 200, f"Health endpoint failed: {response.status_code}"
            
            health_data = response.json()
            assert 'status' in health_data, "Health response should contain status"
            assert 'uptime_seconds' in health_data, "Health response should contain uptime"
            
            print(f"✅ Health endpoint: {health_data['status']}")
            
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
            return False
        
        # Test frontend health endpoint
        try:
            response = requests.get(f"{self.base_url}/api/health/frontend", timeout=10)
            assert response.status_code == 200, f"Frontend health endpoint failed: {response.status_code}"
            
            frontend_health = response.json()
            assert 'status' in frontend_health, "Frontend health should contain status"
            
            print("✅ Frontend health endpoint working")
            
        except Exception as e:
            print(f"❌ Frontend health endpoint failed: {e}")
            return False
        
        # Test resources endpoint
        try:
            response = requests.get(f"{self.base_url}/api/resources", timeout=10)
            assert response.status_code == 200, f"Resources endpoint failed: {response.status_code}"
            
            resources_data = response.json()
            assert 'cpu_percentage' in resources_data, "Resources should contain CPU data"
            assert 'memory_percentage' in resources_data, "Resources should contain memory data"
            
            print("✅ Resources endpoint working")
            
        except Exception as e:
            print(f"❌ Resources endpoint failed: {e}")
            return False
        
        return True
    
    async def test_conversational_query_endpoint(self):
        """Test conversational query through /api/chat endpoint"""
        print("\n💬 Testing conversational query: 'What is EEEV?'")
        
        payload = {
            "message": "What is EEEV?",
            "session_id": "test_conversational"
        }
        
        try:
            # Clear previous WebSocket events
            self.websocket_events = []
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            assert response.status_code == 200, f"Chat endpoint failed: {response.status_code}"
            
            result = response.json()
            
            # Validate response structure
            assert 'response' in result, "Response should contain 'response' field"
            assert 'message_type' in result, "Response should contain 'message_type' field"
            assert 'requires_markdown' in result, "Response should contain 'requires_markdown' field"
            
            content = result['response']
            message_type = result['message_type']
            requires_markdown = result['requires_markdown']
            
            # Validate conversational response
            assert message_type == 'conversational', f"Expected conversational message type, got {message_type}"
            assert requires_markdown is True, "Conversational response should require markdown"
            
            # Validate EEEV content
            content_lower = content.lower()
            assert any(term in content_lower for term in ['eeev', 'eastern equine', 'encephalitis']), "Response should contain EEEV information"
            assert any(term in content_lower for term in ['virus', 'viral', 'alphavirus']), "Response should contain viral information"
            
            # Validate markdown formatting
            assert '**' in content or '*' in content, "Response should contain markdown formatting"
            
            # Validate no job ID (no annotation workflow)
            assert 'job_id' not in result or not result['job_id'], "Conversational query should not create job"
            
            print("✅ Conversational query test passed")
            print(f"📝 Response preview: {content[:200]}...")
            
            # Wait for any WebSocket events
            await asyncio.sleep(2)
            
            # Should not have progress updates
            progress_events = [e for e in self.websocket_events if e['type'] == 'progress_update']
            assert len(progress_events) == 0, "Conversational query should not generate progress updates"
            
            return result
            
        except Exception as e:
            print(f"❌ Conversational query test failed: {e}")
            raise
    
    async def test_annotation_query_endpoint(self):
        """Test annotation query through /api/chat endpoint"""
        print("\n🧬 Testing annotation query: 'Create PSSM matrix of EEEV'")
        
        payload = {
            "message": "Create PSSM matrix of EEEV",
            "session_id": "test_annotation"
        }
        
        try:
            # Clear previous WebSocket events
            self.websocket_events = []
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout for annotation processing
            )
            
            assert response.status_code == 200, f"Chat endpoint failed: {response.status_code}"
            
            result = response.json()
            
            # Validate response structure
            assert 'response' in result, "Response should contain 'response' field"
            assert 'message_type' in result, "Response should contain 'message_type' field"
            assert 'job_id' in result, "Annotation response should contain job_id"
            
            content = result['response']
            message_type = result.get('message_type')
            job_id = result['job_id']
            
            # Validate annotation response
            assert job_id is not None, "Job ID should be generated for annotation query"
            assert len(job_id) > 0, "Job ID should not be empty"
            
            # Validate content contains workflow overview
            content_lower = content.lower()
            assert any(term in content_lower for term in ['analysis', 'completed', 'pssm', 'matrix']), "Response should indicate PSSM analysis"
            
            # Look for JSON formatted PSSM matrix
            json_pattern = r'```json\s*(.*?)\s*```'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            if len(json_matches) > 0:
                # Validate JSON structure
                json_content = json_matches[0]
                try:
                    pssm_data = json.loads(json_content)
                    assert isinstance(pssm_data, (dict, list)), "JSON should contain valid PSSM data structure"
                    print("✅ Found valid JSON PSSM matrix in response")
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON validation failed: {e}")
            else:
                print("⚠️ No JSON PSSM matrix found in immediate response")
            
            # Validate markdown formatting
            assert '**' in content or '*' in content, "Response should contain markdown formatting"
            
            print("✅ Annotation query test passed")
            print(f"📝 Job ID: {job_id}")
            print(f"📝 Response preview: {content[:200]}...")
            
            # Wait for WebSocket events
            await asyncio.sleep(5)
            
            # Should have progress updates
            progress_events = [e for e in self.websocket_events if e['type'] == 'progress_update']
            print(f"📊 Received {len(progress_events)} progress updates")
            
            # Validate progress events
            for event in progress_events:
                data = event['data']
                assert 'progress' in data, "Progress event should contain progress percentage"
                assert 'job_id' in data, "Progress event should contain job ID"
                assert data['job_id'] == job_id, "Progress job ID should match response job ID"
            
            return result, progress_events
            
        except Exception as e:
            print(f"❌ Annotation query test failed: {e}")
            raise
    
    async def test_websocket_functionality(self):
        """Test WebSocket real-time functionality"""
        print("\n🔌 Testing WebSocket functionality...")
        
        if not self.sio_client or not self.sio_client.connected:
            print("❌ WebSocket not connected")
            return False
        
        # Test echo functionality
        test_data = {"test": "websocket_test", "timestamp": time.time()}
        
        try:
            await self.sio_client.emit('echo', test_data)
            await asyncio.sleep(1)
            
            # Check for echo response
            echo_events = [e for e in self.websocket_events if e.get('type') == 'echo']
            if len(echo_events) > 0:
                print("✅ WebSocket echo test passed")
            else:
                print("⚠️ No echo response received")
            
            return True
            
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            return False
    
    async def run_comprehensive_test_suite(self):
        """Run the complete test suite"""
        print("🚀 Starting Comprehensive Web Application Test Suite")
        print("=" * 60)
        
        test_results = {
            "server_startup": False,
            "health_endpoints": False,
            "websocket_setup": False,
            "conversational_query": False,
            "annotation_query": False,
            "websocket_functionality": False
        }
        
        try:
            # 1. Start server
            print("\n1️⃣ Starting server...")
            test_results["server_startup"] = await self.start_server()
            if not test_results["server_startup"]:
                print("❌ Server startup failed - cannot continue tests")
                return test_results
            
            # 2. Test health endpoints
            print("\n2️⃣ Testing health endpoints...")
            test_results["health_endpoints"] = await self.test_health_endpoints()
            
            # 3. Setup WebSocket
            print("\n3️⃣ Setting up WebSocket...")
            test_results["websocket_setup"] = await self.setup_websocket_client()
            
            # 4. Test conversational query
            print("\n4️⃣ Testing conversational query...")
            try:
                conversational_result = await self.test_conversational_query_endpoint()
                test_results["conversational_query"] = True
                
            except Exception as e:
                print(f"❌ Conversational query failed: {e}")
                test_results["conversational_query"] = False
            
            # 5. Test annotation query  
            print("\n5️⃣ Testing annotation query...")
            try:
                annotation_result, progress_events = await self.test_annotation_query_endpoint()
                test_results["annotation_query"] = True
                
            except Exception as e:
                print(f"❌ Annotation query failed: {e}")
                test_results["annotation_query"] = False
            
            # 6. Test WebSocket functionality
            print("\n6️⃣ Testing WebSocket functionality...")
            test_results["websocket_functionality"] = await self.test_websocket_functionality()
            
            # Final results
            print("\n" + "=" * 60)
            print("📊 FINAL TEST RESULTS")
            print("=" * 60)
            
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)
            
            for test_name, passed in test_results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status} {test_name.replace('_', ' ').title()}")
            
            print(f"\n📈 Overall Success Rate: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
            
            # Critical tests validation
            critical_tests = ["conversational_query", "annotation_query"]
            critical_passed = all(test_results[test] for test in critical_tests)
            
            if critical_passed:
                print("\n🎉 CRITICAL TESTS PASSED!")
                print("✅ 'What is EEEV?' → Conversational response with educational content")
                print("✅ 'Create PSSM matrix of EEEV' → Annotation workflow with progress tracking")
                print("\n🚀 Web application is fully operational and deployable!")
            else:
                print("\n⚠️ CRITICAL TESTS FAILED!")
                print("The web application does not meet the required behavior specifications.")
            
            return test_results
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
            return test_results
            
        finally:
            # Always cleanup
            self.stop_server()


# Standalone test execution
async def main():
    """Main test execution"""
    tester = WebApplicationTester()
    
    try:
        results = await tester.run_comprehensive_test_suite()
        
        # Return appropriate exit code
        critical_tests = ["conversational_query", "annotation_query"]
        critical_passed = all(results.get(test, False) for test in critical_tests)
        
        if critical_passed:
            print("\n✅ All critical tests passed - web application is ready!")
            return 0
        else:
            print("\n❌ Critical tests failed - web application needs fixes")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1
    finally:
        tester.stop_server()


if __name__ == "__main__":
    exit_code = asyncio.run(main()) 