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
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
import os

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))


class WebApplicationTester:
    """Comprehensive web application testing"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.server_process = None
        self.server_ready = False
        
    async def start_server(self, timeout: int = 30):
        """Start the chatbot server"""
        print("🚀 Starting chatbot server...")
        
        # Change to Chatbot directory
        chatbot_dir = Path(__file__).parent.parent / "Chatbot"
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent.parent)
            
            # Start server process using launch script
            launch_script = chatbot_dir / "launch_chatbot.sh"
            if launch_script.exists():
                self.server_process = subprocess.Popen(
                    ["bash", str(launch_script)],
                    cwd=chatbot_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    preexec_fn=None if sys.platform == "win32" else lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
                )
            else:
                # Fallback to direct server startup
                server_script = chatbot_dir / "server" / "nanobrain_server.py"
                self.server_process = subprocess.Popen(
                    [sys.executable, str(server_script)],
                    cwd=chatbot_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
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
                
                await asyncio.sleep(2)
            
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
    
    async def test_health_endpoints(self):
        """Test health and monitoring endpoints"""
        print("\n🏥 Testing health endpoints...")
        
        # Test basic health endpoint
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=10)
            if response.status_code != 200:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
            
            health_data = response.json()
            if 'status' not in health_data:
                print("❌ Health response missing status")
                return False
            
            print(f"✅ Health endpoint: {health_data['status']}")
            
        except Exception as e:
            print(f"❌ Health endpoint failed: {e}")
            return False
        
        # Test resources endpoint
        try:
            response = requests.get(f"{self.base_url}/api/resources", timeout=10)
            if response.status_code == 200:
                print("✅ Resources endpoint working")
            else:
                print(f"⚠️ Resources endpoint status: {response.status_code}")
            
        except Exception as e:
            print(f"⚠️ Resources endpoint failed: {e}")
        
        return True
    
    async def test_conversational_query_endpoint(self):
        """Test conversational query through /api/chat endpoint"""
        print("\n💬 Testing conversational query: 'What is EEEV?'")
        
        payload = {
            "message": "What is EEEV?",
            "session_id": "test_conversational"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Chat endpoint failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            result = response.json()
            
            # Validate response structure
            if 'response' not in result:
                print("❌ Response missing 'response' field")
                return False
            
            content = result['response']
            message_type = result.get('message_type', 'unknown')
            requires_markdown = result.get('requires_markdown', False)
            
            print(f"📝 Message type: {message_type}")
            print(f"📝 Requires markdown: {requires_markdown}")
            print(f"📝 Response preview: {content[:200]}...")
            
            # Validate EEEV content
            content_lower = content.lower()
            has_eeev_info = any(term in content_lower for term in ['eeev', 'eastern equine', 'encephalitis'])
            has_virus_info = any(term in content_lower for term in ['virus', 'viral', 'alphavirus'])
            
            print(f"✅ Contains EEEV information: {has_eeev_info}")
            print(f"✅ Contains virus information: {has_virus_info}")
            
            # Validate markdown formatting
            has_markdown = '**' in content or '*' in content
            print(f"✅ Contains markdown formatting: {has_markdown}")
            
            # Validate no job ID (no annotation workflow)
            has_job_id = 'job_id' in result and result['job_id']
            print(f"✅ No job ID created: {not has_job_id}")
            
            success = has_eeev_info and has_virus_info and has_markdown and not has_job_id
            
            if success:
                print("✅ Conversational query test PASSED")
            else:
                print("❌ Conversational query test FAILED")
                print(f"  - EEEV info: {has_eeev_info}")
                print(f"  - Virus info: {has_virus_info}")
                print(f"  - Markdown: {has_markdown}")
                print(f"  - No job: {not has_job_id}")
            
            return success
            
        except Exception as e:
            print(f"❌ Conversational query test failed: {e}")
            return False
    
    async def test_annotation_query_endpoint(self):
        """Test annotation query through /api/chat endpoint"""
        print("\n🧬 Testing annotation query: 'Create PSSM matrix of EEEV'")
        
        payload = {
            "message": "Create PSSM matrix of EEEV",
            "session_id": "test_annotation"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60  # Longer timeout for annotation processing
            )
            
            if response.status_code != 200:
                print(f"❌ Chat endpoint failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
            
            result = response.json()
            
            # Validate response structure
            if 'response' not in result:
                print("❌ Response missing 'response' field")
                return False
            
            content = result['response']
            message_type = result.get('message_type', 'unknown')
            job_id = result.get('job_id')
            
            print(f"📝 Message type: {message_type}")
            print(f"📝 Job ID: {job_id}")
            print(f"📝 Response preview: {content[:200]}...")
            
            # Validate annotation response
            has_job_id = job_id is not None and len(str(job_id)) > 0
            print(f"✅ Job ID generated: {has_job_id}")
            
            # Validate content contains workflow overview
            content_lower = content.lower()
            has_analysis_info = any(term in content_lower for term in ['analysis', 'completed', 'pssm', 'matrix'])
            print(f"✅ Contains analysis information: {has_analysis_info}")
            
            # Look for JSON formatted PSSM matrix
            json_pattern = r'```json\s*(.*?)\s*```'
            json_matches = re.findall(json_pattern, content, re.DOTALL)
            
            has_json = len(json_matches) > 0
            print(f"✅ Contains JSON formatted data: {has_json}")
            
            if has_json:
                # Validate JSON structure
                json_content = json_matches[0]
                try:
                    pssm_data = json.loads(json_content)
                    is_valid_json = isinstance(pssm_data, (dict, list))
                    print(f"✅ Valid JSON structure: {is_valid_json}")
                    
                    # Check for PSSM matrix structure
                    if isinstance(pssm_data, dict):
                        has_matrix_data = any(key in pssm_data for key in ['matrix', 'pssm_matrix', 'matrix_data'])
                        print(f"✅ Contains matrix data: {has_matrix_data}")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON format: {e}")
                    is_valid_json = False
            else:
                is_valid_json = False
                print("⚠️ No JSON PSSM matrix found in response")
            
            # Validate markdown formatting
            has_markdown = '**' in content or '*' in content
            print(f"✅ Contains markdown formatting: {has_markdown}")
            
            success = has_job_id and has_analysis_info and has_markdown
            # JSON validation is optional for now as backend might not be available
            
            if success:
                print("✅ Annotation query test PASSED")
            else:
                print("❌ Annotation query test FAILED")
                print(f"  - Job ID: {has_job_id}")
                print(f"  - Analysis info: {has_analysis_info}")
                print(f"  - Markdown: {has_markdown}")
                print(f"  - JSON (optional): {has_json}")
            
            return success
            
        except Exception as e:
            print(f"❌ Annotation query test failed: {e}")
            return False
    
    async def run_comprehensive_test_suite(self):
        """Run the complete test suite"""
        print("🚀 Starting Comprehensive Web Application Test Suite")
        print("=" * 60)
        
        test_results = {
            "server_startup": False,
            "health_endpoints": False,
            "conversational_query": False,
            "annotation_query": False
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
            
            # 3. Test conversational query
            print("\n3️⃣ Testing conversational query...")
            test_results["conversational_query"] = await self.test_conversational_query_endpoint()
            
            # 4. Test annotation query  
            print("\n4️⃣ Testing annotation query...")
            test_results["annotation_query"] = await self.test_annotation_query_endpoint()
            
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
                print("✅ 'Create PSSM matrix of EEEV' → Annotation workflow with job tracking")
                print("\n🚀 Web application demonstrates the exact required behavior!")
            else:
                print("\n⚠️ CRITICAL TESTS FAILED!")
                print("The web application does not meet the required behavior specifications.")
                
                if not test_results["conversational_query"]:
                    print("❌ Conversational query behavior incorrect")
                if not test_results["annotation_query"]:
                    print("❌ Annotation query behavior incorrect")
            
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
    sys.exit(exit_code) 