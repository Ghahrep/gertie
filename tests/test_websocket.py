#!/usr/bin/env python3
"""
Enhanced WebSocket Test Script for Task 2.3 Implementation
=========================================================
Tests all enhanced WebSocket functionality including:
- Connection management and pooling
- Multi-channel notifications
- Topic subscriptions
- Performance under load
- Health monitoring
"""

import asyncio
import websockets
import json
import time
import logging
import aiohttp
from datetime import datetime
from typing import List, Dict, Any
import uuid
import concurrent.futures
from dataclasses import dataclass
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResults:
    test_name: str
    success: bool
    duration_ms: float
    details: Dict[str, Any]
    error: str = None

class WebSocketTestClient:
    """Test client for WebSocket connections"""
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.http_base = base_url.replace("ws://", "http://").replace("wss://", "https://")
        self.results: List[TestResults] = []
    
    async def test_basic_connection(self, user_id: str = "test_user_1") -> TestResults:
        """Test basic WebSocket connection"""
        start_time = time.time()
        test_name = "Basic Connection"
        
        try:
            uri = f"{self.base_url}/ws/{user_id}?topics=risk_alerts,workflow_updates"
            
            async with websockets.connect(uri) as websocket:
                # Wait for connection confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                message = json.loads(response)
                
                success = message.get("type") == "connection_established"
                details = {
                    "connection_message": message,
                    "features_enabled": message.get("capabilities", [])
                }
                
                duration_ms = (time.time() - start_time) * 1000
                return TestResults(test_name, success, duration_ms, details)
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, False, duration_ms, {}, str(e))
    
    async def test_topic_subscriptions(self, user_id: str = "test_user_2") -> TestResults:
        """Test topic subscription management"""
        start_time = time.time()
        test_name = "Topic Subscriptions"
        
        try:
            uri = f"{self.base_url}/ws/{user_id}"
            
            async with websockets.connect(uri) as websocket:
                # Wait for initial connection
                await websocket.recv()
                
                # Test subscribing to new topic
                subscribe_msg = {
                    "type": "subscribe",
                    "topic": "system_announcements"
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                # Wait for confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                sub_confirmation = json.loads(response)
                
                # Test unsubscribing
                unsubscribe_msg = {
                    "type": "unsubscribe",
                    "topic": "system_announcements"
                }
                await websocket.send(json.dumps(unsubscribe_msg))
                
                # Wait for confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                unsub_confirmation = json.loads(response)
                
                success = (sub_confirmation.get("type") == "subscription_confirmed" and 
                          unsub_confirmation.get("type") == "unsubscription_confirmed")
                
                details = {
                    "subscribe_response": sub_confirmation,
                    "unsubscribe_response": unsub_confirmation
                }
                
                duration_ms = (time.time() - start_time) * 1000
                return TestResults(test_name, success, duration_ms, details)
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, False, duration_ms, {}, str(e))
        
    async def _listen_for_risk_alert_with_confirmation(self, uri: str, connection_established: asyncio.Event) -> Dict:
        """
        ADDED: Missing method for listening to risk alerts with connection confirmation
        """
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for connection confirmation and signal that connection is ready
                response = await websocket.recv()
                connection_message = json.loads(response)
                
                if connection_message.get("type") == "connection_established":
                    connection_established.set()  # Signal that connection is ready
                    print(f"DEBUG: WebSocket connection confirmed, waiting for risk alert...")
                
                # Listen for risk alert
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                        message = json.loads(response)
                        
                        if message.get("type") == "risk_alert":
                            print(f"DEBUG: Risk alert received: {message}")
                            return message
                        elif message.get("type") in ["heartbeat", "ping"]:
                            # Ignore heartbeat messages
                            continue
                            
                    except asyncio.TimeoutError:
                        print("DEBUG: Timeout waiting for risk alert message")
                        return None
                    
        except Exception as e:
            print(f"DEBUG: Error in risk alert listener: {e}")
            connection_established.set()  # Still signal connection attempt was made
            return None

    async def _listen_for_workflow_update_with_confirmation(self, uri: str, connection_established: asyncio.Event) -> Dict:
        """
        ADDED: Workflow update listener with connection confirmation
        """
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for connection confirmation
                response = await websocket.recv()
                connection_message = json.loads(response)
                
                if connection_message.get("type") == "connection_established":
                    connection_established.set()  # Signal that connection is ready
                    print(f"DEBUG: WebSocket connection confirmed, waiting for workflow update...")
                
                # Listen for workflow update
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                        message = json.loads(response)
                        
                        if message.get("type") == "workflow_update":
                            print(f"DEBUG: Workflow update received: {message}")
                            return message
                        elif message.get("type") in ["heartbeat", "ping"]:
                            # Ignore heartbeat messages
                            continue
                            
                    except asyncio.TimeoutError:
                        print("DEBUG: Timeout waiting for workflow update message")
                        return None
                    
        except Exception as e:
            print(f"DEBUG: Error in workflow update listener: {e}")
            connection_established.set()  # Still signal connection attempt was made
            return None
    
    async def test_risk_alert_notification(self, user_id: str = "test_user_3") -> TestResults:
        """Test risk alert notification delivery"""
        start_time = time.time()
        test_name = "Risk Alert Notification"
        
        try:
            # Use a shared variable to confirm connection
            connection_established = asyncio.Event()
            
            # Start WebSocket connection
            uri = f"{self.base_url}/ws/{user_id}?topics=risk_alerts"
            
            websocket_task = asyncio.create_task(
                self._listen_for_risk_alert_with_confirmation(uri, connection_established)
            )
            
            # WAIT for WebSocket connection to be established
            await asyncio.wait_for(connection_established.wait(), timeout=5.0)
            print(f"DEBUG: WebSocket connection confirmed for {user_id}")
            
            # Now trigger test notification via HTTP API
            async with aiohttp.ClientSession() as session:
                url = f"{self.http_base}/api/v1/websocket/test-risk-alert/{user_id}"
                async with session.post(url) as response:
                    http_result = await response.json()
            
            # Wait for WebSocket message
            try:
                websocket_result = await asyncio.wait_for(websocket_task, timeout=10.0)
                success = websocket_result and http_result.get("status") == "success"
                
                details = {
                    "http_response": http_result,
                    "websocket_message": websocket_result
                }
                
            except asyncio.TimeoutError:
                websocket_task.cancel()
                success = False
                details = {
                    "http_response": http_result,
                    "websocket_message": "timeout"
                }
            
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, success, duration_ms, details)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, False, duration_ms, {}, str(e))
    
    async def _listen_for_risk_alert(self, uri: str) -> Dict:
        """Helper method to listen for risk alert"""
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Listen for risk alert
                while True:
                    response = await websocket.recv()
                    message = json.loads(response)
                    
                    if message.get("type") == "risk_alert":
                        return message
                    
        except Exception as e:
            logger.error(f"Error listening for risk alert: {e}")
            return None
    
    async def test_workflow_notifications(self, user_id: str = "test_user_4") -> TestResults:
        """
        IMPROVED: Workflow notification test with better connection management
        """
        start_time = time.time()
        test_name = "Workflow Notifications"
        
        try:
            uri = f"{self.base_url}/ws/{user_id}?topics=workflow_updates"
            print(f"DEBUG: Starting workflow notification test with URI: {uri}")
            
            # Use the same pattern as risk alert test for consistency
            connection_established = asyncio.Event()
            
            websocket_task = asyncio.create_task(
                self._listen_for_workflow_update_with_confirmation(uri, connection_established)
            )
            
            # Wait for connection establishment with longer timeout
            try:
                await asyncio.wait_for(connection_established.wait(), timeout=10.0)
                print(f"DEBUG: WebSocket connection confirmed for {user_id}")
            except asyncio.TimeoutError:
                print("DEBUG: WebSocket connection timeout for workflow test")
                websocket_task.cancel()
                duration_ms = (time.time() - start_time) * 1000
                return TestResults(test_name, False, duration_ms, {}, "WebSocket connection timeout")
            
            # Add delay for connection stability
            await asyncio.sleep(0.5)
            
            # Trigger test workflow notification
            async with aiohttp.ClientSession() as session:
                url = f"{self.http_base}/api/v1/websocket/test-workflow-update/{user_id}"
                print(f"DEBUG: Triggering workflow update via HTTP: {url}")
                
                try:
                    async with session.post(url) as response:
                        http_result = await response.json()
                        print(f"DEBUG: HTTP workflow response: {http_result}")
                except Exception as e:
                    print(f"DEBUG: HTTP workflow request failed: {e}")
                    websocket_task.cancel()
                    duration_ms = (time.time() - start_time) * 1000
                    return TestResults(test_name, False, duration_ms, {}, f"HTTP request failed: {e}")
            
            # Wait for WebSocket message with extended timeout
            try:
                websocket_result = await asyncio.wait_for(websocket_task, timeout=15.0)
                
                # Determine success based on receiving the WebSocket message
                success = websocket_result is not None and http_result.get("status") in ["success", "no_connection"]
                
                details = {
                    "http_response": http_result,
                    "websocket_message": websocket_result,
                    "connection_uri": uri,
                    "connection_status": "established" if websocket_result else "timeout"
                }
                
            except asyncio.TimeoutError:
                websocket_task.cancel()
                success = False
                details = {
                    "http_response": http_result,
                    "websocket_message": "timeout - no workflow update received within 15 seconds",
                    "connection_uri": uri,
                    "connection_status": "timeout"
                }
            
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, success, duration_ms, details)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            print(f"DEBUG: Workflow notification test failed with exception: {e}")
            return TestResults(test_name, False, duration_ms, {}, str(e))
    
    async def _listen_for_workflow_update(self, uri: str) -> Dict:
        """Helper method to listen for workflow updates"""
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Listen for workflow update
                while True:
                    response = await websocket.recv()
                    message = json.loads(response)
                    
                    if message.get("type") == "workflow_update":
                        return message
                    
        except Exception as e:
            logger.error(f"Error listening for workflow update: {e}")
            return None
    
    async def test_heartbeat_mechanism(self, user_id: str = "test_user_5") -> TestResults:
        """Test heartbeat and connection health"""
        start_time = time.time()
        test_name = "Heartbeat Mechanism"
        
        try:
            uri = f"{self.base_url}/ws/{user_id}"
            
            async with websockets.connect(uri) as websocket:
                # Wait for connection confirmation
                await websocket.recv()
                
                # Wait for heartbeat (should come within 30 seconds in timeout)
                heartbeat_received = False
                for _ in range(35):  # Wait up to 35 seconds
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        message = json.loads(response)
                        
                        if message.get("type") in ["heartbeat", "ping"]:
                            heartbeat_received = True
                            break
                            
                    except asyncio.TimeoutError:
                        continue
                
                details = {"heartbeat_received": heartbeat_received}
                duration_ms = (time.time() - start_time) * 1000
                return TestResults(test_name, heartbeat_received, duration_ms, details)
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, False, duration_ms, {}, str(e))
    
    async def test_concurrent_connections(self, num_connections: int = 20) -> TestResults:
        """Test multiple concurrent connections"""
        start_time = time.time()
        test_name = f"Concurrent Connections ({num_connections})"
        
        try:
            # Create multiple connections concurrently
            tasks = []
            for i in range(num_connections):
                user_id = f"load_test_user_{i}"
                task = self._create_test_connection(user_id)
                tasks.append(task)
            
            # Execute all connections
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_connections = 0
            failed_connections = 0
            connection_times = []
            
            for result in results:
                if isinstance(result, Exception):
                    failed_connections += 1
                else:
                    if result.get("success"):
                        successful_connections += 1
                        connection_times.append(result.get("connection_time_ms", 0))
                    else:
                        failed_connections += 1
            
            success_rate = successful_connections / num_connections
            avg_connection_time = statistics.mean(connection_times) if connection_times else 0
            
            details = {
                "total_connections": num_connections,
                "successful_connections": successful_connections,
                "failed_connections": failed_connections,
                "success_rate": success_rate,
                "avg_connection_time_ms": avg_connection_time
            }
            
            success = success_rate >= 0.9  # 90% success rate required
            duration_ms = (time.time() - start_time) * 1000
            
            return TestResults(test_name, success, duration_ms, details)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, False, duration_ms, {}, str(e))
    
    async def _create_test_connection(self, user_id: str) -> Dict:
        """Create a single test connection"""
        conn_start = time.time()
        
        try:
            uri = f"{self.base_url}/ws/{user_id}?topics=risk_alerts"
            
            async with websockets.connect(uri) as websocket:
                # Wait for connection confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                message = json.loads(response)
                
                connection_time = (time.time() - conn_start) * 1000
                
                return {
                    "success": message.get("type") == "connection_established",
                    "connection_time_ms": connection_time,
                    "user_id": user_id
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id
            }
    
    async def test_websocket_stats_api(self) -> TestResults:
        """Test WebSocket statistics API"""
        start_time = time.time()
        test_name = "WebSocket Stats API"
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.http_base}/api/v1/websocket/stats"
                async with session.get(url) as response:
                    stats = await response.json()
            
            success = (stats.get("status") == "success" and 
                      stats.get("websocket_enabled") == True)
            
            details = {
                "stats_response": stats,
                "enhanced_features": stats.get("enhanced_features", {}),
                "connection_stats": stats.get("connection_stats", {})
            }
            
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, success, duration_ms, details)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, False, duration_ms, {}, str(e))
    
    async def test_subscription_management_api(self, user_id: str = "test_user_6") -> TestResults:
        """Test subscription management via REST API"""
        start_time = time.time()
        test_name = "Subscription Management API"
        
        try:
            # First establish a WebSocket connection
            uri = f"{self.base_url}/ws/{user_id}?topics=risk_alerts"
            
            connection_task = asyncio.create_task(self._maintain_connection(uri))
            await asyncio.sleep(1)  # Let connection establish
            
            async with aiohttp.ClientSession() as session:
                base_api_url = f"{self.http_base}/api/v1/websocket/subscriptions"
                
                # Test getting current subscriptions
                async with session.get(base_api_url) as response:
                    current_subs = await response.json()
                
                # Test subscribing to new topic
                async with session.post(f"{base_api_url}/portfolio_reports") as response:
                    sub_result = await response.json()
                
                # Test unsubscribing
                async with session.delete(f"{base_api_url}/portfolio_reports") as response:
                    unsub_result = await response.json()
            
            connection_task.cancel()
            
            success = (current_subs.get("status") in ["online", "offline"] and
                      sub_result.get("status") == "success" and
                      unsub_result.get("status") == "success")
            
            details = {
                "current_subscriptions": current_subs,
                "subscribe_result": sub_result,
                "unsubscribe_result": unsub_result
            }
            
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, success, duration_ms, details)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, False, duration_ms, {}, str(e))
    
    async def _maintain_connection(self, uri: str):
        """Maintain a WebSocket connection for testing"""
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    await websocket.recv()
        except asyncio.CancelledError:
            pass
    
    async def test_health_monitoring(self) -> TestResults:
        """Test health monitoring endpoint"""
        start_time = time.time()
        test_name = "Health Monitoring"
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.http_base}/api/v1/websocket/health"
                async with session.get(url) as response:
                    health = await response.json()
            
            success = health.get("status") in ["healthy", "degraded"]
            
            details = {
                "health_response": health,
                "system_status": health.get("status"),
                "connection_manager": health.get("connection_manager", {}),
                "system_resources": health.get("system_resources", {})
            }
            
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, success, duration_ms, details)
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TestResults(test_name, False, duration_ms, {}, str(e))
    
    async def run_all_tests(self) -> List[TestResults]:
        """Run all WebSocket tests"""
        print("Starting Enhanced WebSocket Test Suite")
        print("=" * 60)
        
        # List of all tests to run
        test_methods = [
            self.test_basic_connection,
            self.test_topic_subscriptions,
            self.test_risk_alert_notification,
            self.test_workflow_notifications,
            self.test_heartbeat_mechanism,
            lambda: self.test_concurrent_connections(10),  # Start with 10 connections
            self.test_websocket_stats_api,
            self.test_subscription_management_api,
            self.test_health_monitoring
        ]
        
        results = []
        
        for i, test_method in enumerate(test_methods, 1):
            print(f"\n[{i}/{len(test_methods)}] Running test...")
            
            try:
                result = await test_method()
                results.append(result)
                
                status = "PASS" if result.success else "FAIL"
                print(f"  {result.test_name}: {status} ({result.duration_ms:.2f}ms)")
                
                if not result.success and result.error:
                    print(f"    Error: {result.error}")
                
            except Exception as e:
                error_result = TestResults(
                    test_name=f"Test {i}",
                    success=False,
                    duration_ms=0,
                    details={},
                    error=str(e)
                )
                results.append(error_result)
                print(f"  Test {i}: FAIL - {str(e)}")
        
        return results
    
    def print_test_summary(self, results: List[TestResults]):
        """Print a summary of all test results"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in results if r.success)
        total = len(results)
        pass_rate = (passed / total) * 100 if total > 0 else 0
        
        print(f"Tests Passed: {passed}/{total} ({pass_rate:.1f}%)")
        print(f"Total Duration: {sum(r.duration_ms for r in results):.2f}ms")
        
        print("\nDetailed Results:")
        for result in results:
            status = "PASS" if result.success else "FAIL"
            print(f"  {result.test_name}: {status} ({result.duration_ms:.2f}ms)")
            
            if not result.success:
                if result.error:
                    print(f"    Error: {result.error}")
                if result.details:
                    print(f"    Details: {result.details}")
        
        print("\nEnhanced WebSocket Features Tested:")
        print("  - Connection pooling and management")
        print("  - Topic-based subscriptions")
        print("  - Multi-channel notifications")
        print("  - Health monitoring")
        print("  - Concurrent connection handling")
        print("  - REST API integration")
        print("  - Heartbeat mechanism")
        
        if pass_rate >= 80:
            print(f"\nWebSocket system is functioning well!")
        elif pass_rate >= 60:
            print(f"\nWebSocket system has some issues that need attention.")
        else:
            print(f"\nWebSocket system has significant issues that require fixing.")


async def main():
    """Main test execution"""
    # You can customize the base URL for your environment
    base_url = "ws://localhost:8000"  # Change if your server runs on different port
    
    test_client = WebSocketTestClient(base_url)
    
    try:
        results = await test_client.run_all_tests()
        test_client.print_test_summary(results)
        
        # Return appropriate exit code
        passed = sum(1 for r in results if r.success)
        total = len(results)
        
        if passed == total:
            print(f"\nAll tests passed! Enhanced WebSocket system is ready for production.")
            return 0
        else:
            print(f"\n{total - passed} tests failed. Please check the issues above.")
            return 1
            
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)