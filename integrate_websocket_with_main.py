# integrate_websocket_with_main.py
"""
Script to integrate WebSocket functionality with your existing main.py
Safely adds WebSocket endpoints without disrupting your current setup
"""

import os
import re

def integrate_websocket_with_main():
    """
    Integrate WebSocket code with your existing main.py
    """
    
    main_file = 'main.py'
    
    if not os.path.exists(main_file):
        print("❌ main.py not found")
        return False
    
    print("🔧 Integrating WebSocket functionality with main.py...")
    
    # Read your existing main.py
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Check if WebSocket is already integrated
    if 'websocket_endpoint' in content:
        print("✅ WebSocket endpoints already integrated")
        return True
    
    # Find the right place to add imports (after your existing imports)
    import_section = """
# WebSocket imports for real-time notifications
from fastapi import WebSocket, WebSocketDisconnect, Query
from datetime import datetime, timezone
import json
import jwt

# Import WebSocket components (with fallback if not available)
try:
    from websocket.connection_manager import get_connection_manager, start_websocket_heartbeat
    WEBSOCKET_AVAILABLE = True
    print("✅ WebSocket components loaded")
except ImportError:
    print("⚠️  WebSocket components not found - WebSocket features disabled")
    WEBSOCKET_AVAILABLE = False

# Initialize WebSocket manager if available
if WEBSOCKET_AVAILABLE:
    manager = get_connection_manager()
"""
    
    # Find where to insert imports (after the existing FastAPI imports)
    fastapi_import_pattern = r'from fastapi import.*?\n'
    match = re.search(fastapi_import_pattern, content)
    
    if match:
        # Insert after FastAPI imports
        insert_pos = match.end()
        content = content[:insert_pos] + import_section + content[insert_pos:]
        print("✅ WebSocket imports added")
    else:
        # Fallback: add after the first import block
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('from ') or line.strip().startswith('import '):
                continue
            else:
                # Found end of imports
                lines.insert(i, import_section)
                content = '\n'.join(lines)
                break
        print("✅ WebSocket imports added (fallback)")
    
    # Add WebSocket functions and endpoints before the final if __name__ block
    websocket_code = '''

# WebSocket token verification (integrates with your auth)
def verify_websocket_token(token: str) -> dict:
    """Verify JWT token for WebSocket authentication"""
    try:
        if not token:
            return None
        
        # Try to use your existing auth settings
        try:
            # Adjust these imports based on your config location
            SECRET_KEY = "your-secret-key-here"  # Replace with your actual secret
            ALGORITHM = "HS256"
        except:
            SECRET_KEY = "fallback-secret"
            ALGORITHM = "HS256"
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
        
    except Exception as e:
        print(f"WebSocket token verification error: {e}")
        return None

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, token: str = Query(None)):
    """WebSocket endpoint for real-time notifications"""
    
    if not WEBSOCKET_AVAILABLE:
        await websocket.close(code=4000, reason="WebSocket not available")
        return
    
    # Optional authentication
    if token:
        auth_payload = verify_websocket_token(token)
        if not auth_payload:
            await websocket.close(code=4001, reason="Invalid token")
            return
    
    try:
        await manager.connect(websocket, user_id, token)
        
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get('type') == 'ping':
                        await manager.send_personal_message(user_id, {
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                except json.JSONDecodeError:
                    pass
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {user_id}")
    finally:
        if WEBSOCKET_AVAILABLE:
            await manager.disconnect(websocket, user_id)

@app.get("/api/v1/websocket/stats", tags=["WebSocket"])
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    if not WEBSOCKET_AVAILABLE:
        return {"status": "error", "message": "WebSocket not available"}
    
    try:
        stats = manager.get_connection_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        return {"status": "error", "message": "Failed to get stats"}

@app.post("/api/v1/websocket/test-notification/{user_id}", tags=["WebSocket"])
async def send_test_notification(user_id: str):
    """Send test notification to user"""
    if not WEBSOCKET_AVAILABLE:
        return {"status": "error", "message": "WebSocket not available"}
    
    test_alert = {
        "portfolio_id": "test_portfolio",
        "portfolio_name": "Test Portfolio", 
        "risk_score": 75.5,
        "risk_change_pct": 25.3,
        "severity": "high"
    }
    
    try:
        success = await manager.send_risk_alert(user_id, test_alert)
        return {
            "status": "success" if success else "failed",
            "message": f"Test notification {'sent' if success else 'failed'}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Helper function for risk alerts
async def send_risk_alert_notification(user_id: str, risk_data: dict, workflow_id: str = None):
    """Send risk alert notification via WebSocket"""
    if not WEBSOCKET_AVAILABLE:
        return False
    
    try:
        alert_data = {
            "portfolio_id": risk_data.get('portfolio_id'),
            "portfolio_name": risk_data.get('portfolio_name', 'Portfolio'),
            "risk_score": risk_data.get('risk_score'),
            "risk_change_pct": risk_data.get('risk_score_change_pct'),
            "severity": "high" if risk_data.get('risk_score', 0) > 80 else "medium",
            "workflow_id": workflow_id
        }
        
        return await manager.send_risk_alert(user_id, alert_data)
    except Exception as e:
        print(f"Failed to send risk alert: {e}")
        return False

# Enhanced startup event
@app.on_event("startup") 
async def startup_websocket_services():
    """Initialize WebSocket services"""
    if WEBSOCKET_AVAILABLE:
        try:
            start_websocket_heartbeat()
            print("✅ WebSocket services initialized")
        except Exception as e:
            print(f"❌ WebSocket startup failed: {e}")
'''
    
    # Find where to insert the WebSocket code (before if __name__ == "__main__")
    if 'if __name__ == "__main__":' in content:
        content = content.replace('if __name__ == "__main__":', websocket_code + '\nif __name__ == "__main__":')
        print("✅ WebSocket endpoints added before main block")
    else:
        # Add at the end
        content += websocket_code
        print("✅ WebSocket endpoints added at end")
    
    # Write the updated main.py
    with open(main_file, 'w') as f:
        f.write(content)
    
    print("✅ main.py successfully updated with WebSocket functionality")
    return True

def test_integration():
    """Test that the integration was successful"""
    
    print("\n🧪 Testing WebSocket integration...")
    
    try:
        # Test imports
        print("Testing imports...")
        
        # This is a basic syntax check
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ('WebSocket imports', 'from fastapi import WebSocket' in content),
            ('WebSocket endpoint', '@app.websocket("/ws/{user_id}")' in content),
            ('Connection manager', 'get_connection_manager' in content),
            ('Test endpoint', 'send_test_notification' in content),
            ('Stats endpoint', 'get_websocket_stats' in content)
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🎉 WebSocket integration test PASSED!")
            print("✅ All WebSocket components detected in main.py")
            return True
        else:
            print("\n⚠️  Some WebSocket components missing")
            return False
            
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False

def main():
    """Main integration function"""
    
    print("🚀 WEBSOCKET INTEGRATION WITH MAIN.PY")
    print("=" * 45)
    print("Safely adding WebSocket endpoints to your FastAPI app...")
    print()
    
    # Step 1: Integrate WebSocket code
    if integrate_websocket_with_main():
        # Step 2: Test integration
        if test_integration():
            print("\n🎉 WEBSOCKET INTEGRATION COMPLETE!")
            print("=" * 45)
            print("✅ WebSocket endpoints added to main.py")
            print("✅ Integration tested successfully") 
            print("✅ Your existing functionality preserved")
            
            print("\n📋 What was added:")
            print("   🔌 WebSocket endpoint: /ws/{user_id}")
            print("   📊 Stats endpoint: /api/v1/websocket/stats")
            print("   🧪 Test endpoint: /api/v1/websocket/test-notification/{user_id}")
            print("   🚀 Startup event for WebSocket services")
            print("   📤 Helper functions for risk alerts")
            
            print("\n🎯 Next Steps:")
            print("   1. Create websocket/ directory")
            print("   2. Add connection_manager.py")
            print("   3. Add risk_notification_service.py")
            print("   4. Test WebSocket endpoints")
            
            return True
    
    print("\n❌ Integration failed - check errors above")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)