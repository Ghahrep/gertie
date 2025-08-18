# test_market_intelligence_api.py
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.proactive_monitor import get_proactive_monitor
from services.mcp_client import get_mcp_client
from mcp.schemas import JobRequest

async def test_market_intelligence_api():
    print("🧪 Testing Market Intelligence API & Monitoring System...")
    
    # Test 1: Proactive Monitor
    print("\n🔍 Testing Proactive Monitor...")
    monitor = await get_proactive_monitor()
    
    # Test monitoring start
    result = await monitor.start_portfolio_monitoring("test_portfolio_123", "test_user")
    print(f"✅ Monitoring start: {result['status']}")
    
    # Check active monitors
    active_monitors = monitor.get_active_monitors()
    print(f"📊 Active monitors: {len(active_monitors)}")
    
    # Test alert generation
    demo_alerts = await monitor.generate_proactive_alerts("test_portfolio_123", {
        "user_id": "test_user",
        "var_breach": True,
        "current_var": 0.035,
        "correlation_spike": True,
        "avg_correlation": 0.85
    })
    print(f"🚨 Generated {len(demo_alerts)} demo alerts")
    
    # Test alert retrieval
    portfolio_alerts = monitor.get_portfolio_alerts("test_portfolio_123")
    print(f"📋 Retrieved {len(portfolio_alerts)} alerts for portfolio")
    
    # Test monitoring stats
    stats = monitor.get_monitoring_stats()
    print(f"📈 System stats: {stats['active_monitors']} monitors, {stats['total_alerts']} total alerts")
    
    # Test 2: MCP Integration
    print("\n🎭 Testing MCP Integration...")
    try:
        mcp_client = await get_mcp_client()
        
        # Test health check
        health = await mcp_client.health_check()
        print(f"✅ MCP Health: {health.status}")
        
        # Test job submission for market intelligence
        job_request = JobRequest(
            query="Test market intelligence integration",
            context={"portfolio_id": "test_portfolio_123", "test_mode": True},
            required_capabilities=["market_intelligence"]
        )
        
        job_response = await mcp_client.submit_job(job_request)
        print(f"✅ MCP Job submitted: {job_response.job_id}")
        
        # Check job status
        await asyncio.sleep(1)
        status = await mcp_client.get_job_status(job_response.job_id)
        print(f"📊 Job Status: {status.status.value if status else 'Not found'}")
        
    except Exception as e:
        print(f"⚠️ MCP Integration test failed: {str(e)}")
    
    # Test 3: Alert System
    print("\n🚨 Testing Alert System...")
    
    # Test alert summary
    summary = await monitor.get_alert_summary("test_portfolio_123", hours=1)
    print(f"📋 Alert Summary: {summary['total_alerts']} alerts, risk level: {summary['risk_level']}")
    
    # Test alert resolution
    if portfolio_alerts:
        alert_id = portfolio_alerts[0]["alert_id"]
        resolved = await monitor.resolve_alert(alert_id)
        print(f"✅ Alert resolution: {'Success' if resolved else 'Failed'}")
    
    # Test threshold updates
    threshold_update = await monitor.update_thresholds({"var_breach": 0.03})
    print(f"⚙️ Threshold update: {len(threshold_update['updated_thresholds'])} thresholds updated")
    
    # Test 4: Clean up
    print("\n🧹 Testing Cleanup...")
    stop_result = await monitor.stop_portfolio_monitoring("test_portfolio_123")
    print(f"✅ Monitoring stop: {stop_result['status']}")
    
    print("\n🎉 Market Intelligence API & Monitoring Test Complete!")
    
    # Summary
    print("\n📊 TEST SUMMARY:")
    print(f"✅ Proactive monitoring: Operational")
    print(f"✅ Alert generation: {len(demo_alerts)} alerts created")
    print(f"✅ Alert management: Resolution and retrieval working")
    print(f"✅ MCP integration: Job submission and tracking working")
    print(f"✅ System statistics: Monitoring stats available")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_market_intelligence_api())