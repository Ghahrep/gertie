# run_mcp_standalone.py
"""
Standalone MCP server that doesn't conflict with existing main.py
Run this to test the MCP foundation without affecting your current system.
"""

import asyncio
import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_mcp_server():
    """Run the MCP server standalone"""
    print("🚀 Starting Standalone MCP Server")
    print("=" * 50)
    print("📍 This runs independently of your main application")
    print("🌐 MCP Server will start on http://localhost:8001")
    print("=" * 50)
    
    try:
        # Import MCP server directly
        from mcp.server import app
        print("✅ MCP modules loaded successfully")
        
        # Log available endpoints
        print("📋 Available MCP endpoints:")
        print("  • GET  /health - Health check")
        print("  • GET  /agents - List registered agents") 
        print("  • POST /register - Register new agent")
        print("  • POST /submit_job - Submit analysis job")
        print("  • GET  /job/{job_id} - Get job status")
        print("  • DELETE /agents/{agent_id} - Unregister agent")
        print("=" * 50)
        
        # Start server
        uvicorn.run(
            app,
            host="127.0.0.1",  # localhost only for testing
            port=8001,
            reload=False,  # No reload to avoid conflicts
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Failed to import MCP modules: {e}")
        print("\n🔧 Make sure you have created:")
        print("  • mcp/__init__.py")
        print("  • mcp/server.py")
        print("  • mcp/workflow_engine.py")
        print("  • mcp/schemas.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"💥 Failed to start MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_mcp_server()