# start_mcp_server.py
"""
Startup script for the MCP (Master Control Plane) server.
Run this before testing the MCP foundation.
"""

import asyncio
import uvicorn
import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the MCP server"""
    print("🚀 Starting MCP (Master Control Plane) Server")
    print("=" * 50)
    
    try:
        # Check if MCP modules exist
        from mcp.server import app
        print("✅ MCP modules loaded successfully")
        
        # Start the server
        print("🌐 MCP Server starting on http://localhost:8001")
        print("📋 Available endpoints:")
        print("  • GET  /health - Health check")
        print("  • GET  /agents - List registered agents")
        print("  • POST /register - Register new agent")
        print("  • POST /submit_job - Submit analysis job")
        print("  • GET  /job/{job_id} - Get job status")
        print("=" * 50)
        
        # Run the server with a different module path to avoid conflicts
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            reload=False,  # Disable reload to avoid conflicts
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import MCP modules: {e}")
        print("\n🔧 Setup Instructions:")
        print("1. Make sure you've created the mcp/ directory in your project root")
        print("2. Add the following files to mcp/:")
        print("   • __init__.py (empty file)")
        print("   • server.py")
        print("   • workflow_engine.py") 
        print("   • schemas.py")
        print("3. Install required dependencies:")
        print("   pip install fastapi uvicorn aiohttp")
        sys.exit(1)
        
    except Exception as e:
        print(f"💥 Failed to start MCP server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()