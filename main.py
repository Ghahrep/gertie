from fastapi import FastAPI

# Create the FastAPI application instance
app = FastAPI(
    title="Portfolio Risk AI Platform",
    description="API for the multi-agent financial intelligence system.",
    version="0.1.0"
)

@app.get("/")
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Portfolio Risk AI Platform!"}