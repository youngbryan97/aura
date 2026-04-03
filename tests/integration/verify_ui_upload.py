################################################################################

import pytest
import asyncio
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import os

# Setup path
sys.path.append(str(Path(__file__).parent.parent))

from interface.server import app, UPLOAD_DIR

client = TestClient(app)

def test_file_upload_endpoint():
    print("--- Testing /api/upload ---")
    
    # Create dummy file
    filename = "test_image.png"
    file_content = b"fake image content"
    
    response = client.post(
        "/api/upload",
        files={"file": (filename, file_content, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["filename"] == filename
    
    # Verify file saved
    saved_path = Path(UPLOAD_DIR) / filename
    assert saved_path.exists()
    assert saved_path.read_bytes() == file_content
    
    # Cleanup
    if saved_path.exists():
        saved_path.unlink()
        
    print("✓ Upload endpoint functional")

if __name__ == "__main__":
    test_file_upload_endpoint()


##
