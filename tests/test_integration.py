import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint returns expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Task Receiver Service is running. Post to /ready to submit a task."}

def test_status_endpoint_empty():
    """Test status endpoint when no task has been received."""
    response = client.get("/status")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Awaiting first task" in response.json()["message"]

def test_ready_endpoint_invalid_secret():
    """Test /ready endpoint rejects invalid secret."""
    test_payload = {
        "email": "test@example.com",
        "secret": "wrong-secret",
        "task": "test-task-123",
        "round": 1,
        "nonce": "test-nonce",
        "brief": "Create a test app",
        "checks": ["Test check"],
        "evaluation_url": "https://example.com/notify",
        "attachments": []
    }
    
    response = client.post("/ready", json=test_payload)
    assert response.status_code == 401
    assert "Unauthorized" in response.json()["detail"]

@patch('main.verify_secret')
@patch('main.asyncio.create_task')
def test_ready_endpoint_valid_request(mock_create_task, mock_verify_secret):
    """Test /ready endpoint accepts valid request with correct secret."""
    # Mock secret verification to return True
    mock_verify_secret.return_value = True
    
    # Mock the async task creation
    mock_create_task.return_value = None
    
    test_payload = {
        "email": "test@example.com", 
        "secret": "correct-secret",
        "task": "test-task-123",
        "round": 1,
        "nonce": "test-nonce",
        "brief": "Create a test app",
        "checks": ["Test check"],
        "evaluation_url": "https://example.com/notify",
        "attachments": []
    }
    
    response = client.post("/ready", json=test_payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert "processing started" in response.json()["message"]
    
    # Verify the background task was created
    mock_create_task.assert_called_once()

def test_ready_endpoint_invalid_email():
    """Test /ready endpoint rejects invalid email format."""
    test_payload = {
        "email": "invalid-email",  # Invalid email format
        "secret": "test-secret",
        "task": "test-task-123", 
        "round": 1,
        "nonce": "test-nonce",
        "brief": "Create a test app",
        "checks": ["Test check"],
        "evaluation_url": "https://example.com/notify",
        "attachments": []
    }
    
    response = client.post("/ready", json=test_payload)
    assert response.status_code == 422  # Validation error

def test_ready_endpoint_missing_fields():
    """Test /ready endpoint rejects request with missing required fields."""
    test_payload = {
        "email": "test@example.com",
        "secret": "test-secret"
        # Missing required fields
    }
    
    response = client.post("/ready", json=test_payload)
    assert response.status_code == 422  # Validation error