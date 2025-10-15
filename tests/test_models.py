import pytest
from models import TaskRequest

def test_task_request_model_instantiation():
    """
    Tests that the TaskRequest model can be instantiated with the example data.
    This test will fail if the incorrect model definition is used.
    """
    example_data = {
        "email": "student@example.com",
        "secret": "my-secure-token",
        "task": "captcha-solver-12345",
        "round": 1,
        "nonce": "ab12-cd34-ef56",
        "brief": "Create a captcha solver that handles ?url=https://.../image.png.",
        "checks": [
            "Repo has MIT license",
            "README.md is professional"
        ],
        "evaluation_url": "https://example.com/notify",
        "attachments": [
            {
                "name": "sample.png",
                "url": "data:image/png;base64,iVBORw..."
            }
        ]
    }

    try:
        TaskRequest(**example_data)
    except Exception as e:
        pytest.fail(f"Failed to instantiate TaskRequest model with example data: {e}")
