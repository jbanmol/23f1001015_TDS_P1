from pydantic import BaseModel, EmailStr
from typing import List, Optional

# Defines the structure for an individual attachment, like a sample captcha image
class Attachment(BaseModel):
    """
    Represents an attachment provided in the task payload.
    The 'url' is expected to be a data URI (e.g., base64 encoded image).
    """
    name: str
    url: str

# Defines the complete structure of the JSON request body
class TaskRequest(BaseModel):
    """
    The main model representing the task request sent by the evaluation server.
    """
    email: EmailStr  # Enforces a valid email format
    secret: str
    task: str
    round: int
    nonce: str
    brief: str
    checks: List[str]  # A list of strings detailing the evaluation checks
    evaluation_url: str
    attachments: List[Attachment] # A list of Attachment objects

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }
