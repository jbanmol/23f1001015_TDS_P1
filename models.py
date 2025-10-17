from pydantic import BaseModel, EmailStr, field_validator
from typing import List, Optional, Union

# Defines the structure for an individual attachment, like a sample captcha image
class Attachment(BaseModel):
    """
    Represents an attachment provided in the task payload.
    The 'url' is expected to be a data URI (e.g., base64 encoded image).
    """
    name: str
    url: str
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Attachment name cannot be empty')
        return v.strip()
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError('Attachment URL cannot be empty')
        if not v.startswith('data:'):
            raise ValueError('Attachment URL must be a data URI')
        return v.strip()

# Defines the complete structure of the JSON request body
class TaskRequest(BaseModel):
    """
    The main model representing the task request sent by the evaluation server.
    ENHANCED: Handles edge cases and validates all fields properly.
    """
    email: EmailStr  # Enforces a valid email format
    secret: str
    task: str
    round: int
    nonce: str
    brief: str
    checks: List[Union[str, dict]]  # Can be strings or complex JavaScript expressions
    evaluation_url: str
    attachments: Optional[List[Attachment]] = []  # CRITICAL: Make attachments optional with default empty list
    
    @field_validator('secret')
    @classmethod
    def validate_secret(cls, v):
        if not v or not v.strip():
            raise ValueError('Secret cannot be empty')
        return v.strip()
    
    @field_validator('task')
    @classmethod
    def validate_task(cls, v):
        if not v or not v.strip():
            raise ValueError('Task ID cannot be empty')
        # Sanitize task ID for safe repository naming
        return v.strip()
    
    @field_validator('round')
    @classmethod
    def validate_round(cls, v):
        if v < 1:
            raise ValueError('Round must be >= 1')
        if v > 10:  # Reasonable upper limit
            raise ValueError('Round must be <= 10')
        return v
    
    @field_validator('nonce')
    @classmethod
    def validate_nonce(cls, v):
        if not v or not v.strip():
            raise ValueError('Nonce cannot be empty')
        return v.strip()
    
    @field_validator('brief')
    @classmethod
    def validate_brief(cls, v):
        if not v or not v.strip():
            raise ValueError('Brief cannot be empty')
        if len(v.strip()) > 5000:  # Reasonable limit
            raise ValueError('Brief too long (max 5000 characters)')
        return v.strip()
    
    @field_validator('checks')
    @classmethod
    def validate_checks(cls, v):
        if not v:
            raise ValueError('Checks list cannot be empty')
        if len(v) > 50:  # Reasonable limit
            raise ValueError('Too many checks (max 50)')
        
        # Convert all checks to strings and validate
        validated_checks = []
        for check in v:
            if isinstance(check, dict):
                # Handle complex JavaScript expressions that might come as objects
                check_str = str(check.get('expression', check))
            else:
                check_str = str(check)
            
            if not check_str.strip():
                raise ValueError('Check cannot be empty')
            validated_checks.append(check_str.strip())
        
        return validated_checks
    
    @field_validator('evaluation_url')
    @classmethod
    def validate_evaluation_url(cls, v):
        if not v or not v.strip():
            raise ValueError('Evaluation URL cannot be empty')
        v = v.strip()
        if not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError('Evaluation URL must start with http:// or https://')
        return v
    
    @field_validator('attachments')
    @classmethod
    def validate_attachments(cls, v):
        if v is None:
            return []  # Convert None to empty list
        if len(v) > 20:  # Reasonable limit
            raise ValueError('Too many attachments (max 20)')
        return v

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
