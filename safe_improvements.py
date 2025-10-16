# SAFE MINIMAL IMPROVEMENTS - Add these to existing main.py WITHOUT changing core logic

import re
from typing import Optional

# 1. SAFE FILENAME SANITIZATION (add to existing attachment handling)
def sanitize_filename_safe(filename: str) -> str:
    """Safely sanitize filenames without breaking existing functionality"""
    if not filename or len(filename) > 100:
        return f"attachment_{hash(filename) % 1000}.tmp"
    
    # Only remove truly dangerous characters, keep everything else
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    
    # Fallback if somehow empty
    return safe_name if safe_name.strip() else f"file_{hash(filename) % 1000}.tmp"

# 2. SAFE INPUT VALIDATION (add before processing)
def validate_input_safe(task_data) -> Optional[str]:
    """Validate inputs without breaking existing functionality"""
    
    # Check task ID length (prevent extremely long names)
    if len(task_data.task) > 100:
        return "Task ID too long (max 100 characters)"
    
    # Check attachment count (prevent abuse)
    if len(task_data.attachments) > 50:
        return "Too many attachments (max 50)"
    
    # Check email domain (basic validation)
    if '@' not in task_data.email:
        return "Invalid email format"
    
    return None  # All good

# 3. SAFE MEMORY CHECK (add before processing)
def check_system_resources_safe() -> bool:
    """Basic resource check without breaking functionality"""
    try:
        import shutil
        free_space_gb = shutil.disk_usage("/tmp").free / (1024**3)
        return free_space_gb > 0.5  # Require at least 500MB free
    except:
        return True  # If check fails, proceed anyway

# 4. SAFE REQUEST SIZE LIMIT (add to attachment processing)
MAX_SAFE_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10MB per attachment

def validate_attachment_size_safe(data_uri: str) -> bool:
    """Check attachment size without breaking existing flow"""
    try:
        # Rough estimate: base64 is ~1.37x original size
        estimated_size = len(data_uri) * 0.75
        return estimated_size < MAX_SAFE_ATTACHMENT_SIZE
    except:
        return True  # If check fails, proceed anyway

# 5. SAFE ERROR LOGGING ENHANCEMENT (replace existing print statements)
def log_safe(level: str, message: str):
    """Enhanced logging that falls back to print if needed"""
    try:
        import logging
        logger = logging.getLogger(__name__)
        if level.upper() == "ERROR":
            logger.error(message)
        elif level.upper() == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    except:
        # Fallback to existing print behavior
        print(f"[{level.upper()}] {message}")

# INTEGRATION EXAMPLE - How to safely add to existing code:
"""
# In save_attachments_locally function, ADD these lines:

# Before processing each attachment:
if not validate_attachment_size_safe(data_uri):
    log_safe("WARNING", f"Attachment {filename} too large, skipping")
    continue

# Before creating file:
safe_filename = sanitize_filename_safe(filename)

# At start of generate_files_and_deploy:
error = validate_input_safe(task_data)
if error:
    log_safe("ERROR", f"Input validation failed: {error}")
    return  # Exit gracefully

if not check_system_resources_safe():
    log_safe("WARNING", "Low system resources detected")
    # Continue anyway - don't block processing
"""