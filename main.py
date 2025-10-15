# Add at the top of main.py - BEFORE all other imports
import os
import shutil

# Cross-platform git executable detection
git_executable = shutil.which('git')
if git_executable:
    os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = git_executable
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

# Now import GitPython at module level
import git

from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from fastapi import Request
from models import TaskRequest # Ensure models.py is available
from config import get_settings
import asyncio
import httpx # Used for making the HTTP notification call
import json # For parsing the structured JSON response from the LLM
import base64
import re
import subprocess

import stat # For robust cleanup on Windows

# Assuming this model is defined elsewhere
# --- Configuration and Setup ---
settings = get_settings()

# --- Helper Function for Security ---
def verify_secret(secret_from_request: str) -> bool:
    """Checks if the provided secret matches the expected student secret."""
    return secret_from_request == settings.STUDENT_SECRET

# --- GITHUB CONSTANTS ---
GITHUB_API_BASE = "https://api.github.com"
# Pages URL is constructed dynamically using the username from settings
GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"
# --------------------------

# LLM Configuration
OPENAI_CHAT_COMPLETIONS_URL = f"{settings.OPENAI_API_BASE.rstrip('/')}/chat/completions"
OPENAI_API_KEY = settings.OPENAI_API_KEY
# Initialize the FastAPI application
app = FastAPI(
    title="Automated Task Receiver & Processor",
    description="Endpoint for receiving task assignments and triggering AI code generation/deployment."
)

# Global storage for the last received task (for demonstration purposes)
received_task_data = {}

# --- REFACTORING: SPLIT deploy_to_github ---

async def setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int):
    """Handles creating the remote repo (R1) or cloning the existing one (R2+) into an EMPTY directory."""
    
 

 
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            # 1. CREATE or INITIALIZE REPO / CLONE EXISTING REPO
            if round_index == 1:
                print(f"   -> R1: Creating remote repository '{repo_name}'...")
                payload = {"name": repo_name, "private": False, "auto_init": True}
                response = await client.post(f"{GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                response.raise_for_status()

                # Initialize local git repo in the EMPTY path
                repo = git.Repo.init(local_path)
                repo.create_remote('origin', repo_url_auth)
                print("   -> R1: Local git repository initialized.")
            
            elif round_index >= 2:
                # Crucial part for Round 2: Cloning the existing work into the EMPTY local_path
                print(f"   -> R{round_index}: Cloning existing repository from {repo_url_http}...")
                # local_path is guaranteed to be empty due to the cleanup and directory creation in the main function
                repo = git.Repo.clone_from(repo_url_auth, local_path)
                print(f"   -> R{round_index}: Repository cloned and ready for update.")
            
            return repo

        except httpx.HTTPStatusError as e:
            print(f"--- [API ERROR] GitHub API call failed with status {e.response.status_code}: {e.response.text} ---")
            raise Exception("GitHub API call failed during repository setup.")
        except git.GitCommandError as e:
            print(f"--- [GIT ERROR] Failed to perform git operation: {e} ---")
            raise Exception("Git operation failed during repository setup.")


async def commit_and_publish(repo, task_id: str, round_index: int, repo_name: str) -> dict:
    """Handles adding, committing, pushing, and configuring GitHub Pages after files are saved."""

    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            # 1. ADD, COMMIT, AND PUSH FILES
            # The new files (generated and attachments) are now in the local_path.
            repo.git.add(A=True)
            commit_message = f"Task {task_id} - Round {round_index}: LLM-generated app update/creation"
            repo.index.commit(commit_message)
            commit_sha = repo.head.object.hexsha
            print(f"   -> Files committed. SHA: {commit_sha}")

            # Ensure main branch consistency and push
            repo.git.branch('-M', 'main')
            print("   -> Branch renamed to 'main'.")
            repo.git.push('--set-upstream', 'origin', 'main', force=True)
            print("   -> Changes pushed to remote 'main' branch.")

            # Wait for GitHub to register the branch
            print("   -> Waiting 10 seconds for GitHub to register the main branch...")
            await asyncio.sleep(10)

            # 2. ENABLE GITHUB PAGES WITH ROBUST RETRIES
            print("   -> Enabling GitHub Pages with robust retries...")
            pages_api_url = f"{GITHUB_API_BASE}/repos/{github_username}/{repo_name}/pages"
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            pages_max_retries = 5
            pages_base_delay = 3

            for retry_attempt in range(pages_max_retries):
                try:
                    pages_response = await client.get(pages_api_url, headers=headers)
                    is_configured = (pages_response.status_code == 200)

                    if is_configured:
                        print(f"   -> Pages exists. Updating configuration (Attempt {retry_attempt + 1}).")
                        (await client.put(pages_api_url, json=pages_payload, headers=headers)).raise_for_status()
                    else:
                        print(f"   -> Creating Pages configuration (Attempt {retry_attempt + 1}).")
                        (await client.post(pages_api_url, json=pages_payload, headers=headers)).raise_for_status()

                    print("   -> Pages configuration successful.")
                    break
                
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 422 and "main branch must exist" in e.response.text and retry_attempt < pages_max_retries - 1:
                        delay = pages_base_delay * (2 ** retry_attempt)
                        print(f"   -> [Timing Issue] Branch not recognized. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        raise
            else:
                raise Exception("Failed to configure GitHub Pages after multiple retries due to branch existence.")

            # 3. CONSTRUCT RETURN VALUES
            print("   -> Waiting 5 seconds for GitHub Pages deployment...")
            await asyncio.sleep(5)

            pages_url = f"{GITHUB_PAGES_BASE}/{repo_name}/"

            return {
                "repo_url": repo_url_http,
                "commit_sha": commit_sha,
                "pages_url": pages_url
            }

        except git.GitCommandError as e:
            print(f"--- [GIT ERROR] Failed to perform git operation: {e} ---")
            raise Exception("Git operation failed during deployment.")
        except httpx.HTTPStatusError as e:
            print(f"--- [API ERROR] GitHub API call failed with status {e.response.status_code}: {e.response.text} ---")
            raise Exception("GitHub API call failed during deployment.")
        except Exception as e:
            print(f"--- [CRITICAL ERROR] Deployment failed: {e} ---")
            raise

# --- REMOVED: Original deploy_to_github (replaced by setup_local_repo and commit_and_publish) ---
# The function name deploy_to_github is now DELETED.

# --- Helper Functions for File System Operations ---

async def save_generated_files_locally(task_id: str, files: dict) -> str:
    """
    Saves the generated files (index.html, README.md, LICENSE) into a local 
    directory named after the task_id within the 'generated_tasks' folder.
    """
    base_dir = "/tmp"
    task_dir = os.path.join(base_dir, task_id)
    
    # Ensure the task-specific directory exists
    # NOTE: This directory is created earlier in the main orchestration function
    os.makedirs(task_dir, exist_ok=True)
    
    print(f"--- [LOCAL_SAVE] Saving files to: {task_dir} ---")
    
    # Write each file from the generated dictionary to the local file system
    for filename, content in files.items():
        file_path = os.path.join(task_dir, filename)
        try:
            # Write the content to the file. Assuming content is a string (text files).
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"   -> Saved: {filename} (Size: {len(content)} bytes)")
        except Exception as e:
            print(f"   -> ERROR saving {filename}: {e}")
            # If saving fails, we treat this as a critical error for the task
            raise Exception(f"Failed to save file {filename} locally.")

    return task_dir


# --- Helper Functions for External Services ---

async def call_llm_for_code(prompt: str, task_id: str) -> dict:
    """
    Calls the configured OpenAI-compatible API to generate the web application code
    and structured metadata (README and LICENSE).
    The response is strictly validated against a JSON schema.
    """
    print(f"--- [LLM_CALL] Attempting to generate code for Task: {task_id} using OpenAI API ---")

    # Define system instruction for the model
    system_prompt = (
        "You are an expert full-stack web developer with deep knowledge of modern web technologies. "
        "Your task is to create production-ready web applications that meet specific requirements.\n\n"
        "RESPONSE FORMAT: Return ONLY a valid JSON object with exactly these three keys:\n"
        "{\n  \"index.html\": \"...\",\n  \"README.md\": \"...\",\n  \"LICENSE\": \"...\"\n}\n\n"
        "TECHNICAL REQUIREMENTS:\n"
        "- 'index.html': Complete, self-contained HTML file with embedded CSS/JS\n"
        "- Use Tailwind CSS via CDN for styling (responsive design mandatory)\n"
        "- Include proper meta tags, semantic HTML, accessibility features\n"
        "- Handle URL parameters when specified (e.g., ?url=, ?token=)\n"
        "- Implement error handling and loading states\n"
        "- Support modern browsers and mobile devices\n\n"
        "GITHUB PAGES DEPLOYMENT REQUIREMENTS:\n"
        "- Ensure all resources load via HTTPS (use CDN links)\n"
        "- Make index.html fully functional without server-side processing\n"
        "- Process JSON/CSV data client-side with proper parsing\n"
        "- Use relative paths for all assets and attachments\n"
        "- Implement proper DOM manipulation for dynamic content\n"
        "- Test offline functionality and static file serving\n"
        "- Ensure charts and visualizations render on page load\n\n"
        "CONTENT REQUIREMENTS:\n"
        "- 'README.md': Professional documentation with setup, usage, and code explanation\n"
        "- 'LICENSE': Full text of MIT License with current year and placeholder name\n\n"
        "QUALITY STANDARDS:\n"
        "- Code must be clean, well-commented, and follow best practices\n"
        "- UI should be intuitive, visually appealing, and professional\n"
        "- Performance optimized (minimal external dependencies)\n"
        "- Security conscious (input validation, XSS prevention)\n\n"
        "PREMIUM QUALITY REQUIREMENTS:\n"
        "- Pixel-perfect design with consistent spacing and typography\n"
        "- Professional color schemes and visual hierarchy\n"
        "- Smooth animations and micro-interactions for premium feel\n"
        "- Comprehensive inline code documentation for maintainability\n"
        "- Industry-standard code organization and structure\n"
        "- Accessibility features (ARIA labels, keyboard navigation)\n\n"
        "CRITICAL ERROR HANDLING REQUIREMENTS:\n"
        "- Wrap ALL data parsing in try-catch blocks\n"
        "- Validate data types before processing (Array.isArray(), typeof checks)\n"
        "- Implement graceful fallbacks for API failures (mock data, offline mode)\n"
        "- Add user-friendly error messages with retry options\n"
        "- Handle CORS, network, and permission errors appropriately\n"
        "- Never let JavaScript errors crash the entire application\n"
        "- Use optional chaining (?.) and nullish coalescing (??) operators\n"
        "- Provide loading states and error boundaries for async operations"
    )

    # Use exponential backoff for the API call
    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                if not OPENAI_API_KEY:
                    raise Exception("OPENAI_API_KEY is not set.")

                # Build OpenAI Chat Completions payload
                openai_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                openai_payload = {
                    "model": "openai/gpt-4o-mini",  # Reliable, fast, and cost-effective model
                    "response_format": {"type": "json_object"},
                    "messages": openai_messages,
                    "max_tokens": 4000,  # Ensure sufficient tokens for complete responses
                    "temperature": 0.3   # Balanced creativity and consistency
                }

                # Log API base and token for debugging
                token_preview = (OPENAI_API_KEY[:6] + "…") if OPENAI_API_KEY else "none"
                print(f"--- [LLM_CALL] OpenAI Base={settings.OPENAI_API_BASE}, Token={token_preview} ---")

                response = await client.post(
                    OPENAI_CHAT_COMPLETIONS_URL,
                    json=openai_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000"),
                        "X-Title": os.getenv("OPENROUTER_X_TITLE", "TDS Task Receiver")
                    }
                )

                response.raise_for_status()

                if not response.text or not response.text.strip():
                    raise ValueError("Received empty response body from LLM.")

                result = response.json()

                if not result.get("choices") or not result["choices"][0].get("message") or not result["choices"][0]["message"].get("content"):
                    raise ValueError("LLM response is missing expected 'content'.")

                json_text = result["choices"][0]["message"]["content"]
                generated_files = json.loads(json_text)

                print(f"--- [LLM_CALL] Successfully generated files on attempt {attempt + 1}. ---")
                return generated_files

        except httpx.HTTPStatusError as e:
            body = e.response.text if e.response else "<no body>"
            print(f"--- [LLM_CALL] HTTP Error on attempt {attempt + 1}: {e}. Body: {body} ---")
        except (httpx.RequestError, KeyError, json.JSONDecodeError, ValueError) as e:
            print(f"--- [LLM_CALL] Processing Error on attempt {attempt + 1}: {e}. ---")

        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            print(f"--- [LLM_CALL] Retrying LLM call in {delay} seconds... ---")
            await asyncio.sleep(delay)

    print("--- [LLM_CALL] Failed to generate code after multiple retries. ---")
    raise Exception("LLM Code Generation Failure")


async def notify_evaluation_server(
    evaluation_url: str, 
    email: str,          
    task_id: str, 
    round_index: int,    
    nonce: str, 
    repo_url: str,
    commit_sha: str,     
    pages_url: str       
) -> bool:
    """
    Calls the evaluation_url to notify the server that the code has been deployed.
    """
    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url  
    }
    
    max_retries = 3
    base_delay = 1
    
    print(f"--- [NOTIFICATION] Attempting to notify server at {evaluation_url} ---")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(evaluation_url, json=payload)
                response.raise_for_status() # Raises an exception for 4xx/5xx status codes
                
                print(f"--- [NOTIFICATION] Successfully notified server. Response: {response.status_code} ---")
                return True
        except httpx.HTTPStatusError as e:
            print(f"--- [NOTIFICATION] HTTP Error on attempt {attempt + 1}: {e}. ---")
        except httpx.RequestError as e:
            print(f"--- [NOTIFICATION] Request Error on attempt {attempt + 1}: {e}. ---")
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            print(f"--- [NOTIFICATION] Retrying in {delay} seconds... ---")
            await asyncio.sleep(delay)
            
    print(f"--- [NOTIFICATION] Failed to notify evaluation server after {max_retries} attempts. ---")
    return False


async def save_attachments_locally(task_dir: str, attachments: list) -> list:
    """
    Decodes and saves attachments (provided as Base64 Data URIs) into the task directory.
    Returns a list of saved filenames.
    """
    saved_files = []
    print(f"--- [ATTACHMENTS] Processing {len(attachments)} attachments for: {task_dir} ---")
    
    for attachment in attachments:
        filename = attachment.name 
        data_uri = attachment.url
        
        if not filename or not data_uri or not data_uri.startswith("data:"):
            print(f"    -> WARNING: Skipping invalid attachment entry: {filename}")
            continue

        # Use regex to extract the Base64 part of the URI (after base64,)
        match = re.search(r"base64,(.*)", data_uri, re.IGNORECASE)
        if not match:
            print(f"    -> ERROR: Could not find base64 data in URI for {filename}")
            continue

        base64_data = match.group(1)
        file_path = os.path.join(task_dir, filename)

        try:
            # Decode the base64 string
            file_bytes = base64.b64decode(base64_data)
            
            # Write the raw bytes to the file
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            
            print(f"    -> Saved Attachment: {filename} (Size: {len(file_bytes)} bytes)")
            saved_files.append(filename)
            
        except Exception as e:
            print(f"    -> CRITICAL ERROR saving attachment {filename}: {e}")
            raise Exception(f"Failed to save attachment {filename} locally.")

    return saved_files
# --- Main Orchestration Logic ---


async def generate_files_and_deploy(task_data: TaskRequest):
    """
    The asynchronous background process that executes the main project workflow.
    It adapts the LLM prompt for multi-round tasks and fixes the cloning order.
    CRITICAL: Must complete within 10 minutes as per project requirements.
    """
    import time
    start_time = time.time()
    MAX_PROCESSING_TIME = 9 * 60  # 9 minutes to allow 1 minute buffer
    
    task_id = task_data.task
    email = task_data.email         
    round_index = task_data.round 
    brief = task_data.brief
    evaluation_url = task_data.evaluation_url
    nonce = task_data.nonce
    attachments = task_data.attachments
    
    def check_time_limit():
        elapsed = time.time() - start_time
        if elapsed > MAX_PROCESSING_TIME:
            raise Exception(f"⏰ TIMEOUT: Processing exceeded {MAX_PROCESSING_TIME/60:.1f} minutes limit")
        return elapsed
    
    print(f"\n--- [PROCESS START] Starting background task for {task_id}, Round {round_index} ---")
    print(f"⏰ Time limit: {MAX_PROCESSING_TIME/60:.1f} minutes for full deployment cycle")
    
    # Deployment configuration
    repo_name = task_id.replace(' ', '-').lower()
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    repo_url_auth = f"https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git"
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"
    
    try:
        # 0. Setup local directory
        base_dir = "/tmp"
        local_path = os.path.join(base_dir, task_id)

        # --- ROBUST CLEANUP LOGIC ---
        # Crucial: Cleans up local directory before cloning or creating a new repo.
        if os.path.exists(local_path):
            print(f"--- [CLEANUP] Deleting existing local directory: {local_path} ---")
            
            def onerror(func, path, exc_info):
                """Error handler for shutil.rmtree to handle permission issues."""
                if exc_info[0] is PermissionError or 'WinError 5' in str(exc_info[1]):
                    os.chmod(path, stat.S_IWUSR)
                    func(path)
                else:
                    raise

            try:
                shutil.rmtree(local_path, onerror=onerror)
                print("--- [CLEANUP] Directory deleted successfully. ---")
            except Exception as e:
                print(f"!!! CRITICAL: Failed to clean up directory. Error: {e}")
                raise Exception(f"Failed to perform local cleanup: {e}")
        
        # Create the fresh, EMPTY directory (ready for clone or init)
        os.makedirs(local_path, exist_ok=True)
        # --- END ROBUST CLEANUP ---
        
        # 1. SETUP REPO (Clone or Init)
        # MUST run before any files are saved to local_path.
        print(f"--- [DEPLOYMENT] Setting up local Git repository for Round {round_index}... ---")
        repo = await setup_local_repo(
            local_path=local_path, 
            repo_name=repo_name, 
            repo_url_auth=repo_url_auth, 
            repo_url_http=repo_url_http, 
            round_index=round_index
        )

        # 2. Process Attachments for Prompt Context
        attachment_list_for_llm_prompt = [att.name for att in attachments]
        attachment_list_str = ", ".join(attachment_list_for_llm_prompt)
        
        # 3. AI Code Generation - Adapt Prompt for Round 2
        
        # --- ENHANCED LLM PROMPT GENERATION ---
        if round_index > 1:
            # ROUND 2+ - MODIFICATION/UPDATE TASK
            llm_prompt = (
                f"🔄 MODIFICATION TASK - ROUND {round_index}\n\n"
                f"TASK OBJECTIVE: {brief}\n\n"
                "CONTEXT: This is an update to an existing web application. You must enhance/modify the current functionality.\n\n"
                "REQUIREMENTS:\n"
                "- Implement the new features/changes specified in the task objective\n"
                "- Maintain backward compatibility where possible\n"
                "- Update documentation to reflect changes\n"
                "- Preserve existing working functionality unless explicitly asked to change it\n"
                "- Add comprehensive error handling for new features\n"
                "- Ensure mobile responsiveness is maintained\n\n"
                "DELIVERABLES:\n"
                "- Updated 'index.html' with new functionality (complete, self-contained file)\n"
                "- Revised 'README.md' documenting the changes and new usage instructions\n"
                "- Updated 'LICENSE' file (MIT License with current year)\n\n"
                "TECHNICAL NOTES:\n"
                "- Use modern JavaScript (ES6+) and best practices\n"
                "- Implement proper input validation and error states\n"
                "- Optimize for performance and user experience\n"
                "- Include loading indicators for async operations\n\n"
                "RELIABILITY & COMPATIBILITY:\n"
                "- Ensure new features don't break existing functionality\n"
                "- Add thorough error handling for all new integrations\n"
                "- Test interactions between old and new features\n"
                "- Maintain data consistency across feature updates\n"
                "- Preserve user state and preferences during modifications\n\n"
                "QUALITY ENHANCEMENT PROCESS:\n"
                "1. ASSESS: Review existing code structure and identify integration points\n"
                "2. ENHANCE: Add new features with premium design consistency\n"
                "3. REFINE: Improve overall user experience and visual cohesion\n"
                "4. VERIFY: Ensure seamless interaction between old and new features\n"
                "5. ELEVATE: Add final polish for professional-grade result"
            )
        else:
            # ROUND 1 - INITIAL CREATION TASK
            llm_prompt = (
                f"🚀 NEW APPLICATION TASK - ROUND {round_index}\n\n"
                f"TASK OBJECTIVE: {brief}\n\n"
                "REQUIREMENTS:\n"
                "- Create a complete, functional web application from scratch\n"
                "- Implement all features specified in the task objective\n"
                "- Handle edge cases and provide user feedback\n"
                "- Support URL parameters if mentioned in the brief\n"
                "- Include proper error handling and validation\n"
                "- Ensure cross-browser compatibility and mobile responsiveness\n\n"
                "DELIVERABLES:\n"
                "- Complete 'index.html' file (fully self-contained with embedded styles/scripts)\n"
                "- Professional 'README.md' with setup instructions and usage guide\n"
                "- Standard 'LICENSE' file (MIT License)\n\n"
                "TECHNICAL SPECIFICATIONS:\n"
                "- Use Tailwind CSS via CDN for consistent, responsive styling\n"
                "- Implement vanilla JavaScript or lightweight libraries only\n"
                "- Include proper HTML5 semantic elements and ARIA attributes\n"
                "- Add meta tags for SEO and social sharing\n"
                "- Implement loading states and user feedback mechanisms\n\n"
                "RELIABILITY FOCUS:\n"
                "- Test edge cases: empty data, network failures, invalid inputs\n"
                "- Provide meaningful fallbacks for all external dependencies\n"
                "- Ensure the app works even without internet connectivity\n"
                "- Add comprehensive error messages for debugging\n"
                "- Use defensive programming patterns throughout\n\n"
                "PREMIUM DEVELOPMENT PROCESS:\n"
                "1. PLAN: Architect component structure and data flow\n"
                "2. BUILD: Implement core functionality with clean code\n"
                "3. POLISH: Add animations, transitions, and visual refinements\n"
                "4. VALIDATE: Test all interactions and error scenarios\n"
                "5. OPTIMIZE: Ensure performance and accessibility standards"
            )
        
        # Add attachment context with enhanced instructions
        if attachment_list_str:
            llm_prompt += (
                f"\n\n📎 ATTACHMENT FILES AVAILABLE:\n"
                f"Files: {attachment_list_str}\n\n"
                "ATTACHMENT USAGE INSTRUCTIONS:\n"
                "- These files will be available in the same directory as index.html\n"
                "- Reference them using relative paths (e.g., './filename.ext')\n"
                "- For CSV files: Parse and display data in tables or visualizations\n"
                "- For JSON files: Use for configuration or data processing\n"
                "- For images: Display with proper alt text and responsive sizing\n"
                "- Implement error handling if files cannot be loaded\n"
                "- Show loading states while processing attachment data\n\n"
                "API INTEGRATION BEST PRACTICES:\n"
                "- Always provide fallback/mock data when external APIs fail\n"
                "- Use fetch() with proper error handling and timeouts\n"
                "- Implement retry logic with exponential backoff\n"
                "- Handle network errors, CORS issues, and rate limiting\n"
                "- Validate API responses before processing (check structure/types)\n"
                "- Provide clear error messages explaining what went wrong\n"
                "- Include offline functionality with cached/default data\n\n"
                "DATA VALIDATION EXAMPLES (CRITICAL FOR RELIABILITY):\n"
                "```javascript\n"
                "// Always validate arrays before forEach\n"
                "if (Array.isArray(events) && events.length > 0) {\n"
                "    events.forEach(event => { /* safe processing */ });\n"
                "} else {\n"
                "    console.warn('Events data invalid, using fallback');\n"
                "    displayFallbackMessage();\n"
                "}\n\n"
                "// Robust JSON parsing\n"
                "try {\n"
                "    const data = JSON.parse(response);\n"
                "    if (data && typeof data === 'object') {\n"
                "        processData(data);\n"
                "    }\n"
                "} catch (error) {\n"
                "    showError('Data parsing failed: ' + error.message);\n"
                "}\n"
                "```\n\n"
                "INTERACTIVE FUNCTIONALITY REQUIREMENTS:\n"
                "- Transform raw JSON/CSV data into interactive elements\n"
                "- Create dynamic tables, charts, and visualizations\n"
                "- Implement click handlers, form interactions, and user controls\n"
                "- Use Chart.js or similar for data visualization\n"
                "- Add search, filter, and sort functionality where applicable\n"
                "- Ensure all buttons and interactive elements work properly\n"
                "- Parse attachment files and display content meaningfully\n"
                "- Never display raw JSON - always convert to user-friendly format\n\n"
                "FINAL QUALITY REFLECTION:\n"
                "Take a moment to review your implementation:\n"
                "• Is this something you'd be proud to showcase professionally?\n"
                "• Does it exceed expectations rather than just meet requirements?\n"
                "• Would users describe this as 'polished' and 'premium'?\n"
                "• Are there small enhancements that would significantly improve UX?\n\n"
                "GITHUB PAGES WORKFLOW STANDARDS:\n"
                "1. INITIALIZATION: Set up proper DOM ready event listeners\n"
                "2. DATA LOADING: Fetch and parse all attachment files on page load\n"
                "3. UI RENDERING: Create interactive elements from parsed data\n"
                "4. ERROR HANDLING: Provide fallbacks when files/data are missing\n"
                "5. RESPONSIVENESS: Ensure functionality works on all screen sizes\n"
                "6. TESTING: Verify functionality works in static file environment\n\n"
                "EXAMPLE INITIALIZATION PATTERN:\n"
                "```javascript\n"
                "document.addEventListener('DOMContentLoaded', async () => {\n"
                "    try {\n"
                "        const data = await loadAttachmentData();\n"
                "        renderDashboard(data);\n"
                "        initializeInteractivity();\n"
                "    } catch (error) {\n"
                "        displayFallbackContent();\n"
                "    }\n"
                "});\n"
                "```\n\n"
                "QUALITY VALIDATION CHECKPOINT:\n"
                "Before finalizing, verify:\n"
                "✓ All interactive elements respond correctly\n"
                "✓ Data transforms from raw format to user-friendly display\n"
                "✓ Error states provide clear, actionable feedback\n"
                "✓ Loading states prevent user confusion\n"
                "✓ Mobile responsiveness works across screen sizes\n"
                "✓ Color contrast meets accessibility standards\n"
                "✓ Code is well-structured and commented for maintenance"
            )
        # --- MODIFICATION END ---
        
        # Time check before LLM call (most time-consuming operation)
        elapsed = check_time_limit()
        print(f"⏰ Time check before LLM call: {elapsed:.1f}s elapsed")
        
        # Call LLM
        generated_files = await call_llm_for_code(llm_prompt, task_id)
        
        # Time check after LLM call
        elapsed = check_time_limit()
        print(f"⏰ Time check after LLM call: {elapsed:.1f}s elapsed")
        
        # 4. Save Generated Code Locally
        # This overwrites the cloned files (index.html, README.md, LICENSE)
        await save_generated_files_locally(task_id, generated_files)
        
        # 5. Save Attachments Locally
        # This adds attachments (like data.csv) to the local directory
        # The attachment saving now happens *after* the clone/init, resolving the Round 2 error.
        await save_attachments_locally(local_path, attachments)

        # 6. COMMIT AND PUBLISH
        print(f"--- [DEPLOYMENT] Committing and Publishing task {task_id}, Round {round_index} to GitHub... ---")
        
        deployment_info = await commit_and_publish(
            repo=repo, 
            task_id=task_id,
            round_index=round_index,
            repo_name=repo_name
        )
        
        repo_url = deployment_info["repo_url"]
        commit_sha = deployment_info["commit_sha"]
        pages_url = deployment_info["pages_url"] 
        
        print(f"--- [DEPLOYMENT] Success! Repo: {repo_url}, Pages: {pages_url} ---")
        
        # Final time check before notification
        elapsed = check_time_limit()
        print(f"⏰ Final time check before notification: {elapsed:.1f}s elapsed")
        
        # 7. Notify the Evaluation Server
        await notify_evaluation_server(
            evaluation_url=evaluation_url, 
            email=email,
            task_id=task_id, 
            round_index=round_index,
            nonce=nonce, 
            repo_url=repo_url,
            commit_sha=commit_sha,
            pages_url=pages_url
        )

    except Exception as e:
        print(f"--- [CRITICAL FAILURE] Task {task_id} failed during processing: {e} ---")
        
    print(f"--- [PROCESS END] Background task for {task_id} completed. ---")


# --- FastAPI Endpoint ---

@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest):
    """
    API endpoint that receives the task payload. 
    It verifies the secret and starts the generation/deployment process in the background.
    """
    global received_task_data
    
    # 1. SECRET VERIFICATION (CRITICAL PROJECT REQUIREMENT)
    if not verify_secret(task_data.secret):
        print(f"--- FAILED SECRET VERIFICATION for task {task_data.task} ---")
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Secret does not match configured student secret."
        )

    # Store data and print initial confirmation
    received_task_data = task_data.model_dump()
    
    print("--- TASK RECEIVED SUCCESSFULLY ---")
    print(f"Task ID: {received_task_data['task']}, Round: {received_task_data['round']}")
    
    # Start the processing function in the background 
    asyncio.create_task(generate_files_and_deploy(task_data))

    # Respond immediately with 200 OK to the evaluation server
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "message": f"Task {task_data.task} received and processing started."}
    )

@app.get("/")
async def root():
    return {"message": "Task Receiver Service is running. Post to /ready to submit a task."}

@app.get("/status")
async def get_status():
    global received_task_data
    if received_task_data:
        # Note: This status only shows the last received request, not the live status of the background task.
        return {"last_received_task": received_task_data}
    else:
        return {"message": "Awaiting first task submission to /ready"}

    