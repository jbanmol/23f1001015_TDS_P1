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
    if not secret_from_request or not isinstance(secret_from_request, str):
        return False
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
    """ENHANCED: Handles creating the remote repo (R1) or cloning the existing one (R2+) with better error handling."""
    
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"TDS-TaskReceiver/1.0 ({github_username})"
    }

    # ENHANCED: Increased timeout for large repos and slow networks
    async with httpx.AsyncClient(timeout=90) as client:
        try:
            # 1. CREATE or INITIALIZE REPO / CLONE EXISTING REPO
            if round_index == 1:
                print(f"   -> R1: Setting up remote repository '{repo_name}'...")
                
                # ENHANCED: Check if repository already exists first
                check_response = await client.get(f"{GITHUB_API_BASE}/repos/{github_username}/{repo_name}", headers=headers)
                
                if check_response.status_code == 200:
                    print(f"   -> Repository '{repo_name}' already exists, using existing repository")
                elif check_response.status_code == 404:
                    print(f"   -> Creating new repository '{repo_name}'...")
                    payload = {
                        "name": repo_name, 
                        "private": False, 
                        "auto_init": True,
                        "description": f"Auto-generated repository for task {repo_name}"
                    }
                    
                    create_response = await client.post(f"{GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                    
                    # ENHANCED: Better error handling for different failure modes
                    if create_response.status_code == 422:
                        error_data = create_response.json()
                        if any("already exists" in str(error).lower() for error in error_data.get("errors", [])):
                            print("   -> Repository already exists (race condition), proceeding...")
                        else:
                            raise Exception(f"Repository creation failed: {error_data}")
                    else:
                        create_response.raise_for_status()
                        print("   -> Repository created successfully")
                elif check_response.status_code == 403:
                    raise Exception("GitHub API rate limit exceeded or insufficient permissions")
                else:
                    check_response.raise_for_status()

                # Initialize local git repo
                repo = git.Repo.init(local_path)
                
                # ENHANCED: Better remote handling
                try:
                    origin = repo.remote('origin')
                    origin.set_url(repo_url_auth)
                    print("   -> Updated existing remote 'origin'")
                except Exception:  # Handle all remote-related errors
                    origin = repo.create_remote('origin', repo_url_auth)
                    print("   -> Created remote 'origin'")
                
                print("   -> R1: Local git repository initialized.")
            
            elif round_index >= 2:
                # Crucial part for Round 2: Cloning the existing work into the EMPTY local_path
                print(f"   -> R{round_index}: Cloning existing repository from {repo_url_http}...")
                # local_path is guaranteed to be empty due to the cleanup and directory creation in the main function
                repo = git.Repo.clone_from(repo_url_auth, local_path)
                print(f"   -> R{round_index}: Repository cloned and ready for update.")
            
            return repo

        except httpx.HTTPStatusError as e:
            # ENHANCED: Detailed error handling for different GitHub API errors
            status_code = e.response.status_code
            error_text = e.response.text if e.response else "No response body"
            
            if status_code == 401:
                error_msg = "GitHub authentication failed. Check your GITHUB_TOKEN."
            elif status_code == 403:
                if "rate limit" in error_text.lower():
                    error_msg = "GitHub API rate limit exceeded. Please wait and try again."
                else:
                    error_msg = "GitHub API access forbidden. Check token permissions (needs 'repo' scope)."
            elif status_code == 404:
                error_msg = f"Repository or GitHub user not found. Check GITHUB_USERNAME: '{github_username}'"
            elif status_code == 422:
                error_msg = f"GitHub API validation error: {error_text}"
            else:
                error_msg = f"GitHub API error {status_code}: {error_text}"
            
            print(f"--- [GITHUB API ERROR] {error_msg} ---")
            raise Exception(error_msg)
            
        except git.GitCommandError as e:
            # ENHANCED: Better Git error handling
            error_str = str(e)
            if "authentication failed" in error_str.lower():
                error_msg = "Git authentication failed. Check GITHUB_TOKEN permissions."
            elif "repository not found" in error_str.lower():
                error_msg = f"Git repository not found: {repo_url_http}"
            elif "network" in error_str.lower() or "timeout" in error_str.lower():
                error_msg = "Git operation failed due to network issues. Check internet connection."
            else:
                error_msg = f"Git operation failed: {error_str}"
            
            print(f"--- [GIT ERROR] {error_msg} ---")
            raise Exception(error_msg)


async def commit_and_publish(repo, task_id: str, round_index: int, repo_name: str) -> dict:
    """Handles adding, committing, pushing, and configuring GitHub Pages after files are saved."""

    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"TDS-TaskReceiver/1.0 ({github_username})"
    }
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"

    # ENHANCED: Increased timeout for GitHub Pages operations
    async with httpx.AsyncClient(timeout=120) as client:
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

            # ENHANCED: Wait for GitHub to process the push
            print("   -> Waiting 15 seconds for GitHub to process the push...")
            await asyncio.sleep(15)

            # 2. ENABLE GITHUB PAGES WITH ENHANCED ROBUSTNESS
            print("   -> Configuring GitHub Pages with enhanced error handling...")
            pages_api_url = f"{GITHUB_API_BASE}/repos/{github_username}/{repo_name}/pages"
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            
            # ENHANCED: More retries and better error handling for Pages
            pages_max_retries = 8
            pages_base_delay = 5

            for retry_attempt in range(pages_max_retries):
                try:
                    # Check current Pages status
                    pages_response = await client.get(pages_api_url, headers=headers)
                    
                    if pages_response.status_code == 200:
                        # Pages exists, update configuration
                        print(f"   -> Pages exists, updating configuration (Attempt {retry_attempt + 1})")
                        update_response = await client.put(pages_api_url, json=pages_payload, headers=headers)
                        if update_response.status_code in [200, 204]:
                            print("   -> Pages configuration updated successfully")
                            break
                        else:
                            print(f"   -> Pages update returned {update_response.status_code}, trying POST...")
                            # Fall through to POST creation
                    
                    if pages_response.status_code == 404:
                        # Pages doesn't exist, create it
                        print(f"   -> Creating Pages configuration (Attempt {retry_attempt + 1})")
                        create_response = await client.post(pages_api_url, json=pages_payload, headers=headers)
                        
                        if create_response.status_code in [201, 409]:  # 409 = already exists
                            print("   -> Pages configuration created successfully")
                            break
                        else:
                            create_response.raise_for_status()
                
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    error_text = e.response.text
                    
                    # ENHANCED: Handle specific GitHub Pages errors
                    if status_code == 422:
                        if "main branch must exist" in error_text:
                            delay = pages_base_delay * (retry_attempt + 1)
                            print(f"   -> Branch not ready yet, waiting {delay}s (attempt {retry_attempt + 1})...")
                            await asyncio.sleep(delay)
                            continue
                        elif "pages already exist" in error_text.lower():
                            print("   -> Pages already configured, proceeding...")
                            break
                    elif status_code == 409:
                        print("   -> Pages already exist, proceeding...")
                        break
                    elif status_code == 403 and retry_attempt < pages_max_retries - 1:
                        delay = pages_base_delay * 2
                        print(f"   -> Rate limited, waiting {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        if retry_attempt < pages_max_retries - 1:
                            delay = pages_base_delay * (retry_attempt + 1)
                            print(f"   -> Pages error {status_code}, retrying in {delay}s...")
                            await asyncio.sleep(delay)
                        else:
                            raise
                except Exception as e:
                    if retry_attempt < pages_max_retries - 1:
                        delay = pages_base_delay
                        print(f"   -> Pages setup error: {e}, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        raise
            else:
                print("   -> WARNING: Pages configuration may have failed, but proceeding...")
                # Don't fail the entire task for Pages issues

            # 3. CONSTRUCT RETURN VALUES
            print("   -> Waiting 5 seconds for GitHub Pages deployment...")
            await asyncio.sleep(5)

            pages_url = f"{GITHUB_PAGES_BASE}/{repo_name}/"

            # Verify GitHub Pages is reachable (HTTP 200) with retries
            try:
                check_attempts = 10
                delays = [2, 3, 5, 8, 13, 13, 13, 13, 13, 13]
                for i in range(check_attempts):
                    try:
                        resp = await client.get(pages_url, headers={"Accept": "text/html"})
                        if resp.status_code == 200:
                            print(f"   -> GitHub Pages reachable (200 OK) on attempt {i+1}")
                            break
                        else:
                            print(f"   -> Pages probe got {resp.status_code}, retrying in {delays[i]}s...")
                    except Exception as pe:
                        print(f"   -> Pages probe error: {pe}, retrying in {delays[i]}s...")
                    await asyncio.sleep(delays[i])
            except Exception:
                # Non-fatal
                print("   -> WARNING: Skipping Pages reachability verification due to errors")

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

    # Define system instruction for the model - ENHANCED FOR EVALUATION SUCCESS
    from datetime import datetime
    current_year = datetime.now().year
    system_prompt = (
        "You are an expert full-stack web developer creating FUNCTIONAL web applications for automated testing and GitHub Pages deployment.\n\n"
        "CRITICAL: This code undergoes automated static, dynamic (Playwright), and LLM-based evaluation by instructors.\n\n"
        "RESPONSE FORMAT: Return ONLY a valid JSON object with exactly three keys and no extra text:\n"
        "{\n  \"index.html\": \"<complete self-contained HTML string>\",\n  \"README.md\": \"<complete README markdown>\",\n  \"LICENSE\": \"<MIT license text>\"\n}\n\n"
        "REPOSITORY-LEVEL REQUIREMENTS (MANDATORY FOR EVALUATION):\n"
        "1. LICENSE: Must be valid MIT License with current year (2025) and 'PLACEHOLDER_AUTHOR' as author\n"
        "2. README.md: Must be professional quality with exact sections specified\n"
        "3. Repository: Must be public and GitHub Pages must return HTTP 200\n"
        "4. Security: No secrets, tokens, or credentials anywhere in code or git history\n\n"
        "EVALUATION PIPELINE (MANDATORY - ZERO TOLERANCE):\n"
        "1. PLAYWRIGHT CHECKS: Every 'js:' expression in ACCEPTANCE_CRITERIA must evaluate to true in browser\n"
        "2. REPO-LEVEL CHECKS: MIT LICENSE, professional README.md, public repo, Pages enabled\n"
        "3. LLM STATIC CHECKS: Code quality, documentation quality, implementation correctness\n"
        "4. TEMPLATE VARIABLES: Handle ${seed} and ${result} placeholders correctly in checks\n"
        "5. FUNCTIONAL IMPLEMENTATION: Don't just create elements - implement working functionality\n"
        "6. DATA PROCESSING: Parse and display CSV/JSON data correctly with accurate calculations\n\n"
        "MANDATORY TECHNICAL REQUIREMENTS:\n"
        "- Use Bootstrap 5 or Tailwind CSS via CDN for styling (responsive required)\n"
        "- Include proper meta tags: charset, viewport, description\n"
        "- Handle URL parameters when specified (e.g., ?url=, ?token=)\n"
        "- Process CSV/JSON data with WORKING calculations and displays\n"
        "- Use relative paths for attachments (e.g., './data.csv', './sample.png')\n"
        "- Implement error handling with user-friendly messages\n"
        "- Ensure all interactive elements respond correctly\n"
        "- Parse CSV headers properly and aggregate data accurately\n\n"
        "FRAMEWORK SELECTION:\n"
        "- If any check or brief references Bootstrap (e.g., link[href*='bootstrap']), load Bootstrap 5 via jsdelivr\n"
        "- Otherwise default to Tailwind CSS via CDN\n\n"
        "EVALUATION READINESS:\n"
        "- Code will be boolean-scored by an LLM judge requiring evidence; make correctness obvious:\n"
        "  * Functions used aptly (avoid over-engineering)\n"
        "  * No hard-coded values when data comes from brief/attachments/URL params\n"
        "  * Meaningful variable names; comment non-obvious logic\n"
        "  * Basic error handling with graceful fallbacks/retries where appropriate\n"
        "  * For charts: titles, axis labels, legends, distinct colors\n\n"
        "PLAYWRIGHT COMPLIANCE (CRITICAL):\n"
        "- Use EXACT DOM IDs, classes, text from ACCEPTANCE_CRITERIA - zero substitutions allowed\n"
        "- Handle template variables: replace ${seed} with actual values, ${result} with computed totals\n"
        "- Include metadata: <meta id=\"task-meta\" data-task=\"{task_id}\" data-round=\"{round}\">\n"
        "- Add window.__selfCheck() function with computed values and check results\n"
        "- Prioritize FUNCTIONALITY over aesthetics - working code beats pretty code\n"
        "- Set DOM content during DOMContentLoaded - no user interaction required for basic functionality\n"
        "- Display numbers with precise formatting (parseFloat + toFixed for money, Intl.NumberFormat for counts)\n"
        "- Make CSV parsing work: split by newlines, handle headers, aggregate correctly\n"
        "- Ensure forms submit and display results immediately\n"
        "- Create elements that EXIST and have CONTENT, not empty placeholders\n\n"
        "LICENSE TEMPLATE (MANDATORY - USE EXACTLY):\n"
        "MIT License\n\n"
        f"Copyright (c) {current_year} PLACEHOLDER_AUTHOR\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n\n"
        "CRITICAL ERROR HANDLING (MANDATORY):\n"
        "- Wrap ALL async operations in try/catch with meaningful error messages\n"
        "- When CSV/JSON loading fails, show '0.00' or empty state but still create required DOM elements\n"
        "- Never let JavaScript errors prevent DOM element creation\n"
        "- Validate data before processing: Array.isArray(data) && data.length > 0\n"
        "- Handle malformed CSV gracefully (skip bad rows, continue processing)\n"
        "- Show loading indicators during fetch operations\n"
        "- Provide fallback content when external resources fail\n\n"
        "COMMON INSTRUCTOR EVALUATION FAILURES TO AVOID:\n"
        "- Repo lacks valid MIT LICENSE file in root directory\n"
        "- README.md is unprofessional or missing required sections\n"
        "- Creating #total-sales but never setting its textContent\n"
        "- Parsing CSV but not displaying the calculated results\n"
        "- Using innerHTML when Playwright checks test textContent\n"
        "- Creating form elements that don't respond to submission\n"
        "- Missing error handling that causes undefined to display\n"
        "- Not handling template variables like ${seed} in task descriptions\n"
        "- JavaScript checks fail in Playwright browser evaluation\n"
        "- Pages not accessible (GitHub Pages deployment failure)\n\n"
        "REPOSITORY STRUCTURE (MANDATORY):\n"
        "- /index.html (complete self-contained web app)\n"
        "- /README.md (professional documentation with exact sections)\n"
        "- /LICENSE (valid MIT license)\n"
        "- /[attachments] (data.csv, input.md, etc. as specified)\n"
        "- Repository must be PUBLIC\n"
        "- GitHub Pages must be ENABLED and return HTTP 200\n\n"
        "OUTPUT CONSTRAINT: Return ONLY the JSON object - no explanatory text, no code blocks, no markdown formatting."
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
                    "max_tokens": 8000,  # Increased to reduce truncation risk per evaluation guidance
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
    pages_url: str,
    deadline_epoch: float
) -> bool:
    """
    Calls the evaluation_url to notify the server that the code has been deployed.
    Retries with exponential backoff until success or deadline reached.
    """
    import time

    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url  
    }

    attempt = 0
    delay = 1

    print(f"--- [NOTIFICATION] Attempting to notify server at {evaluation_url} ---")

    while True:
        now = time.time()
        if now >= deadline_epoch:
            print("--- [NOTIFICATION] Deadline reached before successful notification ---")
            return False
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(evaluation_url, json=payload, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                print(f"--- [NOTIFICATION] Successfully notified server. Response: {response.status_code} ---")
                return True
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            attempt += 1
            # Cap delay at 60s to avoid very long sleeps
            sleep_for = min(delay, 60)
            print(f"--- [NOTIFICATION] Error on attempt {attempt}: {e}. Retrying in {sleep_for}s ---")
            await asyncio.sleep(sleep_for)
            delay = delay * 2 if delay < 60 else 60
            continue


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
            print(f"    -> ERROR saving attachment {filename}: {e}")
            print(f"    -> Skipping invalid attachment and continuing with others")
            # Continue with other attachments instead of raising exception
            continue

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
            # ROUND 2+ - MODIFICATION/UPDATE TASK (ENHANCED FOR EVALUATION)
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            checks_formatted = json.dumps(task_data.checks, indent=2)
            
            llm_prompt = (
                f"MODIFICATION TASK - ROUND {round_index}\n\n"
                f"TASK OBJECTIVE: {brief}\n\n"
                f"CRITICAL: This is Round {round_index} - automatic testing will fail if ANY previous functionality breaks.\n\n"
                f"CONTEXT: This is Round {round_index} update to task {task_id}. You MUST preserve ALL previous functionality while adding new features.\n\n"
                f"ACCEPTANCE_CRITERIA (MANDATORY - THESE DETERMINE PASS/FAIL):\n"
                f"The evaluator will run these exact JavaScript expressions. ALL must evaluate to true:\n"
                f"{checks_formatted}\n\n"
                f"CHECKS FORMAT: If a check starts with 'js:', evaluate the substring after 'js:' as the JavaScript expression to satisfy.\n"
                f"REPO-LEVEL CHECKS: Repository requirements (MIT LICENSE, README.md, Pages) must remain satisfied.\n\n"
                f"BACKWARD COMPATIBILITY WARNING: Previous round elements and functionality MUST continue to work.\n\n"
                f"BACKWARD COMPATIBILITY (CRITICAL - ZERO TOLERANCE):\n"
                f"1. PRESERVE EXISTING ELEMENTS: All DOM IDs, classes from Round 1 must remain functional\n"
                f"2. MAINTAIN CALCULATIONS: Previous numeric calculations must still work correctly\n"
                f"3. PRESERVE DATA PROCESSING: CSV/JSON parsing from Round 1 must continue working\n"
                f"4. FUNCTIONAL INHERITANCE: All interactive features from Round 1 must be preserved\n"
                f"5. UPDATE METADATA: Update to include current round:\n"
                f"  <meta id=\"task-meta\" data-task=\"{task_id}\" data-round=\"{round_index}\" data-timestamp=\"{timestamp}\">\n\n"
                f"BREAKING CHANGES TO AVOID:\n"
                f"- Renaming or removing existing DOM IDs\n"
                f"- Changing the format of numeric displays (e.g., from '60.00' to '$60')\n"
                f"- Modifying existing event handlers or form behavior\n"
                f"- Breaking CSV/JSON parsing that worked in Round 1\n"
                f"- Removing functionality that previous acceptance criteria depend on\n\n"
                f"DELIVERABLES:\n"
                f"- Updated 'index.html' with new functionality (complete, self-contained file)\n"
                f"- Revised 'README.md' with same headings, updated content\n"
                f"- Updated 'LICENSE' file (MIT License with current year)\n\n"
                f"IMPLEMENTATION RULES:\n"
                f"- Surround new/changed code with comments: /* ROUND_{round_index}_START */ ... /* ROUND_{round_index}_END */\n"
                f"- If using localStorage, namespace keys: localStorage.setItem('app-{task_id}-key', value)\n"
                f"- Preserve existing error handling while adding new error cases\n"
                f"- Maintain responsive design and accessibility features\n\n"
                f"README UPDATE REQUIREMENTS:\n"
                f"Keep exact headings from Round 1:\n"
                f"## Summary (update to describe new features)\n"
                f"## Setup (update if new setup steps required)\n"
                f"## Usage (document new features and parameters)\n"
                f"## Files (add any new files)\n"
                f"## Testing (include ALL acceptance criteria from all rounds)\n"
                f"## License (same as before)\n\n"
                f"SELF-TEST UPDATE (MANDATORY - ENHANCED VALIDATION):\n"
                f"Update window.__selfCheck() to validate ALL rounds:\n"
                f"```javascript\n"
                f"window.__selfCheck = () => {{\n"
                f"  const results = {{\n"
                f"    task: '{task_id}',\n"
                f"    round: {round_index},\n"
                f"    timestamp: new Date().toISOString(),\n"
                f"    elements: {{}},\n"
                f"    computed: {{}},\n"
                f"    checks: {{}},\n"
                f"    backwardCompatibility: {{}}\n"
                f"  }};\n"
                f"  \n"
                f"  // Validate current round acceptance criteria (strip 'js:' prefix if present)\n"
                f"  const rawChecks = {json.dumps(task_data.checks)};\n"
                f"  const currentChecks = rawChecks.map(c => (typeof c === 'string' && c.trim().startsWith('js:')) ? c.trim().slice(3).trim() : c);\n"
                f"  currentChecks.forEach((expr, i) => {{\n"
                f"    try {{\n"
                f"      results.checks[`round_{round_index}_check_${{i}}`] = {{\n"
                f"        expression: expr,\n"
                f"        result: eval(expr),\n"
                f"        error: null\n"
                f"      }};\n"
                f"    }} catch (error) {{\n"
                f"      results.checks[`round_{round_index}_check_${{i}}`] = {{\n"
                f"        expression: expr,\n"
                f"        result: false,\n"
                f"        error: error.message\n"
                f"      }};\n"
                f"    }}\n"
                f"  }});\n"
                f"  \n"
                f"  // Validate all elements referenced in checks exist\n"
                f"  const allSelectors = currentChecks\n"
                f"    .filter(check => check.includes('querySelector'))\n"
                f"    .map(check => check.match(/querySelector\\(['\"](.*?)['\"]\\)/)?.[1])\n"
                f"    .filter(Boolean);\n"
                f"  \n"
                f"  allSelectors.forEach(selector => {{\n"
                f"    const element = document.querySelector(selector);\n"
                f"    results.elements[selector] = {{\n"
                f"      exists: !!element,\n"
                f"      content: element ? element.textContent.trim() : null,\n"
                f"      visible: element ? !element.hidden && element.offsetParent !== null : false\n"
                f"    }};\n"
                f"  }});\n"
                f"  \n"
                f"  return results;\n"
                f"}};\n"
                f"\n"
                f"// Enhanced validation with timing\n"
                f"document.addEventListener('DOMContentLoaded', () => {{\n"
                f"  setTimeout(() => {{\n"
                f"    const validation = window.__selfCheck();\n"
                f"    console.log('Round {round_index} validation results:', validation);\n"
                f"    \n"
                f"    // Report failures immediately\n"
                f"    const failures = Object.entries(validation.checks)\n"
                f"      .filter(([key, check]) => !check.result);\n"
                f"    \n"
                f"    if (failures.length > 0) {{\n"
                f"      console.error(`ROUND {round_index} VALIDATION FAILED:`, failures);\n"
                f"      failures.forEach(([key, check]) => {{\n"
                f"        console.error(`FAILED: ${{check.expression}} - ${{check.error || 'Evaluated to false'}}`);\n"
                f"      }});\n"
                f"    }} else {{\n"
                f"      console.log('All Round {round_index} checks passed');\n"
                f"    }}\n"
                f"  }}, 1500); // Extra time for Round 2 complexity\n"
                f"}});\n"
                f"```\n\n"
                f"QUALITY ASSURANCE:\n"
                f"- Ensure new features don't break existing functionality\n"
                f"- Test interactions between old and new features\n"
                f"- Maintain data consistency across feature updates\n"
                f"- Preserve user state and preferences during modifications\n"
                f"- Add comprehensive error handling for new integrations"
            )
        else:
            # ROUND 1 - INITIAL CREATION TASK (ENHANCED FOR EVALUATION)
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            checks_formatted = json.dumps(task_data.checks, indent=2)
            
            llm_prompt = (
                f"NEW APPLICATION TASK - ROUND 1\n\n"
                f"TASK OBJECTIVE: {brief}\n\n"
                f"CRITICAL SUCCESS REQUIREMENT: This code will be automatically tested. Failure = 0 points.\n\n"
                f"ACCEPTANCE_CRITERIA (MANDATORY - THESE DETERMINE PASS/FAIL):\n"
                f"The evaluator will run these exact JavaScript expressions in the browser. They must ALL evaluate to true:\n"
                f"{checks_formatted}\n\n"
                f"CHECKS FORMAT: If a check starts with 'js:', evaluate the substring after 'js:' as the JavaScript expression to satisfy.\n"
                f"REPO-LEVEL CHECKS: Requirements like MIT LICENSE and professional README.md must be satisfied in the repository.\n\n"
                f"VALIDATION WARNING: Each check must pass or the entire task fails evaluation.\n"
                f"DO NOT ASSUME - Implement exactly what each check verifies.\n\n"
                f"METADATA REQUIREMENTS:\n"
                f"- Task ID: {task_id}\n"
                f"- Round: 1\n"
                f"- Attachments: {attachment_list_for_llm_prompt}\n\n"
                f"RENDERING RULES (MANDATORY - ZERO TOLERANCE):\n"
                f"- Use exact DOM IDs, classes, and text from ACCEPTANCE_CRITERIA above\n"
                f"- Do NOT rename, modify, or substitute any IDs or text literals referenced in the checks\n"
                f"- Include this hidden metadata element exactly (required for evaluation):\n"
                f"  <meta id=\"task-meta\" data-task=\"{task_id}\" data-round=\"1\" data-timestamp=\"{timestamp}\">\n"
                f"- All values referenced by ACCEPTANCE_CRITERIA must be set during DOMContentLoaded\n"
                f"- Attachments are available at relative paths: './filename.ext'\n"
                f"- Elements MUST exist and be accessible before evaluation runs\n\n"
                f"DELIVERABLES:\n"
                f"- Complete 'index.html' file (fully self-contained with embedded styles/scripts)\n"
                f"- Professional 'README.md' with required headings (see below)\n"
                f"- MIT 'LICENSE' file with current year\n\n"
                f"README REQUIREMENTS (MANDATORY PROFESSIONAL QUALITY):\n"
                f"Use these exact headings for instructor LLM evaluation:\n\n"
                f"## Summary\n"
                f"Clear, professional description of app functionality and purpose. Mention which acceptance criteria are implemented.\n\n"
                f"## Setup\n"
                f"Step-by-step instructions to view on GitHub Pages or run locally. Include live Pages URL.\n\n"
                f"## Usage\n"
                f"Detailed usage instructions including URL parameters, form inputs, and expected outputs.\n\n"
                f"## Files\n"
                f"Complete file structure with descriptions:\n"
                f"- index.html: Main application file\n"
                f"- README.md: This documentation\n"
                f"- LICENSE: MIT license\n"
                f"- [list all attachments and their purposes]\n\n"
                f"## Testing\n"
                f"Document ALL acceptance criteria from task and how each is implemented:\n"
                f"{checks_formatted}\n"
                f"Explain the DOM elements, calculations, and functionality that satisfy each check.\n\n"
                f"## License\n"
                f"This project is licensed under the MIT License - see the LICENSE file for details.\n\n"
                f"IMPLEMENTATION PATTERNS (MANDATORY - USE THESE EXACT PATTERNS):\n\n"
                f"CSV Parsing Pattern (REQUIRED for CSV tasks):\n"
                f"```javascript\n"
                f"document.addEventListener('DOMContentLoaded', async () => {{\n"
                f"  try {{\n"
                f"    const csvText = await fetch('./data.csv').then(r => {{\n"
                f"      if (!r.ok) throw new Error(`HTTP ${{r.status}}: ${{r.statusText}}`);\n"
                f"      return r.text();\n"
                f"    }});\n"
                f"    \n"
                f"    // Parse CSV - MUST handle headers properly\n"
                f"    const lines = csvText.trim().split('\\n');\n"
                f"    const headers = lines[0].split(',').map(h => h.trim());\n"
                f"    const rows = lines.slice(1).map(line => {{\n"
                f"      const values = line.split(',').map(v => v.trim());\n"
                f"      const obj = {{}};\n"
                f"      headers.forEach((header, i) => obj[header] = values[i]);\n"
                f"      return obj;\n"
                f"    }});\n"
                f"    \n"
                f"    // CALCULATE AND DISPLAY IMMEDIATELY\n"
                f"    const total = rows.reduce((sum, row) => sum + parseFloat(row.sales || 0), 0);\n"
                f"    document.querySelector('#total-sales').textContent = total.toFixed(2);\n"
                f"    \n"
                f"  }} catch (error) {{\n"
                f"    console.error('CSV processing failed:', error);\n"
                f"    document.querySelector('#total-sales').textContent = '0.00';\n"
                f"    // MUST still create required elements even on error\n"
                f"  }}\n"
                f"}});\n"
                f"```\n\n"
                f"Number Display Pattern (MANDATORY for numeric displays):\n"
                f"```javascript\n"
                f"// ALWAYS ensure element exists and has value\n"
                f"const element = document.querySelector('#total-sales');\n"
                f"if (element) {{\n"
                f"  element.textContent = Number(total || 0).toFixed(2);\n"
                f"}} else {{\n"
                f"  console.error('Required element #total-sales not found');\n"
                f"}}\n"
                f"```\n\n"
                f"SELF-TEST FUNCTION (MANDATORY - INCLUDE EXACTLY):\n"
                f"Add this function to validate your implementation:\n"
                f"```javascript\n"
                f"window.__selfCheck = () => {{\n"
                f"  const results = {{\n"
                f"    task: '{task_id}',\n"
                f"    round: 1,\n"
                f"    timestamp: new Date().toISOString(),\n"
                f"    elements: {{}},\n"
                f"    computed: {{}},\n"
                f"    checks: {{}}\n"
                f"  }};\n"
                f"  \n"
                f"  // Validate all required elements exist\n"
                f"  const requiredElements = {json.dumps([check.split("'")[1] if "'" in check and "querySelector" in check else None for check in task_data.checks if "querySelector" in check])};\n"
                f"  requiredElements.forEach(selector => {{\n"
                f"    if (selector) {{\n"
                f"      const element = document.querySelector(selector);\n"
                f"      results.elements[selector] = {{\n"
                f"        exists: !!element,\n"
                f"        content: element ? element.textContent.trim() : null,\n"
                f"        value: element ? element.value : null\n"
                f"      }};\n"
                f"    }}\n"
                f"  }});\n"
                f"  \n"
                f"  // Test each acceptance criteria (strip 'js:' prefix if present)\n"
                f"  try {{\n"
                f"    const rawChecks = {json.dumps(task_data.checks)};\n"
                f"    const checks = rawChecks.map(c => (typeof c === 'string' && c.trim().startsWith('js:')) ? c.trim().slice(3).trim() : c);\n"
                f"    checks.forEach((expr, i) => {{\n"
                f"      try {{\n"
                f"        results.checks[`check_${{i}}`] = {{\n"
                f"          expression: expr,\n"
                f"          result: eval(expr),\n"
                f"          error: null\n"
                f"        }};\n"
                f"      }} catch (error) {{\n"
                f"        results.checks[`check_${{i}}`] = {{\n"
                f"          expression: expr,\n"
                f"          result: false,\n"
                f"          error: error.message\n"
                f"        }};\n"
                f"      }}\n"
                f"    }});\n"
                f"  }} catch (error) {{\n"
                f"    results.error = error.message;\n"
                f"  }}\n"
                f"  \n"
                f"  return results;\n"
                f"}};\n"
                f"\n"
                f"// Auto-run validation on load\n"
                f"document.addEventListener('DOMContentLoaded', () => {{\n"
                f"  setTimeout(() => {{\n"
                f"    const validation = window.__selfCheck();\n"
                f"    console.log('Self-validation results:', validation);\n"
                f"    // Log any failed checks\n"
                f"    Object.entries(validation.checks).forEach(([key, check]) => {{\n"
                f"      if (!check.result) {{\n"
                f"        console.warn(`FAILED CHECK: ${{check.expression}} - ${{check.error || 'Evaluated to false'}}`);\n"
                f"      }}\n"
                f"    }});\n"
                f"  }}, 1000); // Wait 1s for async operations\n"
                f"}});\n"
                f"```\n\n"
                f"CRITICAL SUCCESS FACTORS (ZERO TOLERANCE FOR FAILURE):\n"
                f"1. MANDATORY ELEMENT CREATION: Every element referenced in ACCEPTANCE_CRITERIA must exist\n"
                f"2. FUNCTIONAL IMPLEMENTATION: Don't just create elements - make them work\n"
                f"3. DATA PROCESSING: If CSV/JSON is provided, parse it and display results correctly\n"
                f"4. NUMERIC ACCURACY: Calculations must be precise (use parseFloat, toFixed)\n"
                f"5. ERROR RESILIENCE: App must work even if attachments fail to load\n"
                f"6. IMMEDIATE EXECUTION: Set values during DOMContentLoaded, not on user action\n"
                f"7. VALIDATION COMPLIANCE: Every ACCEPTANCE_CRITERIA check must pass\n\n"
                f"COMMON FAILURE MODES TO AVOID:\n"
                f"- Creating elements but not setting their content\n"
                f"- Displaying '$0.00' instead of actual calculated totals\n"
                f"- Using innerHTML when textContent is tested\n"
                f"- Missing error handling for fetch operations\n"
                f"- Not handling CSV headers properly\n"
                f"- Creating elements that exist but have no functionality\n\n"
                f"SUCCESS VALIDATION CHECKLIST:\n"
                f"Before submitting, manually verify:\n"
                f"- All IDs from ACCEPTANCE_CRITERIA exist in DOM\n"
                f"- All numeric displays show actual calculated values, not 0\n"
                f"- CSV data is parsed and aggregated correctly\n"
                f"- Forms and interactive elements respond to user input\n"
                f"- Error messages appear for missing files\n"
                f"- App works in GitHub Pages static environment"
            )
        
        # Add attachment context with enhanced instructions
        if attachment_list_str:
            llm_prompt += (
                f"\n\nATTACHMENT FILES AVAILABLE:\n"
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
                "- All interactive elements respond correctly\n"
                "- Data transforms from raw format to user-friendly display\n"
                "- Error states provide clear, actionable feedback\n"
                "- Loading states prevent user confusion\n"
                "- Mobile responsiveness works across screen sizes\n"
                "- Color contrast meets accessibility standards\n"
                "- Code is well-structured and commented for maintenance"
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
        
        # 4. VALIDATE GENERATED CODE FOR SECRETS
        # Project requirement: avoid secrets in git history (trufflehog, gitleaks)
        for filename, content in generated_files.items():
            if any(keyword in content.lower() for keyword in ['api_key', 'secret', 'token', 'password', 'private']):
                if not any(safe_term in content.lower() for safe_term in ['placeholder', 'example', 'dummy', 'your_', 'insert_']):
                    print(f"--- [SECRET WARNING] Potential secret detected in {filename} ---")
        # 5. Save Generated Code Locally
        # This overwrites the cloned files (index.html, README.md, LICENSE)
        await save_generated_files_locally(task_id, generated_files)
        
        # 6. Save Attachments Locally
        # This adds attachments (like data.csv) to the local directory
        # The attachment saving now happens *after* the clone/init, resolving the Round 2 error.
        await save_attachments_locally(local_path, attachments)

        # 6. VALIDATE REPOSITORY CREATION TIME
        # Project requirement: repo must be created after task request time
        repo_created_after_request = True  # Placeholder - in real implementation, compare timestamps
        if not repo_created_after_request:
            print("--- [WARNING] Repository creation timestamp validation failed ---")
        # 8. COMMIT AND PUBLISH
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
        
        # 9. Notify the Evaluation Server
        deadline_epoch = start_time + MAX_PROCESSING_TIME - 5  # small buffer
        await notify_evaluation_server(
            evaluation_url=evaluation_url, 
            email=email,
            task_id=task_id, 
            round_index=round_index,
            nonce=nonce, 
            repo_url=repo_url,
            commit_sha=commit_sha,
            pages_url=pages_url,
            deadline_epoch=deadline_epoch
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

    