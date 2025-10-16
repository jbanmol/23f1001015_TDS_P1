# GITHUB API ROBUSTNESS ENHANCEMENTS - Apply to main.py

async def enhanced_setup_local_repo(local_path: str, repo_name: str, repo_url_auth: str, repo_url_http: str, round_index: int):
    """ENHANCED: Handles creating the remote repo (R1) or cloning the existing one (R2+) with better error handling."""
    
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"TDS-TaskReceiver/1.0 ({github_username})"  # Required by GitHub API
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
                    repo_exists = True
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
                            repo_exists = True
                        else:
                            raise Exception(f"Repository creation failed: {error_data}")
                    else:
                        create_response.raise_for_status()
                        repo_exists = False
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
                except git.exc.NoSuchRemoteError:
                    origin = repo.create_remote('origin', repo_url_auth)
                    print("   -> Created remote 'origin'")
                
                print("   -> R1: Local git repository initialized.")
                
            elif round_index >= 2:
                print(f"   -> R{round_index}: Cloning existing repository...")
                
                # ENHANCED: Pre-check repository exists before cloning
                check_response = await client.get(f"{GITHUB_API_BASE}/repos/{github_username}/{repo_name}", headers=headers)
                
                if check_response.status_code == 404:
                    raise Exception(f"Repository '{repo_name}' not found. Round {round_index} requires existing repository from Round 1.")
                elif check_response.status_code == 403:
                    raise Exception("GitHub API rate limit exceeded or repository is private/inaccessible")
                
                check_response.raise_for_status()
                
                # ENHANCED: Clone with retries and better error handling
                max_clone_retries = 3
                for attempt in range(max_clone_retries):
                    try:
                        repo = git.Repo.clone_from(repo_url_auth, local_path)
                        print(f"   -> R{round_index}: Repository cloned successfully on attempt {attempt + 1}")
                        break
                    except git.GitCommandError as clone_error:
                        if attempt < max_clone_retries - 1:
                            wait_time = 5 * (attempt + 1)
                            print(f"   -> Clone failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            raise Exception(f"Failed to clone repository after {max_clone_retries} attempts: {clone_error}")
            
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

async def enhanced_commit_and_publish(repo, task_id: str, round_index: int, repo_name: str) -> dict:
    """ENHANCED: Handles committing, pushing, and GitHub Pages with robust error handling."""

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
            # 1. ADD, COMMIT, AND PUSH FILES WITH ENHANCED ERROR HANDLING
            print("   -> Adding files to git...")
            repo.git.add(A=True)
            
            # Check if there are any changes to commit
            if not repo.index.diff("HEAD"):
                print("   -> No changes detected, creating empty commit to trigger Pages...")
                commit_message = f"Task {task_id} - Round {round_index}: Trigger deployment"
                repo.git.commit("--allow-empty", "-m", commit_message)
            else:
                commit_message = f"Task {task_id} - Round {round_index}: LLM-generated app update/creation"
                repo.index.commit(commit_message)
            
            commit_sha = repo.head.object.hexsha
            print(f"   -> Files committed. SHA: {commit_sha}")

            # ENHANCED: Better branch handling
            try:
                current_branch = repo.active_branch.name
                if current_branch != 'main':
                    print(f"   -> Renaming branch from '{current_branch}' to 'main'...")
                    repo.git.branch('-M', 'main')
            except:
                # If we can't determine current branch, ensure we're on main
                repo.git.checkout('-b', 'main')
                print("   -> Created and switched to 'main' branch")

            # ENHANCED: Push with retries
            max_push_retries = 3
            for attempt in range(max_push_retries):
                try:
                    repo.git.push('--set-upstream', 'origin', 'main', force=True)
                    print("   -> Changes pushed to remote 'main' branch successfully")
                    break
                except git.GitCommandError as push_error:
                    if attempt < max_push_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        print(f"   -> Push failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise Exception(f"Failed to push after {max_push_retries} attempts: {push_error}")

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
            print("   -> Waiting 10 seconds for Pages deployment to start...")
            await asyncio.sleep(10)

            pages_url = f"{GITHUB_PAGES_BASE}/{repo_name}/"

            return {
                "repo_url": repo_url_http,
                "commit_sha": commit_sha,
                "pages_url": pages_url
            }

        except git.GitCommandError as e:
            error_msg = f"Git operation failed: {str(e)}"
            print(f"--- [GIT ERROR] {error_msg} ---")
            raise Exception(error_msg)
        except httpx.HTTPStatusError as e:
            error_msg = f"GitHub API error {e.response.status_code}: {e.response.text}"
            print(f"--- [GITHUB API ERROR] {error_msg} ---")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Deployment failed: {str(e)}"
            print(f"--- [DEPLOYMENT ERROR] {error_msg} ---")
            raise Exception(error_msg)

# USAGE: Replace the existing setup_local_repo and commit_and_publish functions
# with enhanced_setup_local_repo and enhanced_commit_and_publish