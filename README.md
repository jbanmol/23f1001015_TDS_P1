---
title: TDS P1 - LLM Code Deployment
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: main.py
pinned: false
---

# Automated Task Receiver & Processor

This project is a FastAPI application that serves as an endpoint for receiving task assignments. Upon receiving a task, it triggers an AI-powered code generation process and deploys the resulting web application to GitHub Pages.

**Live Endpoint:** https://jbanmol-tds-p1.hf.space

## HuggingFace Spaces Deployment Instructions

This project is optimized for deployment on [HuggingFace Spaces](https://huggingface.co/spaces). Follow these steps to deploy the application:

### 1. Fork the Repository

Start by forking this repository to your own GitHub account.

### 2. Create a HuggingFace Space

- Go to https://huggingface.co/new-space
- Choose a name for your Space
- Select **Docker** as the SDK
- Import the forked repository or upload files directly

### 3. Configure Environment Variables

This is a critical step. The application requires several secret keys and configuration variables to be set. In your Space settings, navigate to the "Settings > Repository secrets" section and add the following:

| Key                 | Value                                     | Description                                                                 |
| ------------------- | ----------------------------------------- | --------------------------------------------------------------------------- |
| `OPENAI_API_KEY`    | `your_openai_api_key`                     | Your API key for the OpenAI or compatible LLM service.                      |
| `GITHUB_TOKEN`      | `your_github_personal_access_token`       | A GitHub Personal Access Token with `repo` and `workflow` scopes.           |
| `STUDENT_SECRET`    | `a_secure_secret_string`                  | A secret string that the application uses to verify incoming requests.      |
| `GITHUB_USERNAME`   | `your_github_username`                    | Your GitHub username, used for creating and deploying to repositories.      |
| `OPENAI_API_BASE`   | `https://aipipe.org/openrouter/v1`        | (Optional) Custom API base URL for OpenAI-compatible services.             |

**Important:** Ensure that these variables are set as secrets to protect them.

### 4. Deploy

After configuring the environment variables, HuggingFace will automatically build and deploy the application using the included `Dockerfile`. Once the deployment is complete, your Space will be available at `https://your-username-space-name.hf.space`.

## Local Development (Optional)

To run the application locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your-forked-repo-url>
   cd <repository-name>
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file:**
   Create a file named `.env` in the root of the project and add the same environment variables as listed above:
   ```
   OPENAI_API_KEY="your_openai_api_key"
   GITHUB_TOKEN="your_github_personal_access_token"
   STUDENT_SECRET="a_secure_secret_string"
   GITHUB_USERNAME="your_github_username"
   OPENAI_API_BASE="https://aipipe.org/openrouter/v1"
   ```

5. **Run the application:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 7860
   ```
   The application will be available at `http://127.0.0.1:7860`.

## API Endpoints

- `GET /` - Health check endpoint
- `GET /status` - View last received task details
- `POST /ready` - Main endpoint for receiving task assignments
- `GET /docs` - Interactive API documentation (Swagger UI)

## Features

- Multi-round task support (create and modify workflow)
- Automatic GitHub repository creation and management
- GitHub Pages deployment with verification
- Secret-based authentication for secure API access
- Comprehensive error handling and validation
- Support for file attachments (CSV, images, JSON, etc.)
- 10-minute processing time limit per task

## Project Structure

```
tds-p1-final/
├── main.py                    # FastAPI application and orchestration logic
├── models.py                  # Pydantic models for request validation
├── config.py                  # Configuration and environment variables
├── Dockerfile                 # Docker configuration for HuggingFace deployment
├── requirements.txt           # Python dependencies
├── test_enhanced_sytem.sh     # End-to-end system test script
└── README.md                  # This file
```

## Requirements

- Python 3.10+
- FastAPI framework
- OpenAI-compatible API access (GPT-4o-mini recommended)
- GitHub account with Personal Access Token
- Git installed in deployment environment
