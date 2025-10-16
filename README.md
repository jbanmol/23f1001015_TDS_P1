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

FastAPI service that receives coding task assignments, uses an LLM to generate web applications, and automatically deploys them to GitHub Pages.

**Live Endpoint:** https://jbanmol-tds-p1.hf.space

## How It Works

1. **Receives Task** - POST request to `/ready` with task brief and acceptance criteria
2. **Generates Code** - LLM creates complete web app (HTML/CSS/JS) based on requirements
3. **Deploys to GitHub** - Creates/updates repository and enables GitHub Pages
4. **Notifies Evaluator** - Sends deployment URL back for automated testing

## Deployment

### HuggingFace Spaces (Recommended)

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Docker SDK**
3. Push this repository to your Space
4. Add environment variables in Space settings:
   ```
   OPENAI_API_KEY=your_key
   GITHUB_TOKEN=your_token
   GITHUB_USERNAME=your_username
   STUDENT_SECRET=your_secret
   OPENAI_API_BASE=https://aipipe.org/openrouter/v1
   ```
5. Space will auto-deploy at `https://your-username-space-name.hf.space`

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with credentials
cat > .env << EOF
OPENAI_API_KEY=your_key
GITHUB_TOKEN=your_token
GITHUB_USERNAME=your_username
STUDENT_SECRET=your_secret
EOF

# Run server
uvicorn main:app --reload
```

## API Endpoints

- `GET /` - Health check
- `GET /status` - View last received task
- `POST /ready` - Main task receiver endpoint
- `GET /docs` - Interactive API documentation

## Features

- ✅ Multi-round task support (create + modify)
- ✅ Automatic GitHub repository creation
- ✅ GitHub Pages deployment with verification
- ✅ Secret validation for security
- ✅ Comprehensive error handling
- ✅ Attachment processing (CSV, images, etc.)
- ✅ 10-minute processing time limit

## Project Structure

```
├── main.py              # FastAPI app and orchestration logic
├── models.py            # Pydantic models for request validation
├── config.py            # Configuration and environment variables
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration for deployment
└── tests/              # Test suite
```

## Requirements

- Python 3.10+
- GitHub Personal Access Token (repo scope)
- OpenAI-compatible API key
- Git installed in environment
