# HuggingFace Spaces Deployment Guide

## Quick Setup

1. **Create a new Space on HuggingFace**:
   - Go to https://huggingface.co/new-space
   - Choose **Docker SDK** (not Gradio/Streamlit)
   - Set visibility to **Public** or **Private** as needed

2. **Upload your code**:
   - Clone this repository
   - Push to your HuggingFace Space repository

3. **Configure Environment Variables**:
   In your Space settings, add these secrets:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GITHUB_TOKEN=your_github_token  
   STUDENT_SECRET=your_secret_string
   GITHUB_USERNAME=your_github_username
   OPENAI_API_BASE=https://aipipe.org/openrouter/v1
   ```

4. **Deploy**:
   - HuggingFace will automatically build and deploy using the Dockerfile
   - Your API will be available at: `https://your-username-space-name.hf.space`

## Testing Your Deployment

Once deployed, test your endpoint:

```bash
curl -X POST https://your-username-space-name.hf.space/ready \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "secret": "your_secret_string", 
    "task": "test-task-123",
    "round": 1,
    "nonce": "test-nonce",
    "brief": "Create a simple hello world web page",
    "checks": ["Page displays hello world"],
    "evaluation_url": "https://httpbin.org/post",
    "attachments": []
  }'
```

## Key Features

- ✅ FastAPI server running on port 7860 (HuggingFace standard)
- ✅ Docker-based deployment with all dependencies
- ✅ Secure environment variable handling
- ✅ Cross-platform git executable detection
- ✅ Comprehensive error handling and logging
- ✅ Multi-round task support (build + revise)

## API Endpoints

- `GET /` - Health check
- `GET /status` - Last task status  
- `POST /ready` - Main task receiver endpoint
- `GET /docs` - Interactive API documentation