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

## Vercel Deployment Instructions

This project is optimized for deployment on [Vercel](https://vercel.com/). Follow these steps to deploy the application:

### 1. Fork the Repository

Start by forking this repository to your own GitHub account.

### 2. Create a Vercel Project

- Go to your Vercel dashboard and click on "Add New... > Project".
- Import the forked repository from your GitHub account.
- Vercel will automatically detect that this is a Python project and configure the build settings. The `vercel.json` file in this repository provides the necessary configuration.

### 3. Configure Environment Variables

This is a critical step. The application requires several secret keys and configuration variables to be set. In your Vercel project settings, navigate to the "Environment Variables" section and add the following:

| Key                 | Value                                     | Description                                                                 |
| ------------------- | ----------------------------------------- | --------------------------------------------------------------------------- |
| `OPENAI_API_KEY`    | `your_openai_api_key`                     | Your API key for the OpenAI or compatible LLM service.                      |
| `GITHUB_TOKEN`      | `your_github_personal_access_token`       | A GitHub Personal Access Token with `repo` and `workflow` scopes.           |
| `STUDENT_SECRET`    | `a_secure_secret_string`                  | A secret string that the application uses to verify incoming requests.      |
| `GITHUB_USERNAME`   | `your_github_username`                    | Your GitHub username, used for creating and deploying to repositories.      |

**Important:** Ensure that these variables are set as "Secret" to protect them.

### 4. Deploy

After configuring the environment variables, trigger a deployment from the Vercel dashboard. Vercel will build and deploy the application. Once the deployment is complete, you will be provided with a URL where the application is running.

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
   ```

5. **Run the application:**
   ```bash
   uvicorn main:app --reload
   ```
   The application will be available at `http://127.0.0.1:8000`.
