#!/usr/bin/env python3
"""
Simple health check script to verify the deployment works
"""

def check_imports():
    """Test all critical imports"""
    try:
        print("✅ Testing imports...")
        import fastapi
        print(f"  - FastAPI: {fastapi.__version__}")
        
        import uvicorn
        print(f"  - Uvicorn: {uvicorn.__version__}")
        
        import pydantic
        print(f"  - Pydantic: {pydantic.__version__}")
        
        import httpx
        print(f"  - HTTPX: {httpx.__version__}")
        
        import git
        print(f"  - GitPython: {git.__version__}")
        
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def check_git_executable():
    """Test git executable detection"""
    try:
        import shutil
        git_path = shutil.which('git')
        print(f"✅ Git executable found at: {git_path}")
        return True
    except Exception as e:
        print(f"❌ Git executable error: {e}")
        return False

def check_config():
    """Test configuration loading"""
    try:
        from config import get_settings
        settings = get_settings()
        print("✅ Configuration loaded successfully")
        print(f"  - GitHub Username: {getattr(settings, 'GITHUB_USERNAME', 'NOT_SET')}")
        print(f"  - OpenAI API Base: {getattr(settings, 'OPENAI_API_BASE', 'NOT_SET')}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_app_creation():
    """Test FastAPI app creation"""
    try:
        from main import app
        print("✅ FastAPI app created successfully")
        print(f"  - Routes: {len(app.routes)}")
        return True
    except Exception as e:
        print(f"❌ App creation error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 HuggingFace Deployment Health Check")
    print("=" * 50)
    
    all_checks = [
        check_imports(),
        check_git_executable(), 
        check_config(),
        check_app_creation()
    ]
    
    if all(all_checks):
        print("\n🎉 All checks passed! Deployment should work.")
        exit(0)
    else:
        print("\n❌ Some checks failed. Review the errors above.")
        exit(1)