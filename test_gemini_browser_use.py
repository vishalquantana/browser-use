import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables from the current directory
load_dotenv('.env')

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

def main():
    # Check if Google API key is set
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY is not set in .env file")
        print("Please add your Google API key to the .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        return
    
    print("✅ Google API key found!")
    print(f"✅ Using API key: {api_key[:10]}...")
    
    # Initialize the Gemini model (using gemini-2.0-flash-exp as specified)
    try:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp', 
            api_key=SecretStr(api_key)
        )
        print("✅ Gemini model initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing Gemini model: {e}")
        return
    
    # Set up browser session
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            viewport_expansion=0,
            user_data_dir='~/.config/browseruse/profiles/default',
        )
    )
    print("✅ Browser session configured!")
    
    async def run_simple_test():
        """Run a simple test to verify browser-use is working"""
        print("\n🚀 Starting browser automation test...")
        
        agent = Agent(
            task='Go to google.com and search for "browser automation with AI"',
            llm=llm,
            max_actions_per_step=3,
            browser_session=browser_session,
        )
        
        try:
            result = await agent.run(max_steps=10)
            print("✅ Test completed successfully!")
            return result
        except Exception as e:
            print(f"❌ Error during test execution: {e}")
            return None
    
    # Run the test
    print("\n🧪 Running browser automation test...")
    try:
        result = asyncio.run(run_simple_test())
        if result:
            print("\n🎉 Browser-use with Gemini Flash is working correctly!")
        else:
            print("\n⚠️  Test completed but with some issues.")
    except Exception as e:
        print(f"❌ Failed to run test: {e}")

if __name__ == '__main__':
    print("🔧 Testing Browser-Use with Gemini Flash Model")
    print("=" * 50)
    main()
