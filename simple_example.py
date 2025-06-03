#!/usr/bin/env python3
"""
Simple Browser-Use Example with Gemini Flash Model

This script demonstrates how to use browser-use with Google's Gemini Flash model
for web automation tasks.

Usage:
    python simple_example.py

Make sure you have:
1. Set GOOGLE_API_KEY in your .env file
2. Installed browser-use and playwright
3. Installed chromium browser
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent

async def main():
    """Main function to run browser automation"""
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        return
    
    print("🚀 Starting Browser-Use with Gemini Flash...")
    
    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash-exp',  # Using Gemini 2.0 Flash as requested
        api_key=SecretStr(api_key)
    )
    
    # Define your task here
    task = input("Enter your task (or press Enter for default): ").strip()
    if not task:
        task = "Go to google.com and search for 'AI browser automation'"
    
    print(f"📋 Task: {task}")
    
    # Create and run agent
    agent = Agent(
        task=task,
        llm=llm,
        max_actions_per_step=3,  # Limit actions per step for safety
    )
    
    try:
        result = await agent.run(max_steps=15)  # Limit total steps
        print(f"\n✅ Task completed successfully!")
        print(f"📄 Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    print("🌐 Browser-Use with Gemini Flash Model")
    print("=" * 40)
    asyncio.run(main())
