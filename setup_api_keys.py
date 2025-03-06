#!/usr/bin/env python3
"""
NanoBrain API Key Setup

This script helps users set up their API keys for the NanoBrain framework.
It guides users through the process of obtaining API keys for various LLM providers
and saves them to the global configuration file.
"""

import os
import sys
import getpass
from pathlib import Path

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    # Import the GlobalConfig class
    from src.GlobalConfig import GlobalConfig
    
    # Initialize the global configuration
    global_config = GlobalConfig()
    global_config.load_config()
except ImportError as e:
    print(f"Error: Could not import GlobalConfig: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def print_header(text):
    """Print a header with the given text."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")

def print_provider_info(provider):
    """Print information about how to obtain an API key for the given provider."""
    if provider == "openai":
        print("OpenAI API Key")
        print("--------------")
        print("To obtain an OpenAI API key:")
        print("1. Go to https://platform.openai.com/account/api-keys")
        print("2. Sign in or create an account")
        print("3. Click on 'Create new secret key'")
        print("4. Copy the key (you won't be able to see it again)")
    
    elif provider == "anthropic":
        print("Anthropic API Key")
        print("-----------------")
        print("To obtain an Anthropic API key:")
        print("1. Go to https://console.anthropic.com/account/keys")
        print("2. Sign in or create an account")
        print("3. Create a new API key")
        print("4. Copy the key")
    
    elif provider == "google":
        print("Google API Key")
        print("--------------")
        print("To obtain a Google API key for Gemini models:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Sign in with your Google account")
        print("3. Create a new API key")
        print("4. Copy the key")
    
    elif provider == "mistral":
        print("Mistral API Key")
        print("---------------")
        print("To obtain a Mistral API key:")
        print("1. Go to https://console.mistral.ai/api-keys/")
        print("2. Sign in or create an account")
        print("3. Create a new API key")
        print("4. Copy the key")
    
    elif provider == "huggingface":
        print("Hugging Face API Token")
        print("----------------------")
        print("To obtain a Hugging Face API token:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Sign in or create an account")
        print("3. Create a new token")
        print("4. Copy the token")

def get_api_key(provider):
    """Get an API key for the given provider from the user."""
    print_provider_info(provider)
    print("\nEnter your API key (or press Enter to skip):")
    key = getpass.getpass(f"{provider} API key: ")
    return key.strip() if key.strip() else None

def main():
    """Main function to guide users through setting up API keys."""
    print_header("NanoBrain API Key Setup")
    
    print("This script will help you set up API keys for various LLM providers.")
    print("The keys will be saved to your global configuration file.")
    print("You can skip any provider by pressing Enter when prompted for the key.")
    print("\nNote: API keys are sensitive information. They will be stored in your")
    print("configuration file, but will not be shared with anyone.")
    
    # Check if any keys are already configured
    existing_keys = []
    for provider in ["openai", "anthropic", "google", "mistral", "huggingface"]:
        if global_config.get_api_key(provider):
            existing_keys.append(provider)
    
    if existing_keys:
        print("\nYou already have API keys configured for the following providers:")
        for provider in existing_keys:
            print(f"  - {provider}")
        
        print("\nDo you want to update these keys? (y/n)")
        update = input("> ").lower()
        if update != "y":
            print("\nSkipping existing keys.")
            existing_keys = set(existing_keys)
        else:
            existing_keys = set()
    else:
        existing_keys = set()
    
    # Get API keys for each provider
    providers = [
        ("openai", "OpenAI (GPT models)"),
        ("anthropic", "Anthropic (Claude models)"),
        ("google", "Google (Gemini models)"),
        ("mistral", "Mistral AI"),
        ("huggingface", "Hugging Face")
    ]
    
    for provider, description in providers:
        if provider in existing_keys:
            continue
        
        print_header(f"Setting up {description}")
        key = get_api_key(provider)
        if key:
            global_config.set_api_key(provider, key)
            print(f"\n{provider} API key saved.")
        else:
            print(f"\nSkipping {provider} API key.")
    
    # Save the configuration
    config_path = global_config.config_path
    if not config_path:
        config_path = os.path.join(script_dir, "config.yml")
    
    global_config.save_config(config_path)
    print_header("Setup Complete")
    print(f"API keys have been saved to: {config_path}")
    print("\nYou can now use the NanoBrain framework with these API keys.")
    print("To update your API keys in the future, run this script again or use:")
    print("  ./nanobrain config edit")

if __name__ == "__main__":
    main() 