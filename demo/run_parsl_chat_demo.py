#!/usr/bin/env python3
"""
Simple runner script for the NanoBrain Parsl Chat Workflow Demo

This script provides an easy way to run the Parsl chat workflow demo
with different configurations and options.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'src'))

from chat_workflow_parsl_demo import main as demo_main
from core.logging_system import set_debug_mode


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the NanoBrain Parsl Chat Workflow Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_parsl_chat_demo.py                    # Run with default settings
  python run_parsl_chat_demo.py --debug            # Run with debug logging
  python run_parsl_chat_demo.py --workers 4        # Run with 4 parallel workers
  python run_parsl_chat_demo.py --no-parsl         # Run without Parsl (local only)
        """
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='Number of parallel workers/agents (default: 3)'
    )
    
    parser.add_argument(
        '--no-parsl',
        action='store_true',
        help='Force use of local executor instead of Parsl'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (overrides environment variable)'
    )
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment based on arguments."""
    
    # Set debug mode
    if args.debug:
        set_debug_mode(True)
        print("🐛 Debug mode enabled")
    
    # Set API key if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
        print("🔑 API key set from command line")
    
    # Set worker count
    if args.workers:
        os.environ['NANOBRAIN_PARSL_WORKERS'] = str(args.workers)
        print(f"👥 Worker count set to {args.workers}")
    
    # Disable Parsl if requested
    if args.no_parsl:
        os.environ['NANOBRAIN_DISABLE_PARSL'] = 'true'
        print("🚫 Parsl disabled - using local executor")
    
    # Set config file if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            os.environ['NANOBRAIN_CONFIG_FILE'] = str(config_path.absolute())
            print(f"⚙️  Using config file: {config_path}")
        else:
            print(f"❌ Config file not found: {config_path}")
            sys.exit(1)


def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    # Check for Parsl
    try:
        import parsl
        print(f"✅ Parsl {parsl.__version__} available")
    except ImportError:
        missing_deps.append("parsl")
        print("⚠️  Parsl not available - will use local executor")
    
    # Check for OpenAI
    try:
        import openai
        print(f"✅ OpenAI library available")
    except ImportError:
        missing_deps.append("openai")
        print("⚠️  OpenAI library not available - will use mock responses")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found")
    else:
        print("⚠️  No OpenAI API key found - will use mock responses")
    
    return missing_deps


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("🚀 NanoBrain Parsl Chat Workflow Demo")
    print("=" * 60)
    print("Features:")
    print("  • Parallel processing with Parsl executor")
    print("  • Multiple conversational agents")
    print("  • Distributed execution capabilities")
    print("  • Real-time performance monitoring")
    print("=" * 60)
    print()


def print_usage_tips():
    """Print usage tips."""
    print("\n💡 Usage Tips:")
    print("  • Type your messages normally to chat with the agents")
    print("  • Use /help to see available commands")
    print("  • Use /quit to exit the demo")
    print("  • Try /batch 5 to test parallel processing with 5 messages")
    print("  • Use Ctrl+C to interrupt if needed")
    print()


async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print banner
        print_banner()
        
        # Check dependencies
        print("🔍 Checking dependencies...")
        missing_deps = check_dependencies()
        
        if missing_deps:
            print(f"\n📦 Missing dependencies: {', '.join(missing_deps)}")
            print("   Install with: pip install " + " ".join(missing_deps))
            print("   Demo will continue with available features.\n")
        
        # Setup environment
        print("⚙️  Setting up environment...")
        setup_environment(args)
        
        # Print usage tips
        print_usage_tips()
        
        # Run the demo
        print("🎯 Starting demo...")
        await demo_main()
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 