#!/usr/bin/env python3
"""
Academy-Nanobrain Communication Demo Runner

This script demonstrates REAL communication between Nanobrain workflows 
and Academy agents with NO MOCKS, STUBS, OR SIMULATED RESPONSES.
"""

import asyncio
import sys
import os

# Add the nanobrain root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from test_real_academy_communication import test_real_academy_communication

async def main():
    """Run the Academy-Nanobrain communication demo"""
    print("🔥 ACADEMY-NANOBRAIN COMMUNICATION DEMO")
    print("=" * 50)
    print("This demo proves REAL inter-framework communication")
    print("with NO mocks, stubs, or simulated responses!")
    print("=" * 50)
    print()
    
    try:
        await test_real_academy_communication()
        print()
        print("🎯 DEMO COMPLETED SUCCESSFULLY!")
        print("✅ Real Academy-Nanobrain communication verified!")
        
    except Exception as e:
        print(f"❌ DEMO FAILED: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
