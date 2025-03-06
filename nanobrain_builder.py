#!/usr/bin/env python3
"""
NanoBrain Builder CLI

Command-line interface for building NanoBrain workflows.
"""

import asyncio
from builder import NanoBrainBuilder

if __name__ == "__main__":
    asyncio.run(NanoBrainBuilder.main()) 