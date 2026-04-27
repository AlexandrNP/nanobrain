#!/usr/bin/env python3
"""
Test Dependencies Checker
 
Verifies that all required testing dependencies are available
for the Chatbot Viral Integration testing framework.
"""

import sys
import importlib


def check_dependencies():
    """Check and report on testing dependencies"""
    print("🔍 Checking Test Dependencies for Chatbot Viral Integration")
    print("=" * 60)
    
    # Core dependencies
    core_deps = [
        ("pytest", "Core testing framework"),
        ("pytest_asyncio", "Async testing support"), 
        ("cachetools", "Production cache manager"),
    ]
    
    # Optional enhanced dependencies
    optional_deps = [
        ("pytest_html", "HTML test reports"),
        ("pytest_cov", "Coverage reporting"),
    ]
    
    missing_core = []
    missing_optional = []
    available_core = []
    available_optional = []
    
    # Check core dependencies
    print("\n📦 Core Testing Dependencies:")
    for module, description in core_deps:
        try:
            importlib.import_module(module)
            available_core.append((module, description))
            print(f"  ✅ {module:<15} - {description}")
        except ImportError:
            missing_core.append((module, description))
            print(f"  ❌ {module:<15} - {description} (MISSING)")
    
    # Check optional dependencies  
    print("\n🔧 Optional Enhanced Dependencies:")
    for module, description in optional_deps:
        try:
            importlib.import_module(module)
            available_optional.append((module, description))
            print(f"  ✅ {module:<15} - {description}")
        except ImportError:
            missing_optional.append((module, description))
            print(f"  ⚠️  {module:<15} - {description} (optional)")
    
    # Summary and recommendations
    print(f"\n📊 Summary:")
    print(f"  Core dependencies:     {len(available_core)}/{len(core_deps)} available")
    print(f"  Optional dependencies: {len(available_optional)}/{len(optional_deps)} available")
    
    if missing_core:
        print(f"\n❌ Missing REQUIRED dependencies:")
        for module, description in missing_core:
            print(f"  - {module}")
        
        print(f"\n🔧 Install missing core dependencies:")
        print(f"  pip install " + " ".join(module for module, _ in missing_core))
        print(f"  OR run: ./scripts/install_test_dependencies.sh")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies (recommended):")
        for module, description in missing_optional:
            print(f"  - {module}: {description}")
        
        print(f"\n🔧 Install optional dependencies:")
        print(f"  pip install " + " ".join(module for module, _ in missing_optional))
    
    print(f"\n🎉 Core testing framework ready!")
    print(f"\n🧪 Run tests with:")
    print(f"  cd tests/chatbot_viral_integration && python3 test_runner.py")
    
    if "pytest" in [m for m, _ in available_core]:
        print(f"  cd tests/chatbot_viral_integration && pytest test_query_classification.py -v")
    
    return True


def main():
    """Main entry point"""
    success = check_dependencies()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 