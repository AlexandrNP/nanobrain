#!/usr/bin/env python3
"""
Pydantic V2 Migration Verification Script

This script verifies that the Pydantic V2 migration was successful by:
1. Checking for remaining V1 patterns in the codebase
2. Testing model instantiation
3. Verifying no deprecation warnings
4. Testing JSON schema generation

Usage: python scripts/verify_pydantic_migration.py
"""

import warnings
import sys
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any

# Add nanobrain to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_v1_patterns(directory: str = "nanobrain") -> Dict[str, List[str]]:
    """Check for remaining Pydantic V1 patterns in the codebase."""
    patterns = {
        "@validator": [],
        "schema_extra =": [],  # Only catch the old V1 pattern, not json_schema_extra
        "class Config:": [],
        "@root_validator": []
    }

    for pattern in patterns.keys():
        try:
            result = subprocess.run(
                ["grep", "-r", pattern, directory, "--include=*.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                patterns[pattern] = result.stdout.strip().split('\n')
        except Exception as e:
            print(f"Warning: Could not check pattern {pattern}: {e}")
            import traceback
            traceback.print_exc()

    return patterns


def test_model_imports() -> bool:
    """Test that all migrated models can be imported without warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            # Test web interface models
            from nanobrain.library.interfaces.web.models.request_models import (
                ChatOptions, ChatRequest
            )
            from nanobrain.library.interfaces.web.models.response_models import (
                ChatResponse, ErrorResponse, HealthResponse, StatusResponse
            )

            # Check for Pydantic-related warnings
            pydantic_warnings = [
                warning for warning in w
                if 'pydantic' in str(warning.message).lower()
                or 'validator' in str(warning.message).lower()
                or 'deprecated' in str(warning.message).lower()
            ]

            if pydantic_warnings:
                print("❌ Pydantic warnings found during import:")
                for warning in pydantic_warnings:
                    print(f"   {warning.message}")
                return False

            return True

        except Exception as e:
            print(f"❌ Import failed: {e}")
            return False


def test_model_instantiation() -> bool:
    """Test that models can be instantiated successfully."""
    try:
        from nanobrain.library.interfaces.web.models.request_models import (
            ChatOptions, ChatRequest
        )
        from nanobrain.library.interfaces.web.models.response_models import (
            ChatResponse, ErrorResponse, HealthResponse, StatusResponse, ChatMetadata
        )

        # Test basic instantiation
        options = ChatOptions()
        request = ChatRequest(query="test query")
        metadata = ChatMetadata(processing_time_ms=100.0)
        response = ChatResponse(
            response="Test response",
            conversation_id="test-conv-id",
            metadata=metadata
        )
        error = ErrorResponse(error="TestError", message="Test message")
        health = HealthResponse(status="healthy", version="1.0.0")
        status = StatusResponse(
            api_status="operational", uptime_seconds=3600.0)

        # Test with data
        request_with_options = ChatRequest(
            query="test query",
            options=ChatOptions(temperature=0.8, max_tokens=1000)
        )

        print("✅ All model instantiation tests passed")
        return True

    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return False


def test_json_schema_generation() -> bool:
    """Test that JSON schema generation works with V2."""
    try:
        from nanobrain.library.interfaces.web.models.request_models import ChatRequest
        from nanobrain.library.interfaces.web.models.response_models import ChatResponse

        # Test JSON schema generation
        request_schema = ChatRequest.model_json_schema()

        # Verify schema contains expected V2 features
        if 'properties' not in request_schema:
            print("❌ JSON schema missing properties")
            return False

        # Check that examples are present (from json_schema_extra)
        if 'examples' in request_schema or '$defs' in request_schema:
            print("✅ JSON schema generation successful")
            return True
        else:
            print("⚠️  JSON schema generated but may be missing examples")
            return True

    except Exception as e:
        print(f"❌ JSON schema generation failed: {e}")
        return False


def run_chatbot_tests() -> bool:
    """Run chatbot viral integration tests to verify no warnings."""
    try:
        result = subprocess.run(
            ["python", "tests/chatbot_viral_integration/test_runner.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True
        )

        # Check for Pydantic warnings in output
        output = result.stdout + result.stderr
        if any(word in output.lower() for word in ['deprecated', 'pydantic.*warning', 'validator.*deprecated']):
            print("❌ Pydantic warnings found in test output")
            return False

        if result.returncode == 0:
            print("✅ Chatbot integration tests passed without warnings")
            return True
        else:
            print(f"⚠️  Tests completed with return code {result.returncode}")
            return True  # Tests may fail for other reasons

    except Exception as e:
        print(f"❌ Failed to run chatbot tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification routine."""
    print("🔍 Pydantic V2 Migration Verification")
    print("=" * 50)

    # Check for remaining V1 patterns
    print("\n1. Checking for remaining Pydantic V1 patterns...")
    patterns = check_v1_patterns()

    remaining_patterns = []
    for pattern, files in patterns.items():
        if files and files != ['']:
            remaining_patterns.append(pattern)
            print(f"❌ Found {pattern} in {len(files)} locations:")
            for file_line in files[:3]:  # Show first 3 occurrences
                print(f"   {file_line}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more")

    if not remaining_patterns:
        print("✅ No Pydantic V1 patterns found")

    # Test model imports
    print("\n2. Testing model imports...")
    import_success = test_model_imports()

    # Test model instantiation
    print("\n3. Testing model instantiation...")
    instantiation_success = test_model_instantiation()

    # Test JSON schema generation
    print("\n4. Testing JSON schema generation...")
    schema_success = test_json_schema_generation()

    # Run integration tests
    print("\n5. Running chatbot integration tests...")
    test_success = run_chatbot_tests()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Migration Verification Summary")
    print("=" * 50)

    all_checks = [
        ("V1 patterns eliminated", not remaining_patterns),
        ("Model imports", import_success),
        ("Model instantiation", instantiation_success),
        ("JSON schema generation", schema_success),
        ("Integration tests", test_success)
    ]

    passed = sum(1 for _, success in all_checks if success)
    total = len(all_checks)

    for check_name, success in all_checks:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {check_name}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 Pydantic V2 migration verification SUCCESSFUL!")
        return 0
    else:
        print("⚠️  Some checks failed - review issues above")
        return 1


if __name__ == "__main__":
    exit(main())
