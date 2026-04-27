#!/usr/bin/env python3
"""
Tool Migration Validation Script
Tests the from_config pattern compliance for all migrated tool classes
"""

import asyncio
import sys
from pathlib import Path

# Add the nanobrain package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.tool import (
    ToolBase, ToolConfig, ToolType, create_tool,
    FunctionTool, AgentTool, StepTool, LangChainTool
)
from nanobrain.core.logging_system import get_logger

logger = get_logger("tool_migration_validator")


async def test_direct_instantiation_prevention():
    """Test that direct instantiation is prevented"""
    logger.info("Testing direct instantiation prevention...")
    
    config = ToolConfig(
        name="test_tool",
        tool_type=ToolType.FUNCTION,
        description="Test tool"
    )
    
    # Test ToolBase (abstract class, should fail)
    try:
        tool = ToolBase(config)
        logger.error("❌ ToolBase allowed direct instantiation - SHOULD BE PREVENTED")
        return False
    except TypeError as e:
        logger.info("✅ ToolBase correctly prevents direct instantiation")
    
    # Test concrete tool classes
    tool_classes = [FunctionTool, AgentTool, StepTool, LangChainTool]
    
    for tool_class in tool_classes:
        try:
            tool = tool_class(config)
            logger.error(f"❌ {tool_class.__name__} allowed direct instantiation - SHOULD BE PREVENTED")
            return False
        except (TypeError, RuntimeError) as e:
            # Framework raises RuntimeError for direct instantiation prevention
            logger.info(f"✅ {tool_class.__name__} correctly prevents direct instantiation: {type(e).__name__}")
    
    return True


async def test_from_config_creation():
    """Test that tools can be created via from_config"""
    logger.info("Testing from_config tool creation...")
    
    # Test FunctionTool
    def test_function(x: int) -> int:
        return x * 2
    
    config = ToolConfig(
        name="test_function_tool",
        tool_type=ToolType.FUNCTION,
        description="Test function tool"
    )
    
    try:
        tool = FunctionTool.from_config(config, func=test_function)
        logger.info(f"✅ FunctionTool created successfully: {tool.name}")
        
        # Test execution
        result = await tool.execute(x=5)
        if result == 10:
            logger.info("✅ FunctionTool execution works correctly")
        else:
            logger.error(f"❌ FunctionTool execution failed - expected 10, got {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ FunctionTool creation failed: {e}")
        return False
    
    return True


async def test_create_tool_factory():
    """Test the updated create_tool factory function"""
    logger.info("Testing create_tool factory...")
    
    def test_function(x: int) -> int:
        return x * 3
    
    # Test with dict config
    config_dict = {
        'name': 'factory_test_tool',
        'tool_type': 'function',
        'description': 'Factory test tool'
    }
    
    try:
        tool = create_tool(config_dict, func=test_function)
        logger.info(f"✅ create_tool with dict config successful: {tool.name}")
        
        # Test execution
        result = await tool.execute(x=4)
        if result == 12:
            logger.info("✅ Factory-created tool execution works correctly")
        else:
            logger.error(f"❌ Factory-created tool execution failed - expected 12, got {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ create_tool factory failed: {e}")
        return False
    
    # Test with ToolConfig object
    config_obj = ToolConfig(
        name="factory_test_tool_2",
        tool_type=ToolType.FUNCTION,
        description="Factory test tool 2"
    )
    
    try:
        tool = create_tool(config_obj, func=test_function)
        logger.info(f"✅ create_tool with ToolConfig object successful: {tool.name}")
    except Exception as e:
        logger.error(f"❌ create_tool with ToolConfig object failed: {e}")
        return False
    
    return True


async def test_function_tool_decorator():
    """Test the updated function_tool decorator"""
    logger.info("Testing function_tool decorator...")
    
    try:
        from nanobrain.core.tool import function_tool
        
        @function_tool(
            name="decorator_test", 
            description="Test decorator tool",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "integer"}
                },
                "required": ["value"]
            }
        )
        def decorator_test_func(value: int) -> int:
            return value * 4
        
        logger.info(f"✅ Function decorator created tool: {decorator_test_func.name}")
        
        # Test execution
        result = await decorator_test_func.execute(value=3)
        if result == 12:
            logger.info("✅ Decorator-created tool execution works correctly")
        else:
            logger.error(f"❌ Decorator-created tool execution failed - expected 12, got {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ function_tool decorator failed: {e}")
        return False
    
    return True


async def test_error_handling():
    """Test error handling for missing dependencies"""
    logger.info("Testing error handling...")
    
    config = ToolConfig(
        name="error_test_tool",
        tool_type=ToolType.FUNCTION,
        description="Error test tool"
    )
    
    # Test missing function parameter
    try:
        tool = FunctionTool.from_config(config)  # Missing func parameter
        logger.error("❌ FunctionTool should have failed without func parameter")
        return False
    except Exception as e:
        logger.info(f"✅ FunctionTool correctly handles missing func parameter: {type(e).__name__}")
    
    return True


async def main():
    """Run all validation tests"""
    logger.info("🔄 Starting Tool Migration Validation Tests")
    logger.info("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Direct Instantiation Prevention", test_direct_instantiation_prevention),
        ("From Config Creation", test_from_config_creation),
        ("Create Tool Factory", test_create_tool_factory),
        ("Function Tool Decorator", test_function_tool_decorator),
        ("Error Handling", test_error_handling),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running Test: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = await test_func()
            test_results.append((test_name, result))
            
            if result:
                logger.info(f"✅ Test {test_name}: PASSED")
            else:
                logger.error(f"❌ Test {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ Test {test_name}: EXCEPTION - {e}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<35} {status}")
    
    logger.info("-" * 60)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Tool migration successful!")
        return True
    else:
        logger.error("⚠️ Some tests failed - Tool migration needs attention")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 