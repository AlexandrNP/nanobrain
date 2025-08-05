"""
Test Constants Implementation
============================

Framework-compliant test result constants following NanoBrain patterns:
- ConfigBase inheritance for configuration loading
- from_config pattern for component creation
- Enum-based type safety for test results
"""

from enum import Enum
from typing import Dict, Any
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.component_base import FromConfigBase


class TestResult(Enum):
    """✅ FRAMEWORK COMPLIANCE: Test result enum with standardized values"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE" 
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class TestConstants(FromConfigBase):
    """
    ✅ FRAMEWORK COMPLIANCE: Test constants following from_config pattern
    
    Provides standardized test result constants loaded from configuration
    following mandatory ConfigBase patterns.
    """
    
    def __init__(self, config: 'TestResultConfig'):
        """Initialize test constants from configuration"""
        self.config = config
        
        # ✅ FRAMEWORK COMPLIANCE: Constants from configuration
        self.SUCCESS = config.success_value
        self.FAILURE = config.failure_value
        self.PASS = config.pass_value
        self.FAIL = config.fail_value
        self.PENDING = config.pending_value
        self.SKIPPED = config.skipped_value
        self.ERROR = config.error_value
    
    @classmethod
    def _get_config_class(cls):
        """✅ MANDATORY: Return config class for from_config pattern"""
        from .test_result_config import TestResultConfig
        return TestResultConfig
    
    def _init_from_config(self, config: 'TestResultConfig', component_config: Dict[str, Any], dependencies: Dict[str, Any]):
        """✅ FRAMEWORK COMPLIANCE: Initialize from configuration"""
        # Constants already set in __init__
        pass
    
    def get_result_enum(self, value: str) -> TestResult:
        """Get TestResult enum from string value"""
        value_map = {
            self.SUCCESS: TestResult.SUCCESS,
            self.FAILURE: TestResult.FAILURE,
            self.PASS: TestResult.PASS,
            self.FAIL: TestResult.FAIL,
            self.PENDING: TestResult.PENDING,
            self.SKIPPED: TestResult.SKIPPED,
            self.ERROR: TestResult.ERROR
        }
        return value_map.get(value, TestResult.ERROR)
    
    def is_success(self, value: str) -> bool:
        """Check if result value indicates success"""
        return value in [self.SUCCESS, self.PASS]
    
    def is_failure(self, value: str) -> bool:
        """Check if result value indicates failure"""
        return value in [self.FAILURE, self.FAIL, self.ERROR] 