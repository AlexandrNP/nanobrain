"""
NanoBrain Test Framework Constants
================================

Framework-compliant test result constants and utilities following
the mandatory from_config pattern and ConfigBase inheritance.
"""

# ✅ IMMEDIATE FIX: Define SUCCESS constant directly to resolve NameError
SUCCESS = "SUCCESS"
FAILURE = "FAILURE"
PASS = "PASS"
FAIL = "FAIL"
PENDING = "PENDING"
SKIPPED = "SKIPPED"
ERROR = "ERROR"

# ✅ FRAMEWORK COMPLIANCE: Advanced test framework components
try:
    from .test_constants import TestConstants, TestResult
    from .test_result_config import TestResultConfig
    
    __all__ = [
        'TestConstants',
        'TestResult', 
        'TestResultConfig',
        'SUCCESS',
        'FAILURE',
        'PASS',
        'FAIL',
        'PENDING',
        'SKIPPED',
        'ERROR'
    ]
    
    # ✅ FRAMEWORK COMPLIANCE: Load enhanced constants from configuration
    try:
        import os
        config_path = os.path.join(os.path.dirname(__file__), "config", "default_test_constants.yml")
        if os.path.exists(config_path):
            _config = TestResultConfig.from_config(config_path)
            _constants = TestConstants.from_config(_config)
            
            # Override with configured values
            SUCCESS = _constants.SUCCESS
            FAILURE = _constants.FAILURE  
            PASS = _constants.PASS
            FAIL = _constants.FAIL
            PENDING = _constants.PENDING
            SKIPPED = _constants.SKIPPED
            ERROR = _constants.ERROR
            
    except Exception as e:
        # Keep default constants if config loading fails
        pass
        
except ImportError:
    # ✅ GRACEFUL DEGRADATION: Basic constants if advanced imports fail
    __all__ = [
        'SUCCESS',
        'FAILURE', 
        'PASS',
        'FAIL',
        'PENDING',
        'SKIPPED',
        'ERROR'
    ] 