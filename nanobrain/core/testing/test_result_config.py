"""
Test Result Configuration
========================

ConfigBase-compliant configuration for test result constants following
mandatory NanoBrain framework patterns.
"""

from pydantic import Field, ConfigDict
from nanobrain.core.config.config_base import ConfigBase


class TestResultConfig(ConfigBase):
    """
    ✅ FRAMEWORK COMPLIANCE: Test result configuration via ConfigBase
    
    Configuration for test result constants following mandatory
    ConfigBase inheritance and from_config pattern.
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "success_value": "SUCCESS",
                    "failure_value": "FAILURE", 
                    "pass_value": "PASS",
                    "fail_value": "FAIL",
                    "pending_value": "PENDING",
                    "skipped_value": "SKIPPED",
                    "error_value": "ERROR",
                    "description": "Standard NanoBrain test result constants"
                }
            ],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "config_type": "test_constants",
                "supports_from_config": True
            }
        }
    )
    
    # ✅ FRAMEWORK COMPLIANCE: Configurable test result values
    success_value: str = Field(
        default="SUCCESS",
        description="String value for successful test results"
    )
    
    failure_value: str = Field(
        default="FAILURE", 
        description="String value for failed test results"
    )
    
    pass_value: str = Field(
        default="PASS",
        description="String value for passing test results"
    )
    
    fail_value: str = Field(
        default="FAIL",
        description="String value for failing test results"
    )
    
    pending_value: str = Field(
        default="PENDING",
        description="String value for pending test results"
    )
    
    skipped_value: str = Field(
        default="SKIPPED",
        description="String value for skipped test results"
    )
    
    error_value: str = Field(
        default="ERROR",
        description="String value for error test results"
    )
    
    description: str = Field(
        default="NanoBrain test result constants",
        description="Description of this test constants configuration"
    ) 