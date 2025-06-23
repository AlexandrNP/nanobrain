"""
Enhanced Email Manager for Viral Protein Analysis Workflow
Phase 3 Implementation - Email Manager with Optional Usage

Provides configurable email management with service-specific usage patterns,
rate limiting, and environment-aware configuration.
"""

import os
import time
import yaml
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from nanobrain.core.logging_system import get_logger

class EmailManager:
    """
    Enhanced email manager with conditional usage and caching.
    
    Features:
    - Service-specific email configuration
    - Environment-aware timeout settings
    - Rate limiting integration
    - Optional authentication support
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "production"):
        """
        Initialize EmailManager with configuration.
        
        Args:
            config_path: Path to email configuration file
            environment: Environment type (production, testing, development)
        """
        self.logger = get_logger("email_manager")
        self.environment = environment
        
        # Load configuration
        if config_path is None:
            config_path = "nanobrain/library/workflows/viral_protein_analysis/config/email_config.yml"
        
        self.config = self._load_config(config_path)
        self.email = self.config.get("email_config", {}).get("default_email", "onarykov@anl.gov")
        
        # Initialize rate limiting tracking
        self._rate_limit_tracker = {}
        
        self.logger.info(f"EmailManager initialized for {environment} environment")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load email configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                self.logger.warning(f"Config file not found: {config_path}, using defaults")
                return self._get_default_config()
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            self.logger.debug(f"Loaded email configuration from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load email config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "email_config": {
                "default_email": "onarykov@anl.gov",
                "service_usage": {
                    "bvbrc_api": {"required": False, "use_email": False},
                    "pubmed_api": {"required": True, "use_email": True},
                    "ncbi_entrez": {"required": True, "use_email": True}
                },
                "environments": {
                    "testing": {"timeout_seconds": 10, "mock_api_calls": True},
                    "production": {"timeout_hours": 48, "mock_api_calls": False},
                    "development": {"timeout_seconds": 30, "mock_api_calls": False}
                }
            }
        }
    
    def get_email_for_service(self, service: str) -> Optional[str]:
        """
        Return email only if required and configured for service.
        
        Args:
            service: Service name (e.g., 'pubmed_api', 'bvbrc_api')
            
        Returns:
            Email address if required for service, None otherwise
        """
        service_config = self.config.get("email_config", {}).get("service_usage", {}).get(service, {})
        
        if service_config.get("use_email", False):
            email = service_config.get("email", self.email)
            self.logger.debug(f"Providing email for {service}: {email}")
            return email
        
        self.logger.debug(f"No email required for {service}")
        return None
    
    def should_authenticate(self, service: str) -> bool:
        """
        Check if service should use authentication.
        
        Args:
            service: Service name
            
        Returns:
            True if authentication should be used
        """
        service_config = self.config.get("email_config", {}).get("service_usage", {}).get(service, {})
        return service_config.get("fallback_with_auth", False)
    
    def get_timeout_config(self) -> Dict[str, Any]:
        """
        Get timeout configuration based on environment.
        
        Returns:
            Timeout configuration dictionary
        """
        env_config = self.config.get("email_config", {}).get("environments", {}).get(self.environment, {})
        
        # Default timeout values
        default_config = {
            "timeout_seconds": 30,
            "timeout_hours": 1,
            "use_cached_responses": True,
            "mock_api_calls": False
        }
        
        # Merge with environment-specific config
        timeout_config = {**default_config, **env_config}
        
        self.logger.debug(f"Timeout config for {self.environment}: {timeout_config}")
        return timeout_config
    
    def check_rate_limit(self, service: str) -> bool:
        """
        Check if service is within rate limits.
        
        Args:
            service: Service name
            
        Returns:
            True if within rate limits, False otherwise
        """
        rate_config = self.config.get("email_config", {}).get("rate_limiting", {}).get(service, {})
        
        if not rate_config:
            return True  # No rate limiting configured
        
        current_time = time.time()
        service_tracker = self._rate_limit_tracker.get(service, {
            "requests": [],
            "last_request": 0
        })
        
        # Clean old requests based on time window
        if "max_requests_per_hour" in rate_config:
            cutoff_time = current_time - 3600  # 1 hour
            service_tracker["requests"] = [
                req_time for req_time in service_tracker["requests"] 
                if req_time > cutoff_time
            ]
            
            max_requests = rate_config["max_requests_per_hour"]
            if len(service_tracker["requests"]) >= max_requests:
                self.logger.warning(f"Rate limit exceeded for {service}: {len(service_tracker['requests'])}/{max_requests} per hour")
                return False
        
        # Check per-second limits
        if "max_requests_per_second" in rate_config:
            min_interval = 1.0 / rate_config["max_requests_per_second"]
            time_since_last = current_time - service_tracker["last_request"]
            
            if time_since_last < min_interval:
                self.logger.warning(f"Rate limit exceeded for {service}: too frequent requests")
                return False
        
        return True
    
    def record_request(self, service: str) -> None:
        """
        Record a request for rate limiting tracking.
        
        Args:
            service: Service name
        """
        current_time = time.time()
        
        if service not in self._rate_limit_tracker:
            self._rate_limit_tracker[service] = {
                "requests": [],
                "last_request": 0
            }
        
        self._rate_limit_tracker[service]["requests"].append(current_time)
        self._rate_limit_tracker[service]["last_request"] = current_time
        
        self.logger.debug(f"Recorded request for {service} at {current_time}")
    
    def get_service_description(self, service: str) -> str:
        """
        Get human-readable description of service.
        
        Args:
            service: Service name
            
        Returns:
            Service description
        """
        service_config = self.config.get("email_config", {}).get("service_usage", {}).get(service, {})
        return service_config.get("description", f"Service: {service}")
    
    def get_eeev_config(self) -> Dict[str, Any]:
        """
        Get EEEV-specific configuration.
        
        Returns:
            EEEV configuration dictionary
        """
        eeev_config = self.config.get("email_config", {}).get("eeev_specific", {})
        
        # Default EEEV configuration
        default_eeev = {
            "literature_search_terms": ["Eastern equine encephalitis virus", "EEEV", "alphavirus"],
            "priority_proteins": ["capsid protein", "envelope protein E1", "envelope protein E2", "6K protein"],
            "expected_genome_size": {"min_kb": 10.5, "max_kb": 12.5, "typical_kb": 11.7}
        }
        
        return {**default_eeev, **eeev_config}
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the email configuration.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check required email configuration
        email_config = self.config.get("email_config", {})
        if not email_config.get("default_email"):
            issues.append("Missing default_email in configuration")
        
        # Check service configurations
        service_usage = email_config.get("service_usage", {})
        required_services = ["pubmed_api", "bvbrc_api", "ncbi_entrez"]
        
        for service in required_services:
            if service not in service_usage:
                issues.append(f"Missing configuration for required service: {service}")
        
        # Check environment configurations
        environments = email_config.get("environments", {})
        if self.environment not in environments:
            issues.append(f"Missing configuration for environment: {self.environment}")
        
        if issues:
            self.logger.warning(f"Configuration validation issues: {issues}")
        else:
            self.logger.info("Email configuration validation passed")
        
        return issues
    
    def get_cache_key(self, service: str, **kwargs) -> str:
        """
        Generate cache key for service requests.
        
        Args:
            service: Service name
            **kwargs: Additional parameters for cache key
            
        Returns:
            Cache key string
        """
        # Create deterministic cache key
        key_parts = [service]
        
        # Add sorted kwargs for consistency
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")
        
        key_string = "_".join(key_parts)
        
        # Hash for consistent length
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        self.logger.debug(f"Generated cache key for {service}: {cache_key}")
        return cache_key 