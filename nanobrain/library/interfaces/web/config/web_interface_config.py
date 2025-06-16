"""
Web Interface Configuration

Configuration classes for the web interface component.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os


@dataclass
class ServerConfig:
    """Server configuration for the web interface."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    access_log: bool = True
    

@dataclass 
class APIConfig:
    """API configuration settings."""
    prefix: str = "/api/v1"
    title: str = "NanoBrain Chat API"
    description: str = "REST API for NanoBrain Chat System" 
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


@dataclass
class CORSConfig:
    """CORS configuration settings."""
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False


@dataclass
class ChatConfig:
    """Chat-specific configuration."""
    default_temperature: float = 0.7
    default_max_tokens: int = 2000
    default_use_rag: bool = False
    enable_streaming: bool = False
    conversation_timeout_seconds: int = 3600
    max_conversation_length: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration for web interface."""
    enable_request_logging: bool = True
    enable_response_logging: bool = True
    log_level: str = "INFO"
    log_requests_body: bool = False
    log_responses_body: bool = False


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    enable_rate_limiting: bool = False
    rate_limit_per_minute: int = 60
    enable_auth: bool = False
    auth_secret_key: Optional[str] = None
    cors_enabled: bool = True


@dataclass 
class WebInterfaceConfig:
    """
    Complete configuration for the web interface.
    
    This class manages all configuration aspects of the web interface,
    including server settings, API configuration, CORS, security, and
    integration with NanoBrain workflows.
    """
    
    # Basic identification
    name: str = "nanobrain_web_interface"
    version: str = "1.0.0"
    description: str = "NanoBrain Web Interface"
    
    # Component configurations
    server: ServerConfig = field(default_factory=ServerConfig)
    api: APIConfig = field(default_factory=APIConfig)
    cors: CORSConfig = field(default_factory=CORSConfig) 
    chat: ChatConfig = field(default_factory=ChatConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Workflow integration
    workflow_config_path: Optional[str] = None
    enable_workflow_metrics: bool = True
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'WebInterfaceConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            WebInterfaceConfig: Loaded configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WebInterfaceConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            WebInterfaceConfig: Configuration instance
        """
        # Extract main config
        main_config = config_dict.get('web_interface', {})
        
        # Build component configs
        server_config = ServerConfig(**main_config.get('server', {}))
        api_config = APIConfig(**main_config.get('api', {}))
        cors_config = CORSConfig(**main_config.get('cors', {}))
        chat_config = ChatConfig(**main_config.get('chat', {}))
        logging_config = LoggingConfig(**main_config.get('logging', {}))
        security_config = SecurityConfig(**main_config.get('security', {}))
        
        return cls(
            name=main_config.get('name', 'nanobrain_web_interface'),
            version=main_config.get('version', '1.0.0'),
            description=main_config.get('description', 'NanoBrain Web Interface'),
            server=server_config,
            api=api_config,
            cors=cors_config,
            chat=chat_config,
            logging=logging_config,
            security=security_config,
            workflow_config_path=main_config.get('workflow_config_path'),
            enable_workflow_metrics=main_config.get('enable_workflow_metrics', True),
            custom_settings=main_config.get('custom_settings', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            'web_interface': {
                'name': self.name,
                'version': self.version,
                'description': self.description,
                'server': {
                    'host': self.server.host,
                    'port': self.server.port,
                    'workers': self.server.workers,
                    'reload': self.server.reload,
                    'access_log': self.server.access_log
                },
                'api': {
                    'prefix': self.api.prefix,
                    'title': self.api.title,
                    'description': self.api.description,
                    'version': self.api.version,
                    'docs_url': self.api.docs_url,
                    'redoc_url': self.api.redoc_url
                },
                'cors': {
                    'allow_origins': self.cors.allow_origins,
                    'allow_methods': self.cors.allow_methods,
                    'allow_headers': self.cors.allow_headers,
                    'allow_credentials': self.cors.allow_credentials
                },
                'chat': {
                    'default_temperature': self.chat.default_temperature,
                    'default_max_tokens': self.chat.default_max_tokens,
                    'default_use_rag': self.chat.default_use_rag,
                    'enable_streaming': self.chat.enable_streaming,
                    'conversation_timeout_seconds': self.chat.conversation_timeout_seconds,
                    'max_conversation_length': self.chat.max_conversation_length
                },
                'logging': {
                    'enable_request_logging': self.logging.enable_request_logging,
                    'enable_response_logging': self.logging.enable_response_logging,
                    'log_level': self.logging.log_level,
                    'log_requests_body': self.logging.log_requests_body,
                    'log_responses_body': self.logging.log_responses_body
                },
                'security': {
                    'enable_rate_limiting': self.security.enable_rate_limiting,
                    'rate_limit_per_minute': self.security.rate_limit_per_minute,
                    'enable_auth': self.security.enable_auth,
                    'auth_secret_key': self.security.auth_secret_key,
                    'cors_enabled': self.security.cors_enabled
                },
                'workflow_config_path': self.workflow_config_path,
                'enable_workflow_metrics': self.enable_workflow_metrics,
                'custom_settings': self.custom_settings
            }
        }
    
    def save_to_yaml(self, config_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_dict = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2) 