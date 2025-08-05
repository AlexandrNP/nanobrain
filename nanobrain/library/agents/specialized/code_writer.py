"""
Code Writer Agent for NanoBrain Framework

Specialized agent for generating and writing code files.
"""

import logging
from typing import Any, Dict, Optional, List

# Updated imports for nanobrain package structure  
from nanobrain.core.agent import AgentConfig
from nanobrain.core.logging_system import get_logger
from .base import SpecializedAgentBase, SimpleSpecializedAgent, ConversationalSpecializedAgent

logger = logging.getLogger(__name__)


class CodeWriterAgentMixin(SpecializedAgentBase):
    """
    Code Writer Agent Mixin - Intelligent Code Generation and Development Automation with Multi-Language Support
    ==========================================================================================================
    
    The CodeWriterAgentMixin provides specialized capabilities for automated code generation, programming assistance,
    and development workflow automation. This mixin integrates advanced natural language processing with programming
    language understanding to generate, analyze, and manipulate source code across multiple programming languages
    and development paradigms.
    
    **Core Architecture:**
        The code writer agent provides enterprise-grade code generation capabilities:
        
        * **Multi-Language Support**: Comprehensive support for Python, JavaScript, Java, C++, HTML, CSS, and more
        * **Intelligent Code Generation**: Natural language to code translation with context awareness
        * **Code Analysis**: Static analysis, pattern recognition, and code quality assessment
        * **Development Workflows**: Integration with development tools and automation pipelines
        * **Template Systems**: Reusable code templates and scaffolding generation
        * **Framework Integration**: Full integration with NanoBrain's specialized agent architecture
    
    **Code Generation Capabilities:**
        
        **Natural Language Processing:**
        * Advanced parsing of programming requirements and specifications
        * Context-aware code generation based on project structure and patterns
        * Intent recognition for different types of code generation tasks
        * Technical documentation and comment generation
        
        **Programming Language Support:**
        * **Python**: Functions, classes, modules, packages, decorators, async/await patterns
        * **JavaScript**: ES6+ syntax, React components, Node.js modules, TypeScript support
        * **Java**: Classes, interfaces, Spring Boot applications, Maven projects
        * **C++**: Classes, templates, STL usage, modern C++ patterns
        * **HTML/CSS**: Responsive layouts, component structures, styling frameworks
        * **SQL**: Database schemas, queries, stored procedures, migrations
        
        **Code Quality Features:**
        * Automatic code formatting and style compliance
        * Best practices integration and pattern enforcement
        * Security vulnerability detection and mitigation
        * Performance optimization suggestions and implementations
        
        **Development Automation:**
        * Project scaffolding and boilerplate generation
        * Configuration file creation and management
        * Build system integration and automation scripts
        * Testing framework setup and test case generation
    
    **Programming Paradigm Support:**
        
        **Object-Oriented Programming:**
        * Class hierarchy design and implementation
        * Design pattern application (Singleton, Factory, Observer, etc.)
        * Inheritance, polymorphism, and encapsulation best practices
        * Interface and abstract class design
        
        **Functional Programming:**
        * Higher-order function implementation
        * Lambda expressions and functional composition
        * Immutable data structure design
        * Functional reactive programming patterns
        
        **Asynchronous Programming:**
        * Async/await pattern implementation
        * Promise and Future-based programming
        * Event-driven architecture design
        * Concurrent and parallel processing patterns
        
        **Microservices Architecture:**
        * Service decomposition and API design
        * Container-based deployment configurations
        * Service mesh integration and communication patterns
        * Distributed system resilience patterns
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse development workflows:
        
        ```yaml
        # Code Writer Agent Configuration
        agent_name: "code_writer_agent"
        agent_type: "specialized"
        
        # Agent card for framework integration
        agent_card:
          name: "code_writer_agent"
          description: "Intelligent code generation and development automation"
          version: "1.0.0"
          category: "development"
          capabilities:
            - "code_generation"
            - "multi_language_support"
            - "development_automation"
        
        # LLM Configuration
        llm_config:
          model: "gpt-4"
          temperature: 0.1        # Low temperature for consistent code generation
          max_tokens: 4000        # Larger token limit for code generation
          
        # Code Generation Configuration
        supported_languages:
          - "python"
          - "javascript"
          - "typescript"
          - "java"
          - "cpp"
          - "html"
          - "css"
          - "sql"
          
        code_style_preferences:
          python:
            formatter: "black"
            linter: "pylint"
            max_line_length: 88
            
          javascript:
            formatter: "prettier"
            linter: "eslint"
            style_guide: "airbnb"
            
        # Template Configuration
        template_directories:
          - "templates/python/"
          - "templates/javascript/"
          - "templates/java/"
          
        # Development Workflow
        workflow_integration:
          version_control: "git"
          ci_cd_platform: "github_actions"
          package_managers:
            python: "pip"
            javascript: "npm"
            java: "maven"
            
        # Quality Assurance
        quality_checks:
          enable_linting: true
          enable_testing: true
          enable_security_scan: true
          code_coverage_threshold: 80
        ```
    
    **Usage Patterns:**
        
        **Basic Code Generation:**
        ```python
        from nanobrain.library.agents.specialized import CodeWriterAgent
        
        # Create code writer agent with configuration
        agent_config = AgentConfig.from_config('config/code_writer_config.yml')
        code_writer = CodeWriterAgent.from_config(agent_config)
        
        # Generate Python function
        function_request = "Create a Python function that calculates the factorial of a number using recursion"
        
        code_result = await code_writer.generate_code(function_request)
        
        # Access generated code
        print("Generated Code:")
        print(code_result.data['code'])
        print(f"Language: {code_result.data['language']}")
        print(f"Quality Score: {code_result.data['quality_score']}")
        ```
        
        **Multi-Language Project Generation:**
        ```python
        # Configure for full-stack development
        fullstack_config = {
            'supported_languages': ['python', 'javascript', 'html', 'css'],
            'project_type': 'web_application',
            'framework_preferences': {
                'backend': 'fastapi',
                'frontend': 'react',
                'database': 'postgresql'
            }
        }
        
        agent_config = AgentConfig.from_config(fullstack_config)
        code_writer = CodeWriterAgent.from_config(agent_config)
        
        # Generate full-stack application structure
        project_spec = {
            'application_type': 'web_api',
            'features': ['user_authentication', 'data_crud', 'real_time_updates'],
            'deployment_target': 'docker_containers'
        }
        
        project_result = await code_writer.generate_project(project_spec)
        
        # Access generated project components
        backend_code = project_result.data['backend']
        frontend_code = project_result.data['frontend']
        database_schema = project_result.data['database']
        docker_config = project_result.data['deployment']
        
        for component, code in project_result.data.items():
            print(f"\\n{component.upper()} Component:")
            print(f"Files: {len(code['files'])}")
            print(f"Main Entry: {code['entry_point']}")
        ```
        
        **Code Analysis and Refactoring:**
        ```python
        # Configure for code analysis and improvement
        analysis_config = {
            'analysis_depth': 'comprehensive',
            'refactoring_suggestions': True,
            'performance_optimization': True,
            'security_analysis': True
        }
        
        agent_config = AgentConfig.from_config(analysis_config)
        code_writer = CodeWriterAgent.from_config(agent_config)
        
        # Analyze existing code
        existing_code = '''
        def calculate_stats(data):
            total = 0
            for item in data:
                total += item
            average = total / len(data)
            return total, average
        '''
        
        analysis_result = await code_writer.analyze_code(
            existing_code,
            language='python',
            analysis_type='comprehensive'
        )
        
        # Access analysis results
        suggestions = analysis_result.data['suggestions']
        optimized_code = analysis_result.data['optimized_code']
        security_issues = analysis_result.data['security_issues']
        performance_metrics = analysis_result.data['performance_metrics']
        
        print("Code Analysis Results:")
        print(f"Suggestions: {len(suggestions)}")
        print(f"Security Issues: {len(security_issues)}")
        print(f"Performance Score: {performance_metrics['score']}")
        
        print("\\nOptimized Code:")
        print(optimized_code)
        ```
        
        **Template-Based Development:**
        ```python
        # Configure for template-based code generation
        template_config = {
            'template_system': 'jinja2',
            'template_directories': ['templates/microservices/', 'templates/apis/'],
            'variable_substitution': True,
            'conditional_generation': True
        }
        
        agent_config = AgentConfig.from_config(template_config)
        code_writer = CodeWriterAgent.from_config(agent_config)
        
        # Generate microservice from template
        service_spec = {
            'service_name': 'user_management',
            'database_type': 'postgresql',
            'authentication': 'jwt',
            'api_endpoints': [
                {'path': '/users', 'methods': ['GET', 'POST']},
                {'path': '/users/{id}', 'methods': ['GET', 'PUT', 'DELETE']},
                {'path': '/auth/login', 'methods': ['POST']},
                {'path': '/auth/logout', 'methods': ['POST']}
            ],
            'deployment': 'kubernetes'
        }
        
        # Generate service using templates
        service_result = await code_writer.generate_from_template(
            template_name='microservice_fastapi',
            variables=service_spec
        )
        
        # Access generated service components
        service_files = service_result.data['files']
        deployment_configs = service_result.data['deployment']
        api_documentation = service_result.data['documentation']
        
        print(f"Generated {len(service_files)} files for microservice")
        print(f"API endpoints: {len(service_spec['api_endpoints'])}")
        print(f"Deployment target: {service_spec['deployment']}")
        ```
        
        **Development Workflow Integration:**
        ```python
        # Configure for CI/CD integration
        workflow_config = {
            'version_control': 'git',
            'ci_cd_platform': 'github_actions',
            'deployment_platforms': ['docker', 'kubernetes', 'aws'],
            'quality_gates': {
                'code_coverage': 85,
                'security_scan': True,
                'performance_benchmark': True
            }
        }
        
        agent_config = AgentConfig.from_config(workflow_config)
        code_writer = CodeWriterAgent.from_config(agent_config)
        
        # Generate complete development workflow
        workflow_spec = {
            'project_name': 'ai_analysis_service',
            'language': 'python',
            'framework': 'fastapi',
            'testing_framework': 'pytest',
            'deployment_strategy': 'blue_green'
        }
        
        workflow_result = await code_writer.setup_development_workflow(workflow_spec)
        
        # Access workflow components
        source_code = workflow_result.data['source_code']
        tests = workflow_result.data['tests']
        ci_config = workflow_result.data['ci_configuration']
        deployment_scripts = workflow_result.data['deployment']
        documentation = workflow_result.data['documentation']
        
        print("Development Workflow Generated:")
        print(f"Source files: {len(source_code['files'])}")
        print(f"Test files: {len(tests['files'])}")
        print(f"CI/CD pipeline: {ci_config['platform']}")
        print(f"Deployment method: {deployment_scripts['strategy']}")
        ```
    
    **Advanced Features:**
        
        **AI-Powered Code Intelligence:**
        * Context-aware code completion and suggestion
        * Intelligent refactoring with semantic understanding
        * Automated bug detection and fix generation
        * Code pattern recognition and standardization
        
        **Enterprise Integration:**
        * Integration with enterprise development tools and IDEs
        * Support for corporate coding standards and guidelines
        * Large-scale codebase analysis and management
        * Team collaboration and code review automation
        
        **Security and Compliance:**
        * Security vulnerability detection and mitigation
        * Compliance checking for industry standards (SOC2, GDPR, etc.)
        * Code obfuscation and intellectual property protection
        * Secure coding practice enforcement
        
        **Performance Optimization:**
        * Algorithmic complexity analysis and optimization
        * Memory usage optimization and leak detection
        * Parallel processing and concurrency optimization
        * Database query optimization and indexing suggestions
    
    **Development Applications:**
        
        **Rapid Prototyping:**
        * Quick MVP development and proof-of-concept creation
        * API prototype generation with mock data
        * User interface mockup and wireframe implementation
        * Database schema prototyping and validation
        
        **Enterprise Development:**
        * Large-scale application architecture design
        * Microservices decomposition and implementation
        * Legacy system modernization and migration
        * Scalable system design and implementation
        
        **Educational and Training:**
        * Code example generation for learning purposes
        * Programming exercise creation and solution generation
        * Best practice demonstration and explanation
        * Interactive coding tutorial development
        
        **Research and Innovation:**
        * Algorithm implementation and experimental code generation
        * Research prototype development and validation
        * Open source contribution and project scaffolding
        * Technical proof-of-concept development
    
    Attributes:
        code_keywords (list): Keywords that trigger code generation functionality
        language_patterns (dict): Programming language detection patterns and syntax
        supported_languages (list): List of supported programming languages
        template_system (object): Code template management system
        quality_analyzer (object): Code quality analysis and improvement system
        workflow_integrator (object): Development workflow automation system
    
    Note:
        This agent requires LLM access for natural language to code translation.
        Generated code should be reviewed and tested before production use.
        The agent supports continuous learning and improvement through feedback
        mechanisms. Template systems require proper configuration for optimal results.
    
    Warning:
        Generated code should always be reviewed for security vulnerabilities and
        business logic correctness. Large code generation tasks may require
        significant computational resources. Be mindful of intellectual property
        and licensing requirements when generating code based on existing examples.
    
    See Also:
        * :class:`SpecializedAgentBase`: Base specialized agent interface
        * :class:`SimpleSpecializedAgent`: Simple specialized agent implementation
        * :class:`ConversationalSpecializedAgent`: Conversational agent capabilities
        * :class:`AgentConfig`: Agent configuration schema
        * :mod:`nanobrain.library.agents.specialized`: Specialized agent implementations
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Code generation keywords
        self.code_keywords = [
            'code', 'function', 'class', 'method', 'script', 'program',
            'generate', 'create', 'write', 'implement', 'develop'
        ]
        
        # Programming language patterns
        self.language_patterns = {
            'python': ['python', 'py', 'def ', 'class ', 'import '],
            'javascript': ['javascript', 'js', 'function ', 'const ', 'let '],
            'java': ['java', 'public class', 'private ', 'public '],
            'cpp': ['c++', 'cpp', '#include', 'int main', 'std::'],
            'html': ['html', '<html>', '<div>', '<body>'],
            'css': ['css', 'style', '{', '}', 'color:', 'font-']
        }
    
    async def _initialize_specialized_features(self) -> None:
        """Initialize code writer specific features."""
        await super()._initialize_specialized_features()
        self.specialized_logger.info("CodeWriter specialized features initialized")
    
    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by code generation logic.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if should be handled by code generation logic
        """
        # Check for explicit code generation parameters
        has_code_params = any(param in kwargs for param in ['function_name', 'class_name', 'language'])
        
        # Check for code generation keywords in input
        has_keywords = any(keyword in input_text.lower() for keyword in self.code_keywords)
        
        # Check for programming language indicators
        has_language_indicators = any(
            any(pattern in input_text.lower() for pattern in patterns)
            for patterns in self.language_patterns.values()
        )
        
        return has_code_params or (has_keywords and has_language_indicators)
    
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process code generation requests directly when possible.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Generated code if handled, None if should fall back to LLM
        """
        try:
            # Check for specific code generation requests
            if 'function_name' in kwargs:
                return await self._generate_function_from_params(**kwargs)
            
            if 'class_name' in kwargs:
                return await self._generate_class_from_params(**kwargs)
            
            # Try to parse simple code generation requests
            result = await self._try_parse_code_request(input_text)
            if result:
                return result
            
            return None
            
        except Exception as e:
            self.specialized_logger.error(f"Error in specialized code processing: {e}")
            return None
    
    async def _generate_function_from_params(self, **kwargs) -> str:
        """Generate a function from explicit parameters."""
        function_name = kwargs.get('function_name', 'generated_function')
        description = kwargs.get('description', 'Generated function')
        parameters = kwargs.get('parameters', {})
        return_type = kwargs.get('return_type', 'Any')
        
        # Generate parameter string
        params_str = ""
        if parameters:
            param_list = [f"{name}: {type_hint}" for name, type_hint in parameters.items()]
            params_str = ", ".join(param_list)
        
        # Generate function code
        code = f'''def {function_name}({params_str}) -> {return_type}:
    """
    {description}
    
    Args:
        {chr(10).join(f"{name}: {desc}" for name, desc in parameters.items()) if parameters else "None"}
    
    Returns:
        {return_type}: Generated return value
    """
    # TODO: Implement function logic
    pass'''
        
        self._track_specialized_operation("generate_function", success=True)
        return code
    
    async def _generate_class_from_params(self, **kwargs) -> str:
        """Generate a class from explicit parameters."""
        class_name = kwargs.get('class_name', 'GeneratedClass')
        description = kwargs.get('description', 'Generated class')
        base_classes = kwargs.get('base_classes', [])
        methods = kwargs.get('methods', [])
        
        # Generate inheritance string
        base_str = ""
        if base_classes:
            base_str = f"({', '.join(base_classes)})"
        
        # Generate class code
        code = f'''class {class_name}{base_str}:
    """
    {description}
    """
    
    def __init__(self):
        """Initialize the {class_name}."""
        super().__init__()
'''
        
        # Add methods if specified
        for method in methods:
            method_name = method.get('name', 'method')
            method_desc = method.get('description', 'Generated method')
            code += f'''
    def {method_name}(self):
        """
        {method_desc}
        """
        # TODO: Implement method logic
        pass
'''
        
        self._track_specialized_operation("generate_class", success=True)
        return code
    
    async def _try_parse_code_request(self, input_text: str) -> Optional[str]:
        """Try to parse simple code generation requests."""
        input_lower = input_text.lower()
        
        # Simple function generation
        if 'function' in input_lower and ('called' in input_lower or 'named' in input_lower):
            import re
            func_match = re.search(r'function.*?(?:called|named)\s+([a-zA-Z_][a-zA-Z0-9_]*)', input_text, re.IGNORECASE)
            if func_match:
                func_name = func_match.group(1)
                code = f'''def {func_name}():
    """Generated function: {func_name}"""
    # TODO: Implement function logic
    pass'''
                self._track_specialized_operation("parse_function", success=True)
                return code
        
        # Simple class generation
        if 'class' in input_lower and ('called' in input_lower or 'named' in input_lower):
            import re
            class_match = re.search(r'class.*?(?:called|named)\s+([a-zA-Z_][a-zA-Z0-9_]*)', input_text, re.IGNORECASE)
            if class_match:
                class_name = class_match.group(1)
                code = f'''class {class_name}:
    """Generated class: {class_name}"""
    
    def __init__(self):
        """Initialize the {class_name}."""
        pass'''
                self._track_specialized_operation("parse_class", success=True)
                return code
        
        return None
    
    async def generate_python_function(self, function_name: str, description: str, 
                                     parameters: Optional[Dict[str, str]] = None,
                                     return_type: str = "Any") -> str:
        """
        Generate a Python function based on specifications.
        
        Args:
            function_name: Name of the function
            description: Description of what the function should do
            parameters: Dictionary of parameter names and types
            return_type: Return type annotation
            
        Returns:
            Generated function code
        """
        try:
            result = await self._generate_function_from_params(
                function_name=function_name,
                description=description,
                parameters=parameters or {},
                return_type=return_type
            )
            self._track_specialized_operation("generate_python_function", success=True)
            return result
        except Exception as e:
            self._track_specialized_operation("generate_python_function", success=False)
            self.specialized_logger.error(f"Error generating Python function: {e}")
            raise
    
    async def generate_python_class(self, class_name: str, description: str,
                                  base_classes: Optional[List[str]] = None,
                                  methods: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a Python class based on specifications.
        
        Args:
            class_name: Name of the class
            description: Description of the class purpose
            base_classes: List of base class names
            methods: List of method specifications
            
        Returns:
            Generated class code
        """
        try:
            result = await self._generate_class_from_params(
                class_name=class_name,
                description=description,
                base_classes=base_classes or [],
                methods=methods or []
            )
            self._track_specialized_operation("generate_python_class", success=True)
            return result
        except Exception as e:
            self._track_specialized_operation("generate_python_class", success=False)
            self.specialized_logger.error(f"Error generating Python class: {e}")
            raise
    
    async def generate_nanobrain_step(self, step_name: str, description: str,
                                    input_types: Optional[List[str]] = None,
                                    output_types: Optional[List[str]] = None) -> str:
        """
        Generate a NanoBrain Step class.
        
        Args:
            step_name: Name of the step class
            description: Description of the step's purpose
            input_types: List of input data types
            output_types: List of output data types
            
        Returns:
            Generated step class code
        """
        try:
            input_str = ", ".join(input_types) if input_types else "Any"
            output_str = ", ".join(output_types) if output_types else "Any"
            
            code = f'''from typing import Any, Dict
from nanobrain.core.step import Step, StepConfig


class {step_name}(Step):
    """
    {description}
    
    Input types: {input_str}
    Output types: {output_str}
    """
    
    def __init__(self, config: StepConfig):
        """Initialize the {step_name}."""
        super().__init__(config)
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the step inputs.
        
        Args:
            inputs: Input data dictionary
            
        Returns:
            Output data dictionary
        """
        # TODO: Implement step processing logic
        return {{"output": inputs.get("input", "")}}
'''
            
            self._track_specialized_operation("generate_nanobrain_step", success=True)
            return code
        except Exception as e:
            self._track_specialized_operation("generate_nanobrain_step", success=False)
            self.specialized_logger.error(f"Error generating NanoBrain step: {e}")
            raise
    
    async def write_code_to_file(self, code: str, file_path: str, 
                               description: Optional[str] = None) -> str:
        """
        Write generated code to a file using available tools.
        
        Args:
            code: Code content to write
            file_path: Path where to save the file
            description: Optional description of the file
            
        Returns:
            Result of the file writing operation
        """
        try:
            # This will use the LLM to handle file writing via tools
            request = f"""Please save the following code to {file_path}:

{description or 'Generated code'}

Code:
```
{code}
```

File path: {file_path}"""
            
            # Fall back to LLM processing for file operations
            if hasattr(self, 'process') and hasattr(super(), 'process'):
                result = await super().process(request)
                self._track_specialized_operation("write_code_to_file", success=True)
                return result
            else:
                self._track_specialized_operation("write_code_to_file", success=False)
                return f"Error: Cannot write code to file - no processing method available"
                
        except Exception as e:
            self._track_specialized_operation("write_code_to_file", success=False)
            self.specialized_logger.error(f"Error writing code to file: {e}")
            return f"Error writing code to file: {str(e)}"


class CodeWriterAgent(CodeWriterAgentMixin, SimpleSpecializedAgent):
    """
    Simple code writer agent that generates code without conversation history.
    
    This agent can generate code based on natural language descriptions
    and use other agents as tools for file operations.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="code_writer",
                description="Specialized agent for generating and writing code",
                model="gpt-4",
                system_prompt="""You are a specialized code generation agent for the NanoBrain framework.

Your capabilities:
1. Generate high-quality code based on natural language descriptions
2. Create functions, classes, and complete programs
3. Follow best practices and coding standards
4. Include comprehensive documentation and type hints
5. Handle multiple programming languages

When generating code:
1. Always include proper documentation and docstrings
2. Use appropriate type hints where applicable
3. Follow language-specific best practices
4. Include error handling where appropriate
5. Make code readable and maintainable

You can generate code for various purposes including:
- Python functions and classes
- NanoBrain framework components (Steps, Agents, Tools)
- Web applications and APIs
- Data processing scripts
- Utility functions and helpers""",
                temperature=0.3
            )
        
        super().__init__(config=config, **kwargs)


class ConversationalCodeWriterAgent(CodeWriterAgentMixin, ConversationalSpecializedAgent):
    """
    Conversational code writer agent that maintains conversation history.
    
    This agent can generate code based on natural language descriptions
    while maintaining conversation context for iterative development.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="conversational_code_writer",
                description="Conversational specialized agent for code generation",
                model="gpt-4",
                system_prompt="""You are a conversational code generation agent for the NanoBrain framework.

You maintain conversation context while generating code, allowing for iterative development and refinement.

Your capabilities:
1. Generate high-quality code based on natural language descriptions
2. Refine and improve code based on feedback
3. Maintain context across multiple code generation requests
4. Explain code and provide implementation guidance
5. Handle complex, multi-step development tasks

When working conversationally:
1. Remember previous code generations and modifications
2. Build upon earlier work in the conversation
3. Provide explanations and rationale for code decisions
4. Ask clarifying questions when requirements are unclear
5. Suggest improvements and alternatives

You can help with:
- Iterative code development and refinement
- Code reviews and improvements
- Complex multi-file projects
- Learning and educational coding assistance
- Debugging and troubleshooting""",
                temperature=0.4  # Slightly higher for conversational creativity
            )
        
        super().__init__(config=config, **kwargs)
 