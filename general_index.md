# Nanobrain Framework — General Index

## Root Files

| Path | Description |
|------|-------------|
| `pyproject.toml` | Python project configuration (dependencies, build settings) |
| `setup.py` | Legacy Python package setup |
| `requirements.txt` | Pip dependency list |
| `README.md` | Main project README |
| `.env` | Environment variables |
| `.gitignore` | Git ignore rules |
| `emergency_backup.sh` | Emergency backup shell script |
| `run_tests_aurora.sh` | Test runner for Aurora HPC |
| `gen_slides.py` | Presentation slide generator |
| `gen_deck.py` | Presentation deck generator |
| `CLEANUP_PLAN.md` | Repository cleanup plan |
| `CURRENT_STATE_ASSESSMENT.md` | Current state assessment of the codebase |
| `CLAUDE.md` | Claude AI assistant instructions |

## nanobrain/ — Core Framework Package

### nanobrain/core/ — Framework Core

| Path | Description |
|------|-------------|
| `core/__init__.py` | Core package exports |
| `core/agent.py` | Agent base classes (Agent, SimpleAgent, ConversationalAgent, AgentConfig) |
| `core/agent_response.py` | Agent response models (AgentResponse, AgentProcessingMetadata, ConversationContext) |
| `core/agent_logging.py` | Agent interaction logging (AgentLogger, AgentInteractionLog) |
| `core/step.py` | Step base classes (BaseStep, Step, TransformStep, AgentStep, StepConfig) |
| `core/workflow.py` | Workflow engine (Workflow, WorkflowConfig, create_workflow) |
| `core/workflow_graph.py` | Workflow DAG graph (WorkflowGraph) |
| `core/workflow_progress.py` | Workflow progress tracking (WorkflowProgress, ProgressReporter) |
| `core/workflow_validation.py` | Workflow validation (WorkflowValidator, ValidationIssue) |
| `core/workflow_orchestrator.py` | Alphavirus workflow orchestrator |
| `core/workflow_divergence.py` | Workflow divergence/branching (WorkflowDivergenceManager) |
| `core/executor.py` | Execution backends (LocalExecutor, ThreadExecutor, ProcessExecutor, ParslExecutor) |
| `core/link.py` | Data flow links (DirectLink, QueueLink, TransformLink, ConditionalLink, FileLink) |
| `core/trigger.py` | Event triggers (DataUnitChangeTrigger, TimerTrigger, ManualTrigger) |
| `core/data_unit.py` | Data containers (DataUnitMemory, DataUnitFile, DataUnitString, DataUnitStream) |
| `core/tool.py` | Tool system (FunctionTool, AgentTool, StepTool, LangChainTool, ToolRegistry) |
| `core/external_tool.py` | External tool integration (ExternalTool, InstallationStatus, DiagnosticReport) |
| `core/component_base.py` | Component base class (FromConfigBase) with config-driven instantiation |
| `core/event_types.py` | Event type enums (DataUnitEventType, TriggerEventType, LinkEventType) |
| `core/a2a_support.py` | Agent-to-Agent protocol support (A2AClient, A2ASupportMixin) |
| `core/mcp_support.py` | Model Context Protocol support (MCPClient, MCPSupportMixin, MCPTool) |
| `core/bioinformatics.py` | Bioinformatics base types (SequenceCoordinate, BioinformaticsDataUnit) |
| `core/card_schemas.py` | Agent/Tool card schemas (AgentCard, ToolCard, CardManager) |
| `core/auto_discovery.py` | Config-driven component discovery (ConfigDrivenDiscovery) |
| `core/logging_system.py` | Structured logging (NanoBrainLogger, SystemLogManager) |
| `core/async_logging.py` | Process-safe async logging (ProcessSafeLogger) |
| `core/optional_deps.py` | Optional dependency management |
| `core/pbs_resource_manager.py` | PBS HPC resource management (PBSResourceManager) |
| `core/progressive_scaling.py` | Progressive scaling mixin for tools |
| `core/prompt_template_manager.py` | Prompt template management (PromptTemplateManager) |
| `core/resource_monitor.py` | System resource monitoring (ResourceMonitor) |
| `core/sequence_manager.py` | Biological sequence management (SequenceManager, FastaParser) |
| `core/shared_resource.py` | Shared resource pool for workers (SharedResourcePool) |
| `core/shared_aggregation_step.py` | Multi-workflow result aggregation step |
| `core/worker_step_pool.py` | Worker step pool management (WorkerStepPool) |
| `core/distributed_resource_registry.py` | Distributed resource registry |
| `core/dynamic_executor_config.py` | Dynamic executor config generation |
| `core/academy_integration.py` | Academy platform integration |

### nanobrain/core/config/ — Configuration System

| Path | Description |
|------|-------------|
| `core/config/config_base.py` | Base config class (ConfigBase) with YAML loading |
| `core/config/config_manager.py` | Global config manager (ConfigManager, ProviderConfig) |
| `core/config/enhanced_config.py` | Enhanced config with A2A/tool cards |
| `core/config/enhanced_config_manager.py` | Enhanced config manager with card export |
| `core/config/yaml_config.py` | YAML config classes (YAMLConfig, YAMLWorkflowConfig) |
| `core/config/schema_generator.py` | JSON Schema generation (SchemaGenerator) |
| `core/config/schema_validator.py` | Config schema validation (SchemaValidator) |

### nanobrain/core/logging/ — Logging Subsystem

| Path | Description |
|------|-------------|
| `core/logging/workflow_tracer.py` | Workflow execution tracing (WorkflowTracer) |
| `core/logging/llm_optimized_tracer.py` | LLM-optimized tracing (LLMOptimizedTracer) |
| `core/logging/trace_analyzer.py` | Trace analysis (TraceAnalyzer) |
| `core/logging/workflow_integration.py` | Workflow tracing integration |
| `core/logging/llm_integration.py` | LLM tracing integration |
| `core/logging/simple_integration.py` | Simple logging integration |
| `core/logging/simple_llm_logger.py` | Simple LLM logger |

### nanobrain/core/guards/ — Trigger Guard System

| Path | Description |
|------|-------------|
| `core/guards/contextvar_core.py` | Context variable-based trigger guards |
| `core/guards/nanobrain_enforcer_meta.py` | Metaclass-based guard enforcement |
| `core/guards/nanobrain_guard_config.py` | Guard configuration |
| `core/guards/nanobrain_guard_mixin.py` | Guard mixin classes (TriggerGuardMixin) |
| `core/guards/nanobrain_trigger_guard.py` | Trigger guard implementation |
| `core/guards/trigger_integration.py` | Guard-trigger integration |

### nanobrain/core/distributed/ — Distributed Execution

| Path | Description |
|------|-------------|
| `core/distributed/workflow_execution.py` | Distributed workflow execution |

### nanobrain/core/testing/ — Test Infrastructure

| Path | Description |
|------|-------------|
| `core/testing/test_constants.py` | Test constants and result enums |
| `core/testing/test_result_config.py` | Test result configuration |

## nanobrain/library/ — Component Library

### library/agents/ — Agent Implementations

| Path | Description |
|------|-------------|
| `library/agents/specialized/base.py` | Specialized agent base classes |
| `library/agents/specialized/code_writer.py` | Code generation agent |
| `library/agents/specialized/file_writer.py` | File I/O agent |
| `library/agents/specialized/parsl_agent.py` | Parsl distributed agent |
| `library/agents/specialized/protein_synonym_agent.py` | Protein synonym resolution agent |
| `library/agents/specialized/query_analysis_agent.py` | Query analysis agent |
| `library/agents/specialized/viral_expert_agent.py` | Viral biology expert agent |
| `library/agents/specialized/virus_extraction_agent.py` | Virus species extraction agent |
| `library/agents/enhanced/collaborative_agent.py` | Collaborative agent with A2A+MCP |
| `library/agents/enhanced/delegation_engine.py` | Task delegation engine |
| `library/agents/conversational/enhanced_collaborative_agent.py` | Enhanced conversational collaborative agent |

### library/tools/ — Tool Implementations

| Path | Description |
|------|-------------|
| `library/tools/bioinformatics/bv_brc_tool.py` | BV-BRC bioinformatics data tool |
| `library/tools/bioinformatics/mmseqs_tool.py` | MMseqs2 sequence clustering tool |
| `library/tools/bioinformatics/muscle_tool.py` | MUSCLE sequence alignment tool |
| `library/tools/bioinformatics/pubmed_client.py` | PubMed literature search client |
| `library/tools/search/elasticsearch_tool.py` | Elasticsearch search tool |
| `library/tools/search/elasticsearch_mcp_server.py` | Elasticsearch MCP server |
| `library/tools/search/csv_processor.py` | CSV file processor |
| `library/tools/search/csv_fuzzy_matcher.py` | CSV fuzzy search matcher |
| `library/tools/search/csv_bulk_indexer.py` | CSV bulk indexer for Elasticsearch |
| `library/tools/search/csv_mapping_generator.py` | CSV-to-Elasticsearch mapping generator |
| `library/tools/search/csv_search_optimizer.py` | CSV search query optimizer |
| `library/tools/search/csv_validator.py` | CSV data validator |

### library/workflows/ — Workflow Implementations

| Path | Description |
|------|-------------|
| `library/workflows/viral_protein_analysis/` | Viral protein PSSM analysis workflow (main scientific workflow) |
| `library/workflows/chatbot_viral_integration/` | Chatbot + viral analysis integration workflow |
| `library/workflows/rag/` | RAG (Retrieval-Augmented Generation) workflow |
| `library/workflows/chat_workflow/` | Basic chat workflow |
| `library/workflows/chat_workflow_parsl/` | Parsl-distributed chat workflow |
| `library/workflows/conversational/` | Conversational viral expert workflow |
| `library/workflows/simple_workflow/` | Simple demo workflow |
| `library/workflows/sub_workflow.py` | Sub-workflow step wrapper |

### library/infrastructure/ — Infrastructure Components

| Path | Description |
|------|-------------|
| `library/infrastructure/steps/` | Infrastructure steps (parallel, routing, web interface, progress tracking) |
| `library/infrastructure/data/` | Data management (sessions, cache, conversations, exports) |
| `library/infrastructure/docker/` | Docker container management |
| `library/infrastructure/deployment/` | Deployment utilities (port management) |
| `library/infrastructure/load_balancing/` | Load balancing (circuit breaker, request queue) |
| `library/infrastructure/monitoring/` | Monitoring (performance, resources, health) |
| `library/infrastructure/interfaces/` | Interface abstractions (CLI, database, external APIs) |
| `library/infrastructure/links/` | Infrastructure link implementations |
| `library/infrastructure/triggers/` | Infrastructure trigger implementations |

### library/interfaces/web/ — Web Interface

| Path | Description |
|------|-------------|
| `library/interfaces/web/web_interface.py` | Main web interface (WebInterface, FastAPI app) |
| `library/interfaces/web/chatbot_server.py` | Chatbot server |
| `library/interfaces/web/demo_server.py` | Demo server |
| `library/interfaces/web/servers/` | Server implementations (universal, factory) |
| `library/interfaces/web/routing/` | Request routing (workflow router, registry, strategies) |
| `library/interfaces/web/analysis/` | Request analysis (intent, domain classifiers) |
| `library/interfaces/web/processing/` | Response processing (format converter, streaming) |
| `library/interfaces/web/api/` | API routers (chat, health, websocket, frontend) |
| `library/interfaces/web/models/` | Pydantic models (request, response, universal, workflow) |
| `library/interfaces/web/middleware/` | Middleware (CORS, logging) |
| `library/interfaces/web/config/` | Web config management |

### library/components/ — Shared Components

| Path | Description |
|------|-------------|
| `library/components/faiss_builder.py` | FAISS vector database builder |
| `library/components/mmd_parser.py` | MMD paper parser |
| `library/components/semantic_chunker.py` | Semantic text chunker |

### library/llms/ — LLM Integration

| Path | Description |
|------|-------------|
| `library/llms/local_llm_client.py` | Local LLM client (vLLM) |
| `library/llms/shared_vllm_server.py` | Shared vLLM server |
| `library/llms/vllm_server_manager.py` | vLLM server lifecycle manager |

### library/utils/ — Utilities

| Path | Description |
|------|-------------|
| `library/utils/parameter_grid.py` | Parameter grid generator for experiments |

## nanobrain/cleanup/ — Repository Cleanup System

| Path | Description |
|------|-------------|
| `cleanup/orchestrator.py` | Main cleanup orchestrator with CLI |
| `cleanup/backup_manager.py` | Repository backup manager |
| `cleanup/cleanup_manager.py` | Temporary file cleanup |
| `cleanup/demo_manager.py` | Demo directory consolidation |
| `cleanup/demo_fixer.py` | Target demo fixes |
| `cleanup/import_resolver.py` | Import issue resolution |
| `cleanup/config_manager.py` | Configuration standardization |
| `cleanup/doc_manager.py` | Documentation consolidation |
| `cleanup/dependency_manager.py` | Dependency cleanup |
| `cleanup/structure_manager.py` | Repository structure reorganization |
| `cleanup/validator.py` | Comprehensive system validation |
| `cleanup/models.py` | Cleanup data models |

## nanobrain/lightweight/ — Lightweight Discovery & Builder

| Path | Description |
|------|-------------|
| `lightweight/cli.py` | CLI for discovery and workflow building |
| `lightweight/discovery.py` | Config-driven component discovery |
| `lightweight/comprehensive_discovery.py` | Comprehensive AST-based discovery |
| `lightweight/refined_discovery.py` | Refined discovery with schema extraction |
| `lightweight/workflow_builder.py` | Simple workflow builder |
| `lightweight/enhanced_workflow_builder.py` | Enhanced workflow builder with validation |
| `lightweight/examples.py` | Usage examples |

## nanobrain/academy_integration/ — Academy Platform Integration

| Path | Description |
|------|-------------|
| `academy_integration/academy_agent_step.py` | Academy agent step wrapper |
| `academy_integration/academy_link.py` | Academy link for remote agent communication |

## demos/ — Demo Implementations

| Path | Description |
|------|-------------|
| `demos/viral_pssm_workflow/` | Viral protein PSSM analysis demo (primary) |
| `demos/rag_database_creation/` | RAG database creation demo (primary) |
| `demos/academylink_aurora_demo/` | Academy link Aurora HPC demo |
| `demos/simple_demo/` | Simple workflow demo |

## tests/ — Test Suite

| Path | Description |
|------|-------------|
| `tests/unit/` | Unit tests for cleanup managers and core components |
| `tests/integration/` | Integration tests (backup, cleanup, orchestrator, demos) |
| `tests/core/` | Core framework tests |
| `tests/performance/` | Performance and stress tests |
| `tests/playwright/` | Frontend E2E tests |
| `tests/production/` | Production simulation and security tests |
| `tests/utils/` | Test utilities |

## config/ — Configuration Files

| Path | Description |
|------|-------------|
| `config/templates/` | Config templates (workflow, executor, environment) |
| `config/workflows/` | Workflow configurations (chat, chatbot_viral) |
| `config/core/` | Core executor configurations |
| `config/infrastructure/` | Infrastructure configs (data units, docker, steps) |

## docs/ — Documentation

| Path | Description |
|------|-------------|
| `docs/01_FRAMEWORK_CORE_ARCHITECTURE.md` | Core architecture documentation |
| `docs/03_WEB_ARCHITECTURE.md` | Web interface architecture |
| `docs/04_LLM_CODE_GENERATION.md` | LLM code generation guide |
| `docs/05_TESTING_AND_VALIDATION.md` | Testing and validation guide |
| `docs/API_REFERENCE.md` | API reference |
| `docs/distributed_execution.md` | Distributed execution guide |
| `docs/step_*.md` | Viral PSSM workflow step documentation |

## Other Directories

| Path | Description |
|------|-------------|
| `archive/` | Archived demo implementations |
| `scripts/` | Utility scripts |
| `docker/` | Docker configurations |
| `containers/` | Container definitions |
| `examples/` | Additional examples |
| `data/` | Data files |
| `results/` | Output results |
| `requirements/` | Additional requirement files |
