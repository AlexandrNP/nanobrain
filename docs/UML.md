# Framework UML Diagrams

This document contains UML class diagrams for the NanoBrain framework, split into logical components.

## Project Structure

```mermaid
graph TD
    Root[NanoBrain Root]
    Src[src/]
    Test[test/]
    Prompts[prompts/]
    Docs[docs/]
    DefaultConfigs[default_configs/]
    
    Root --> Src
    Root --> Test
    Root --> Prompts
    Root --> Docs
    Root --> DefaultConfigs
    
    Src --> CoreClasses["Core Classes<br>(ConfigManager, DirectoryTracer, etc.)"]
    Src --> DataClasses["Data Classes<br>(DataUnitBase, DataUnitMemory, etc.)"]
    Src --> LinkClasses["Link Classes<br>(LinkBase, LinkDirect, etc.)"]
    Src --> ExecutionClasses["Execution Classes<br>(Runner, Router, etc.)"]
    Src --> AgentClasses["Agent Classes<br>(Agent, Step, Workflow, etc.)"]
    
    Test --> TestFiles["Test Files<br>(test_agent.py, test_workflow.py, etc.)"]
    Test --> MockFiles["Mock Files<br>(mock_langchain.py)"]
    Test --> TestConfigs["Test Configs<br>(test_default_configs/)"]
    
    Prompts --> PromptTemplates["Prompt Templates<br>(templates.py)"]
    Prompts --> ToolCallingPrompt["Tool Calling Prompt<br>(tool_calling_prompt.py)"]
    
    Docs --> UMLDiagrams["UML Diagrams<br>(UML.md)"]
    Docs --> ToolCallingDocs["Tool Calling Docs<br>(tool_calling.md, tool_calling_summary.md)"]
```

## 1. Core Foundation and Configuration
```mermaid
classDiagram
    class ConfigManager["ConfigManager<br><i>Genetic regulation system</i>"] {
        -working_memory: WorkingMemory
        -adaptability: float
        +get_config(class_dir: str) : dict
        +update_config(updates: dict, threshold: float) : bool
    }
    
    class DirectoryTracer["DirectoryTracer<br><i>Cellular localization signals</i>"] {
        -module_name: str
        -relative_path: str
        +get_relative_path() : str
        +get_absolute_path() : str
    }
    
    class ConfigLoader["ConfigLoader<br><i>Protein synthesis machinery</i>"] {
        -working_memory: WorkingMemory
        -learning_rate: float
        -concept_network: dict
        +load(config_path: str)
        +create(class_name: str, **kwargs)
        -_construct_objects(config: dict)
    }
    
    class WorkingMemory["WorkingMemory<br><i>Short-term synaptic memory</i>"] {
        -items: dict
        -capacity: int
        -access_times: dict
        +store(key: str, value: Any) : bool
        +retrieve(key: str) : Any
        -_remove_lru()
    }

    ConfigLoader --> WorkingMemory : uses
    ConfigManager --> WorkingMemory : uses
```

## 2. Data Storage and Memory
```mermaid
classDiagram
    class DataUnitBase["DataUnitBase<br><i>Neural memory engram</i>"] {
        <<abstract>>
        #config_manager: ConfigManager
        -data: Any
        -decay_rate: float
        -persistence_level: float
        +get() : Any
        +set(data: Any)
        +decay()
        +consolidate()
    }
    
    class DataUnitMemory["DataUnitMemory<br><i>Working memory in prefrontal cortex</i>"] {
        -base_unit: DataUnitBase
        +get() : Any
        +set(data: Any)
        +decay()
        +consolidate()
    }
    
    class DataUnitFile["DataUnitFile<br><i>Long-term memory storage</i>"] {
        -base_unit: DataUnitBase
        -file_path: str
        -buffer: WorkingMemory
        +get() : Any
        +set(data: Any)
        +decay()
        +consolidate()
    }
    
    class WorkingMemory["WorkingMemory<br><i>Short-term synaptic memory</i>"] {
        -items: dict
        -capacity: int
        -access_times: dict
        +store(key: str, value: Any)
        +retrieve(key: str) : Any
    }
    
    class DataStorageBase["DataStorageBase<br><i>Memory system with retrieval mechanism</i>"] {
        -input: DataUnitBase
        -output: DataUnitBase
        -trigger: TriggerBase
        -processing_history: List[Dict]
        +process(inputs: List[Any]) : Any
        +start_monitoring()
        +stop_monitoring()
        +get_history() : List[Dict]
        +clear_history()
    }
    
    class Step["Step<br><i>Functional neural circuit</i>"] {
        <<abstract>>
        #executor: ExecutorBase
        #circuit_breaker: CircuitBreaker
        #state: ComponentState
        +process(inputs: List[Any]) : Any
        +execute(inputs: List[Any]) : Any
        +get_result() : Any
    }
    
    class TriggerBase["TriggerBase<br><i>Sensory neuron</i>"] {
        -runnable: Any
        -activation_gate: ActivationGate
        +check_condition() : bool
        +monitor()
    }

    DataUnitMemory --|> DataUnitBase
    DataUnitFile --|> DataUnitBase
    DataUnitFile --> WorkingMemory : uses
    DataStorageBase --|> Step
    DataStorageBase --> DataUnitBase : uses
    DataStorageBase --> TriggerBase : uses
```

## 3. Connection and Links
```mermaid
classDiagram
    class LinkBase["LinkBase<br><i>Neural synapse</i>"] {
        #input: DataUnitBase
        #output: DataUnitBase
        #config_manager: ConfigManager
        #connection_strength: ConnectionStrength
        #activation_gate: ActivationGate
        +process_signal() : float
        +transfer()
        +recover()
    }
    
    class LinkDirect["LinkDirect<br><i>Fast ionotropic synapses</i>"] {
        -base_link: LinkBase
        +transfer()
    }
    
    class LinkFile["LinkFile<br><i>Neuromodulatory pathways</i>"] {
        -base_link: LinkBase
        -input_folder: str
        -output_folder: str
        +transfer()
    }
    
    class ConnectionStrength["ConnectionStrength<br><i>Synaptic weight</i>"] {
        -strength: float
        -min_strength: float
        -max_strength: float
        +increase(amount: float)
        +decrease(amount: float)
        +adapt(source: float, target: float)
    }
    
    class ActivationGate["ActivationGate<br><i>Ion channel</i>"] {
        -threshold: float
        -level: float
        -state: ComponentState
        +receive_signal(strength: float) : bool
        +decay(amount: float)
    }

    LinkDirect --|> LinkBase
    LinkFile --|> LinkBase
    LinkBase --> ConnectionStrength : has
    LinkBase --> ActivationGate : has
```

## 4. Execution and Control
```mermaid
classDiagram
    class IRunnable["IRunnable<br><i>Neural activation interface</i>"] {
        <<interface>>
        +invoke() : Any
        +check_runnable_config() : bool
        +get_config() : dict
        +update_config() : bool
    }
    
    class Runner["Runner<br><i>Single neuron processing</i>"] {
        -package: PackageBase
        -circuit_breaker: CircuitBreaker
        -working_memory: WorkingMemory
        +invoke() : Any
        +check_runnable_config() : bool
    }
    
    class Router["Router<br><i>Neural junction with multiple outputs</i>"] {
        -executor: ExecutorBase
        -input_source: LinkBase
        -output_sinks: List[LinkBase]
        -circuit_breakers: Dict[CircuitBreaker]
        +execute()
        -_adaptive_route(data: Any)
        -_broadcast_route(data: Any)
    }
    
    class ExecutorBase["ExecutorBase<br><i>Neurotransmitter system</i>"] {
        <<abstract>>
        #config_manager: ConfigManager
        #runnable_types: Set[str]
        #energy_level: float
        +can_execute(type: str) : bool
        +execute(runnable: Any) : Any
        +recover_energy()
    }
    
    class ExecutorFunc["ExecutorFunc<br><i>Specialized neuron</i>"] {
        -base_executor: ExecutorBase
        -function: Callable
        +execute(runnable: Any) : Any
    }
    
    class ExecutorParallel["ExecutorParallel<br><i>Neural ensemble</i>"] {
        -base_executor: ExecutorBase
        -max_workers: int
        -queue: Queue
        +execute_batch(runnables: List) : List
    }

    Runner ..|> IRunnable
    Router ..|> IRunnable
    ExecutorFunc --|> ExecutorBase
    ExecutorParallel --|> ExecutorBase
```

## 5. Regulation and Control
```mermaid
classDiagram
    class SystemRegulator["SystemRegulator<br><i>Homeostatic control</i>"] {
        -target_value: float
        -acceptable_range: float
        -correction_strength: float
        +regulate(current: float) : float
        +is_stable() : bool
    }
    
    class SystemModulator["SystemModulator<br><i>Neuromodulator system</i>"] {
        -modulators: Dict[str, float]
        +get_modulator(name: str) : float
        +set_modulator(name: str, value: float)
        +update_from_event(event: str)
        +apply_regulation()
    }
    
    class CircuitBreaker["CircuitBreaker<br><i>Neural refractory period</i>"] {
        -failure_count: int
        -threshold: int
        -reset_timeout: float
        +record_success()
        +record_failure()
        +can_execute() : bool
    }
    
    class DeadlockDetector["DeadlockDetector<br><i>Inhibitory feedback circuit</i>"] {
        -resource_graph: Dict
        -timeout: float
        +request_resource(owner: str, resource: str) : bool
        +release_resource(owner: str, resource: str)
        -_detect_cycle(start: str) : bool
    }

    SystemModulator --> SystemRegulator : uses
```

## 6. Agent and Tool Calling
```mermaid
classDiagram
    class Step["Step<br><i>Specialized neural circuit</i>"] {
        #executor: ExecutorBase
        #circuit_breaker: CircuitBreaker
        #state: ComponentState
        +process(inputs: List[Any]) : Any
        +execute(inputs: List[Any]) : Any
        +get_result() : Any
    }
    
    class Agent["Agent<br><i>Higher-order cognitive processing</i>"] {
        -llm: Union[BaseLLM, BaseChatModel]
        -memory: List[Dict]
        -prompt_template: PromptTemplate
        -memory_window_size: int
        -tools: List[Step]
        -langchain_tools: List[BaseTool]
        +process(inputs: List[Any]) : Any
        +process_with_tools(inputs: List[Any]) : Any
        +add_tool(step: Step)
        +remove_tool(step: Step)
        +execute_tool(tool_name: str, args: List[Any]) : Any
        +get_context_history() : str
        +get_full_history() : str
        +clear_memories()
        +save_to_shared_context(key: str)
        +load_from_shared_context(key: str)
    }
    
    class BaseLLM["BaseLLM<br><i>Language processing region</i>"] {
        <<abstract>>
        +predict(prompt: str) : str
    }
    
    class BaseChatModel["BaseChatModel<br><i>Conversational processing region</i>"] {
        <<abstract>>
        +predict_messages(messages: List) : AIMessage
    }
    
    class LLMProviders["LLM Providers<br><i>Specialized language regions</i>"] {
        <<interface>>
        +OpenAI/ChatOpenAI
        +ChatAnthropic
        +ChatGoogleGenerativeAI
        +LlamaCpp
        +ChatMistralAI
    }
    
    class Workflow["Workflow<br><i>Neural network ensemble</i>"] {
        -steps: List[Step]
        -connections: Dict[str, List[str]]
        -inhibitory_signals: Dict[str, InhibitorySignal]
        -modulators: Dict[str, float]
        +add_step(step: Step, name: str)
        +connect(source: str, target: str)
        +process(inputs: Dict[str, Any]) : Dict[str, Any]
        +execute(inputs: Dict[str, Any]) : Dict[str, Any]
        -_organize_hierarchy()
        -_apply_modulator_effects()
        -_decay_inhibition()
    }
    
    class PromptTemplate["PromptTemplate<br><i>Cognitive schema</i>"] {
        -template: str
        -variables: List[str]
        +format(**kwargs) : str
        +from_template(template: str) : PromptTemplate
    }
    
    class BaseTool["BaseTool<br><i>Specialized cognitive tool</i>"] {
        <<abstract>>
        -name: str
        -description: str
        +__call__(*args, **kwargs) : Any
    }

    Agent --|> Step
    Workflow --> Step : contains
    Agent --> BaseTool : creates
    Agent --> PromptTemplate : uses
    Agent --> BaseLLM : uses
    Agent --> BaseChatModel : uses
    BaseLLM <-- LLMProviders : implements
    BaseChatModel <-- LLMProviders : implements
```

## 7. Testing Structure
```mermaid
classDiagram
    class TestAgent["TestAgent<br><i>Agent test suite</i>"] {
        +setUp()
        +test_initialization()
        +test_initialize_llm()
        +test_load_prompt_template()
        +test_load_prompt_template_from_file()
        +test_process()
        +test_get_full_history()
        +test_get_context_history()
        +test_clear_memories()
        +test_update_memories()
        +test_shared_context_operations()
    }
    
    class TestAgentTools["TestAgentTools<br><i>Agent tools test suite</i>"] {
        +setUp()
        +test_tool_registration()
        +test_add_tool()
        +test_remove_tool()
        +test_tool_creation()
        +test_process_with_tools()
    }
    
    class TestAgentCustomToolPrompt["TestAgentCustomToolPrompt<br><i>Custom tool prompt test suite</i>"] {
        +setUp()
        +test_parse_tool_call()
        +test_process_with_custom_prompt()
        +test_execute_tool_directly()
    }
    
    class MockLangchain["MockLangchain<br><i>LangChain mock implementations</i>"] {
        +MockChatOpenAI
        +MockOpenAI
        +MockSystemMessage
        +MockHumanMessage
        +MockAIMessage
        +MockPromptTemplate
        +MockConversationBufferMemory
        +MockConversationBufferWindowMemory
        +MockBaseTool
        +tool()
    }

    TestAgent --> MockLangchain : uses
    TestAgentTools --> MockLangchain : uses
    TestAgentCustomToolPrompt --> MockLangchain : uses
```

## Legend

The diagrams use standard UML notation:
- Inheritance: Solid arrow with triangle (--|>)
- Implementation: Dashed arrow with triangle (..|>)
- Composition/Usage: Solid arrow with diamond (-->)
- Abstract classes: Marked with <<abstract>>
- Interfaces: Marked with <<interface>>
- Biological analogies: Shown in italic text below class names

## Notes

1. The project is now organized into three main directories:
   - `src/`: Contains all source code files
   - `test/`: Contains all test files and mock implementations
   - `prompts/`: Contains prompt templates and tool calling prompts

2. Each diagram focuses on a specific aspect of the framework while showing clear relationships between components.

3. The diagrams are organized to show the hierarchical nature of the framework, from core foundation to specialized components.

4. Biological analogies are maintained in the relationships between components, mirroring natural neural systems.

5. The Agent class now includes tool calling capabilities, allowing it to use other Step objects as tools to perform specific tasks.

6. The Agent class supports multiple LLM providers (OpenAI, Anthropic, Google, Meta/Llama, Mistral) and can work with both chat-based models (BaseChatModel) and completion-based models (BaseLLM). 