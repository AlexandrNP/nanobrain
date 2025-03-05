# Framework UML Diagrams

This document contains UML class diagrams for the NanoBrain framework, split into logical components.

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

    DataUnitMemory --|> DataUnitBase
    DataUnitFile --|> DataUnitBase
    DataUnitFile --> WorkingMemory : uses
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

## Legend

The diagrams use standard UML notation:
- Inheritance: Solid arrow with triangle (--|>)
- Implementation: Dashed arrow with triangle (..|>)
- Composition/Usage: Solid arrow with diamond (-->)
- Abstract classes: Marked with <<abstract>>
- Interfaces: Marked with <<interface>>
- Biological analogies: Shown in italic text below class names

## Notes

1. Each diagram focuses on a specific aspect of the framework while showing clear relationships between components.
2. The diagrams are organized to show the hierarchical nature of the framework, from core foundation to specialized components.
3. Biological analogies are maintained in the relationships between components, mirroring natural neural systems.
4. The separation into different diagrams helps manage complexity while maintaining clarity of relationships within each subsystem. 