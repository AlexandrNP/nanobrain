import asyncio
from typing import List, Any
from src.enums import ComponentState
from src.ExecutorBase import ExecutorBase
from src.concurrency import DeadlockDetector
from src.regulations import SystemModulator
from src.Step import Step


class Workflow(Step):
    """
    Container for multiple steps forming a workflow.
    
    Biological analogy: Functional brain network.
    Justification: Like how functional brain networks involve multiple
    interconnected cortical areas working together to accomplish complex
    tasks, workflows involve multiple interconnected steps working
    together to accomplish complex processing.
    """
    def __init__(self, executor: ExecutorBase, steps: List[Step] = None, **kwargs):
        # Initialize the Step base class
        super().__init__(executor, **kwargs)
        
        # Workflow-specific attributes
        self.steps = []
        self.links = []
        self.deadlock_detector = DeadlockDetector()
        self.step_order = {}  # For hierarchical processing
        self.active_inhibition = {}  # Step ID -> inhibition level
        self.system_modulators = SystemModulator()
        self.network_efficiency = 0.5  # Overall network efficiency (0.0-1.0)
        
        # Create steps from configuration if provided
        if 'steps_config' in kwargs:
            self.create_steps_from_config(kwargs['steps_config'])
        elif steps:
            self.steps = steps
            
        # Create links from configuration if provided
        if 'links_config' in kwargs:
            self.create_links_from_config(kwargs['links_config'])
            
        # Organize steps into a hierarchy
        self.organize_hierarchy()
    
    def create_steps_from_config(self, steps_config: List[dict]):
        """
        Create steps from configuration dictionaries.
        
        Biological analogy: Cell differentiation from genetic instructions.
        Justification: Like how cells differentiate into specific types based on
        genetic and environmental factors, steps are created with specific
        configurations for their roles.
        """
        for step_config in steps_config:
            step_class = step_config.pop('class')
            step = self.config_manager.create_instance(step_class, executor=self.executor, **step_config)
            self.steps.append(step)
            
    def create_links_from_config(self, links_config: List[dict]):
        """
        Create links between steps from configuration.
        
        Biological analogy: Synapse formation between neurons.
        Justification: Like how neurons form specific connections based on
        molecular signals, links are created between steps based on configuration.
        """
        for link_config in links_config:
            # Get source and target steps
            source_idx = link_config.pop('source_step')
            target_idx = link_config.pop('target_step')
            
            if 0 <= source_idx < len(self.steps) and 0 <= target_idx < len(self.steps):
                source_step = self.steps[source_idx]
                target_step = self.steps[target_idx]
                
                # Create link using configuration
                link_class = link_config.pop('class', 'LinkDirect')
                link = self.config_manager.create_instance(
                    link_class,
                    input_data=source_step.output,
                    output_data=target_step.input,
                    **link_config
                )
                
                # Register link with steps
                source_step.register_output_sink(link)
                target_step.register_input_source(link)
                self.links.append(link)
    
    async def execute(self):
        """
        Executes steps in the workflow with network dynamics.
        
        Biological analogy: Activation flow through a brain network.
        Justification: Like how activity propagates through interconnected
        brain regions with regulatory mechanisms, execution flows through
        workflow steps with coordination mechanisms.
        """
        results = {}
        executed_steps = set()
        
        # Apply modulator effects to network
        self.apply_modulator_effects()
        
        # Execute steps based on their hierarchical level
        for level in sorted(set(self.step_order.values())):
            level_steps = [step for step, order in self.step_order.items() if order == level]
            
            # Execute steps at this level in parallel
            tasks = []
            for step_idx, step in enumerate(self.steps):
                if step_idx in level_steps and step_idx not in executed_steps:
                    step_id = f"step_{step_idx}"
                    
                    # Check for inhibition
                    inhibition = self.active_inhibition.get(step_id, 0.0)
                    if inhibition > 0.5:
                        # Step is currently inhibited
                        continue
                    
                    # Request resources with deadlock detection
                    if not self.deadlock_detector.request_resource(id(self), step_id):
                        # Skip this step for now, will be retried
                        continue
                        
                    # Execute step
                    task = asyncio.create_task(self.execute_step(step, step_id))
                    tasks.append((step_idx, task))
                    executed_steps.add(step_idx)
            
            # Wait for all tasks at this level to complete
            for step_idx, task in tasks:
                try:
                    result = await task
                    results[f"step_{step_idx}"] = result
                    
                    # Success updates system state
                    self.system_modulators.update_from_event("success", 0.05)
                except Exception as e:
                    # Failure affects system state
                    self.system_modulators.update_from_event("failure", 0.1)
                    results[f"step_{step_idx}"] = None
        
        # Apply homeostatic regulation to system modulators
        self.system_modulators.apply_regulation()
        
        # Store the overall result
        self.result = results
        
        return results
    
    async def execute_step(self, step: Step, step_id: str):
        """
        Execute a single step with system dynamics.
        
        Biological analogy: Controlled activation of a specific brain region.
        Justification: Like how the brain activates specific regions while
        maintaining overall network coordination, the workflow executes
        specific steps while managing resource allocation.
        """
        try:
            # Apply any modulation to the step
            self.modulate_step(step)
            
            # Execute the step
            result = await step.execute()
            
            # Release resources
            self.deadlock_detector.release_resource(id(self), step_id)
            
            return result
        except Exception as e:
            # Release resources even on failure
            self.deadlock_detector.release_resource(id(self), step_id)
            
            # Apply inhibition to this step to prevent immediate retry
            self.active_inhibition[step_id] = 0.8  # Strong inhibition
            
            # Schedule inhibition decay
            asyncio.create_task(self.decay_inhibition(step_id))
            
            raise e
    
    async def decay_inhibition(self, step_id: str):
        """
        Gradually reduce inhibition on a step.
        
        Biological analogy: Recovery of inhibited neural circuits.
        Justification: Like how inhibited neural circuits gradually
        recover their excitability, inhibited workflow steps gradually
        recover their ability to execute.
        """
        while self.active_inhibition.get(step_id, 0) > 0:
            await asyncio.sleep(1.0)
            current = self.active_inhibition.get(step_id, 0)
            self.active_inhibition[step_id] = max(0, current - 0.1)
            
            if self.active_inhibition[step_id] == 0:
                del self.active_inhibition[step_id]
    
    def apply_modulator_effects(self):
        """
        Apply effects of system modulators to the workflow.
        
        Biological analogy: Neuromodulator effects on brain networks.
        Justification: Like how brain-wide neuromodulators affect network
        properties (e.g., dopamine affects working memory function),
        system modulators affect workflow properties.
        """
        # Performance modulator affects network efficiency
        performance = self.system_modulators.get_modulator("performance")
        self.network_efficiency = 0.3 + (performance * 0.7)  # 0.3-1.0 range
        
        # Reliability modulator affects inhibition
        reliability = self.system_modulators.get_modulator("reliability")
        # High reliability reduces existing inhibition (increases stability)
        if reliability > 0.7:
            for step_id in list(self.active_inhibition.keys()):
                self.active_inhibition[step_id] *= (1.0 - (reliability - 0.7))
    
    def modulate_step(self, step: Step):
        """
        Apply system modulator effects to a specific step.
        
        Biological analogy: Regional neuromodulation.
        Justification: Like how neuromodulators can have specific effects
        on particular brain regions, system modulators can have specific
        effects on particular workflow steps.
        """
        # Adaptability modulator affects step plasticity
        adaptability = self.system_modulators.get_modulator("adaptability")
        step.adaptability = (step.adaptability * 0.7) + (adaptability * 0.3)
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process inputs by executing the workflow.
        
        Biological analogy: Distributed processing in neural networks.
        Justification: Like how complex operations in the brain emerge from
        the interactions of many neurons rather than a single computation,
        workflow processing emerges from the interactions of many steps.
        """
        # For workflows, we ignore the inputs and just execute the workflow
        return await self.execute()
    
    def organize_hierarchy(self):
        """
        Organize steps into a hierarchical processing order.
        
        Biological analogy: Hierarchical organization of brain networks.
        Justification: Like how the brain organizes processing in hierarchical
        networks (e.g., visual processing hierarchy), workflows organize
        steps in dependency-based hierarchies.
        """
        # Simple strategy: build dependency graph and assign levels
        dependencies = {i: set() for i in range(len(self.steps))}
        
        # Find dependencies based on connections
        for i, step in enumerate(self.steps):
            for j, other_step in enumerate(self.steps):
                if i == j:
                    continue
                    
                # Check if other_step depends on step (output of step is input to other_step)
                if step.output_sink and other_step.input_sources:
                    for input_source in other_step.input_sources:
                        if step.output_sink.output == input_source.input:
                            dependencies[j].add(i)  # j depends on i
        
        # Assign levels based on dependencies
        assigned_levels = {}
        
        def assign_level(step_idx, level=0):
            if step_idx in assigned_levels:
                assigned_levels[step_idx] = max(assigned_levels[step_idx], level)
            else:
                assigned_levels[step_idx] = level
                
            # Assign levels to all steps that depend on this one
            for dep_idx, deps in dependencies.items():
                if step_idx in deps:
                    assign_level(dep_idx, level + 1)
        
        # Find root steps (no dependencies) and assign levels
        for step_idx, deps in dependencies.items():
            if not deps:
                assign_level(step_idx, 0)
        
        self.step_order = assigned_levels