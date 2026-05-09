"""
Event-Driven Alphavirus Workflow - Enhanced from_config Compliance

Pure event routing workflow with minimal logic.
All business logic delegated to steps with event-driven execution.
Enhanced for framework compliance with Dict-based configuration format.
"""

import asyncio
import time
import importlib
from typing import Dict, Any

from nanobrain.core.workflow import Workflow, WorkflowConfig


class AlphavirusWorkflow(Workflow):
    """
    Event-Driven Viral Protein Analysis Workflow

    Pure event routing workflow with minimal logic.
    All execution happens via event-driven data flow.

    Enhanced for framework compliance:
    - Dict-based steps configuration
    - Enhanced from_config patterns
    - Configuration-driven component creation
    - Step-level data units and triggers
    """

    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize event-driven workflow as pure event router"""
        super()._init_from_config(config, component_config, dependencies)

        # Create ONLY workflow-level input/output data units
        self.input_data_unit = None
        self.output_data_unit = None

        # Initialize workflow input/output data units using proper from_config pattern
        if hasattr(config, 'input_data_units') and config.input_data_units:
            input_configs = config.input_data_units
            if 'workflow_input' in input_configs:
                unit_config = input_configs['workflow_input']
                # Use proper from_config pattern - import class and call its from_config method
                class_path = unit_config.get('class', 'nanobrain.core.data_unit.DataUnitMemory')
                module_path, class_name = class_path.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_path)
                data_unit_class = getattr(module, class_name)

                # Ensure config has class field for proper data unit creation
                enhanced_config = unit_config.copy()
                if 'class' not in enhanced_config:
                    enhanced_config['class'] = class_path

                self.input_data_unit = data_unit_class.from_config(enhanced_config)

        if hasattr(config, 'output_data_units') and config.output_data_units:
            output_configs = config.output_data_units
            if 'workflow_output' in output_configs:
                unit_config = output_configs['workflow_output']
                # Use proper from_config pattern - import class and call its from_config method
                class_path = unit_config.get('class', 'nanobrain.core.data_unit.DataUnitFile')
                module_path, class_name = class_path.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_path)
                data_unit_class = getattr(module, class_name)

                # Ensure config has class field for proper data unit creation
                enhanced_config = unit_config.copy()
                if 'class' not in enhanced_config:
                    enhanced_config['class'] = class_path

                self.output_data_unit = data_unit_class.from_config(enhanced_config)

        # Enhanced framework handles automatic component instantiation
        # No programmatic component creation needed

        # Steps are automatically instantiated from Dict-based configuration
        self.steps = {}
        if hasattr(config, 'steps') and isinstance(config.steps, dict):
            for step_id, step_config in config.steps.items():
                # Enhanced ConfigBase automatically handles class+config instantiation
                step_instance = self._resolve_step_from_config(step_config)
                self.steps[step_id] = step_instance

        # Links and triggers automatically resolved by enhanced framework
        # No manual creation required - framework handles via class+config pattern

        # NO triggers at workflow level - all are step-level
        # NO tools at workflow level - all are step-level
        # NO business logic - pure event routing

    def _resolve_step_from_config(self, step_config):
        """Resolve step from configuration using enhanced patterns"""
        # Handle case where enhanced framework has already resolved steps to instances
        if hasattr(step_config, '__class__') and hasattr(step_config, 'process'):
            # step_config is already a step instance - return it directly
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"✅ Using pre-resolved step: {step_config.__class__.__name__}")
            return step_config

        # Handle case where step_config is still a dictionary configuration
        if isinstance(step_config, dict):
            step_class = step_config.get('class')
            config_path = step_config.get('config')

            if not step_class or not config_path:
                raise ValueError("Step configuration must include 'class' and 'config' fields")

            # Enhanced from_config handles automatic instantiation
            module_path, class_name = step_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            step_cls = getattr(module, class_name)

            try:
                step_instance = step_cls.from_config(
                    config_path,
                    workflow_directory=getattr(self.config, 'workflow_directory', '')
                )

                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(f"✅ Created step: {step_class}")
                return step_instance

            except Exception as e:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.error(f"❌ Failed to create step {step_class}: {e}")
                raise

        # Invalid step_config type
        raise ValueError(f"Invalid step configuration type: {type(step_config)}. Expected dict or step instance.")



    async def initialize(self) -> None:
        """Initialize workflow and all components"""
        await super().initialize()

        # Initialize workflow data units
        if self.input_data_unit:
            await self.input_data_unit.initialize()
        if self.output_data_unit:
            await self.output_data_unit.initialize()

        # Initialize all steps
        for step_id, step in self.steps.items():
            await step.initialize()
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"Initialized step: {step_id}")

        # Initialize all links (sets up automatic transfer triggers)
        for link_id, link in self.links.items():
            await link.initialize()
            await link.start()  # Start auto-transfer triggers
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"Initialized and started link: {link_id}")
        # Register lightweight execution tracking for step I/O
        try:
            self._register_step_execution_tracking()
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info("🔎 Step execution tracking registered")
        except Exception:
            pass


        # Simple approach: Links will be activated directly when data is set to workflow_input

    async def _activate_workflow_input_links(self) -> None:
        """Simple solution: Directly activate all links that have workflow_input as source"""
        try:
            activated_links = []

            # Debug: Show all link configurations
            if hasattr(self, 'nb_logger') and self.nb_logger:
                # Check different possible link storage locations
                links_dict = getattr(self, 'links', {})
                child_links = getattr(self, 'child_links', {})
                workflow_links = getattr(self, 'workflow_links', {})

                self.nb_logger.info("🔧 Links storage debug:")
                self.nb_logger.info(f"🔧   self.links: {len(links_dict)} items")
                self.nb_logger.info(f"🔧   self.child_links: {len(child_links)} items")
                self.nb_logger.info(f"🔧   self.workflow_links: {len(workflow_links)} items")

                # Use whichever has links
                if links_dict:
                    active_links = links_dict
                    link_source = "self.links"
                elif child_links:
                    active_links = child_links
                    link_source = "self.child_links"
                elif workflow_links:
                    active_links = workflow_links
                    link_source = "self.workflow_links"
                else:
                    active_links = {}
                    link_source = "NONE"

                self.nb_logger.info(f"🔧 Using {link_source} with {len(active_links)} links:")
                for link_id, link in active_links.items():
                    config_source = getattr(link.config, 'source', 'NO_CONFIG_SOURCE') if hasattr(link, 'config') else 'NO_CONFIG'
                    self.nb_logger.info(f"🔧   Link {link_id}: config.source = '{config_source}'")

            # Use the correct links collection (discovered above)
            links_to_check = active_links if 'active_links' in locals() else getattr(self, 'workflow_links', {})

            for link_id, link in links_to_check.items():
                # Check if this link has workflow_input as source
                if (hasattr(link, 'config') and hasattr(link.config, 'source') and
                    link.config.source == 'workflow_input'):

                    # Directly trigger the link transfer
                    await link._on_source_data_changed({
                        'operation': 'set',
                        'data_unit': 'workflow_input',
                        'timestamp': time.time()
                    })
                    activated_links.append(link_id)

                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.info(f"🔗 Activated link: {link_id}")

            if activated_links:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(f"✅ Activated {len(activated_links)} links from workflow_input: {activated_links}")
            else:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.warning("⚠️ No links found with workflow_input as source")

        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to activate workflow_input links: {e}")

    async def _create_workflow_input_trigger(self) -> None:
        """Create trigger for workflow_input to activate the first link when data is set"""
        try:
            workflow_input = self.step_input_data_units.get('workflow_input')

            # Debug: Check what type workflow_input actually is
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🔧 workflow_input type: {type(workflow_input)}, value: {workflow_input}")

            if not workflow_input:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.warning("No workflow_input found - skipping workflow input trigger creation")
                return

            # Create trigger config for workflow input (framework-compliant dictionary)
            trigger_config = {
                'trigger_type': 'data_updated',  # Maps to DataUnitChangeTrigger in new architecture
                'data_unit': 'workflow_input',
                'event_type': 'set',
                'description': 'Triggers workflow execution when data is set to workflow_input',
                'enable_logging': True
            }

            # Create step context for trigger resolution
            step_context = {
                'step_input_data_units': self.step_input_data_units,
                'step_output_data_units': self.step_output_data_units,
                'step_name': self.name,
                'step_scope_only': True
            }

            # Debug: Check what data units are available
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🔧 Available step input data units: {list(self.step_input_data_units.keys())}")
                self.nb_logger.info(f"🔧 Available step output data units: {list(self.step_output_data_units.keys())}")

            # Create trigger directly using from_config with dependencies
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🔧 Creating workflow input trigger with config: {trigger_config}")

            from nanobrain.core.trigger import DataUnitChangeTrigger

            # Create trigger directly with dependencies
            workflow_trigger = DataUnitChangeTrigger.from_config(
                trigger_config,
                data_unit=workflow_input,
                event_type='set'
            )

            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info(f"🔧 Trigger creation result: {workflow_trigger}")

            if workflow_trigger:
                # Bind action to trigger all workflow_input links when data is set
                workflow_trigger.bind_action(self._on_workflow_input_data_set)

                # Start monitoring
                await workflow_trigger.start_monitoring()

                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info("✅ Created workflow input trigger - will activate all workflow_input links when data is set")
            else:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.warning("Failed to create workflow input trigger")

        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to create workflow input trigger: {e}")

    async def _on_workflow_input_data_set(self, trigger_event: Dict[str, Any]) -> None:
        """Triggered when data is set to workflow_input - activates all associated links"""
        try:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.info("🔥 Workflow input trigger fired - activating links from workflow_input")

            # Find and activate all links that have workflow_input as their source
            activated_links = []
            for link_id, link in self.links.items():
                link_activated = False

                # Check if this link has workflow_input as source
                if hasattr(link, 'config') and hasattr(link.config, 'source') and link.config.source == 'workflow_input':
                    # Activate the link by calling its transfer method
                    await link._on_source_data_changed(trigger_event)
                    activated_links.append(link_id)
                    link_activated = True
                elif hasattr(link, 'source') and hasattr(link.source, 'name') and link.source.name == 'workflow_input':
                    # Alternative: activate via source data unit name
                    await link._on_source_data_changed(trigger_event)
                    activated_links.append(link_id)
                    link_activated = True

                if link_activated and hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(f"🔗 Activated link: {link_id}")

            if activated_links:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(f"✅ Successfully activated {len(activated_links)} links from workflow_input: {activated_links}")
            else:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.warning("⚠️ No links found with workflow_input as source")

        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to activate workflow_input links: {e}")

    def _find_first_link(self):
        """Find the first link that starts from workflow_input"""
        if hasattr(self, 'nb_logger') and self.nb_logger:
            # Debug: Show all links and their source references
            for link_id, link in self.links.items():
                source_ref = getattr(link, 'source_ref', 'NO_SOURCE_REF')
                config_source = getattr(link.config, 'source', 'NO_CONFIG_SOURCE') if hasattr(link, 'config') else 'NO_CONFIG'
                source_name = getattr(link.source, 'name', 'NO_SOURCE_NAME') if hasattr(link, 'source') else 'NO_SOURCE'
                self.nb_logger.info(f"🔧 Link {link_id}: source_ref='{source_ref}', config.source='{config_source}', source.name='{source_name}'")

        # Check multiple possible ways the source reference might be stored
        for link_id, link in self.links.items():
            # Method 1: Check source_ref attribute
            if hasattr(link, 'source_ref') and link.source_ref == 'workflow_input':
                return link

            # Method 2: Check config.source
            if hasattr(link, 'config') and hasattr(link.config, 'source') and link.config.source == 'workflow_input':
                return link

            # Method 3: Check source data unit name
            if hasattr(link, 'source') and hasattr(link.source, 'name') and link.source.name == 'workflow_input':
                return link

        return None

    async def process(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANT: Process workflow via pure event-driven flow

        Args:
            input_data: Input data for the workflow

        Returns:
            Dict containing workflow results via event-driven completion
        """
        try:
            # Get workflow-level data units (step-owned, not instance variables)
            workflow_input = self.step_input_data_units.get('workflow_input')
            workflow_output = self.step_output_data_units.get('workflow_output')

            if not workflow_input:
                raise ValueError("Workflow input data unit 'workflow_input' not found")
            if not workflow_output:
                raise ValueError("Workflow output data unit 'workflow_output' not found")

            # Set up event-driven completion detection
            completion_event = asyncio.Event()
            result_data = {'output': None}

            def on_workflow_completion(data_unit, data):
                """Event-driven completion handler"""
                if data is not None:
                    result_data['output'] = data
                    completion_event.set()

            # Register for event-driven completion notification
            workflow_output.register_change_listener(on_workflow_completion)

            try:
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info("🚀 Started enhanced event-driven workflow execution")

                # Trigger event-driven pipeline by setting input data
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(f"🔄 Setting data to workflow_input: {input_data}")
                await workflow_input.set(input_data or {})
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info("✅ Data successfully set to workflow_input")

                # Simple solution: Directly activate first link from workflow_input
                await self._activate_workflow_input_links()

                # Wait for event-driven completion (no polling!)
                timeout = getattr(self.config, 'execution_timeout_seconds', 300)  # Configurable timeout, default 5 minutes
                if hasattr(self, 'nb_logger') and self.nb_logger:
                    self.nb_logger.info(f"⏱️ Waiting for workflow completion with {timeout}s timeout")

                try:
                    await asyncio.wait_for(completion_event.wait(), timeout=timeout)

                    if hasattr(self, 'nb_logger') and self.nb_logger:
                        self.nb_logger.info("✅ Enhanced event-driven workflow completed successfully")

                    return result_data['output']

                except asyncio.TimeoutError:
                    # Add context on which steps progressed to aid debugging
                    try:
                        progressed_steps = [sid for sid, step in self.steps.items() if hasattr(step, 'nb_logger')]
                        self.nb_logger.error(f"⏰ Workflow timed out after {timeout}s. Steps present: {progressed_steps}")
                    except Exception:
                        pass
                    raise TimeoutError(f"Workflow execution timed out after {timeout} seconds")

            finally:
                # Clean up event listener
                try:
                    workflow_output.unregister_change_listener(on_workflow_completion)
                except:
                    pass  # Ignore cleanup errors

        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.error(f"❌ Enhanced event-driven workflow execution failed: {e}")
            raise


    def _register_step_execution_tracking(self) -> None:
        """Attach listeners to step data units to log current step and cache hints."""
        try:
            for step_id, step in self.steps.items():
                # Track input updates
                if hasattr(step, 'step_input_data_units'):
                    for name, du in step.step_input_data_units.items():
                        if hasattr(du, 'register_change_listener'):
                            du.register_change_listener(lambda ev, sid=step_id, n=name: self.nb_logger.info(f"🟡 Step input ready: {sid}.{n}"))
                # Track output updates
                if hasattr(step, 'step_output_data_units'):
                    for name, du in step.step_output_data_units.items():
                        if hasattr(du, 'register_change_listener'):
                            du.register_change_listener(lambda ev, sid=step_id, n=name: self.nb_logger.info(f"🟢 Step output produced: {sid}.{n}"))
        except Exception as e:
            if hasattr(self, 'nb_logger') and self.nb_logger:
                self.nb_logger.warning(f"⚠️ Failed to register step tracking: {e}")
