from mixins import RunnableMixin
from enums import ExecutorBase
from LinkBase import LinkBase
from regulations import ConnectionStrength
from typing import List
import random


class Router(RunnableMixin):
    """
    Routes one input to multiple outputs.
    
    Biological analogy: Divergent neural pathway.
    Justification: Like how thalamic neurons broadcast sensory information to
    multiple cortical areas, routers distribute input data to multiple processing steps.
    """
    def __init__(self, executor: ExecutorBase, input_source: LinkBase, 
                 output_sinks: List[LinkBase], **kwargs):
        super().__init__(executor, **kwargs)
        self.input_source = input_source
        self.output_sinks = output_sinks
        self.fanout_reliability = 0.9  # Reliability decreases with more outputs
        self.routing_strategy = "broadcast"  # broadcast, random, weighted
        self.sink_weights = [ConnectionStrength() for _ in output_sinks]  # Connection strengths
    
    async def execute(self):
        """
        Gets data from input and transfers to outputs based on routing strategy.
        
        Biological analogy: Information distribution in divergent pathways.
        Justification: Like how some neurons broadcast information to multiple
        targets (e.g., thalamic relay neurons), the router distributes input
        data to multiple processing pathways.
        """
        # Get input data
        await self.input_source.transfer()
        data = self.input_source.output.get()
        
        results = []
        
        # Different routing strategies
        if self.routing_strategy == "broadcast":
            # Send to all outputs (like broadcasting in thalamus)
            for sink in self.output_sinks:
                sink.input.set(data)
                await sink.transfer()
                results.append(sink.output.get())
                
        elif self.routing_strategy == "random":
            # Send to random subset (like probabilistic release)
            for sink in self.output_sinks:
                if random.random() < self.fanout_reliability:
                    sink.input.set(data)
                    await sink.transfer()
                    results.append(sink.output.get())
                    
        elif self.routing_strategy == "weighted":
            # Send based on connection weights (like preferential pathways)
            for i, sink in enumerate(self.output_sinks):
                if random.random() < self.sink_weights[i].strength:
                    sink.input.set(data)
                    await sink.transfer()
                    results.append(sink.output.get())
                    # Strengthen successful pathways
                    self.sink_weights[i].increase(0.01)
                else:
                    # Weaken unused pathways
                    self.sink_weights[i].decrease(0.005)
        
        return results