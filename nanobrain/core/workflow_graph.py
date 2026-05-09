"""
Workflow Graph Management System
===============================

Extracted from workflow.py monolith to provide dedicated graph management
functionality with debug/production error handling modes.
"""

from typing import Dict, Set, List, Optional, Tuple, Any
from .step import BaseStep
from .link import LinkBase
from .logging_system import get_logger
from .workflow_progress import handle_error


class WorkflowGraph:
    """
    Internal graph representation of workflow structure.

    Manages the graph of steps (nodes) and links (edges) within a workflow.
    Provides graph analysis capabilities including cycle detection and
    topological sorting for execution order determination.
    """

    def __init__(self):
        """Initialize empty workflow graph."""
        try:
            self.nodes: Dict[str, BaseStep] = {}  # step_id -> Step instance
            self.edges: Dict[str, Dict] = {}  # link_id -> Link info
            # step_id -> set of connected step_ids
            self.adjacency: Dict[str, Set[str]] = {}
            # step_id -> set of predecessor step_ids
            self.reverse_adjacency: Dict[str, Set[str]] = {}

            # Graph metadata
            self._is_valid = False
            self._execution_order: Optional[List[str]] = None
            self._strongly_connected_components: Optional[List[List[str]]] = None

            self.logger = get_logger("workflow.graph")
        except Exception as e:
            handle_error(e, "WorkflowGraph.__init__")
            # Fallback initialization
            self.nodes = {}
            self.edges = {}
            self.adjacency = {}
            self.reverse_adjacency = {}
            self._is_valid = False
            self._execution_order = None
            self._strongly_connected_components = None
            self.logger = get_logger("workflow.graph")

    def add_step(self, step_id: str, step: BaseStep) -> None:
        """Add a step node to the graph."""
        try:
            if step_id in self.nodes:
                raise ValueError(f"Step {step_id} already exists in workflow graph")

            self.nodes[step_id] = step
            self.adjacency[step_id] = set()
            self.reverse_adjacency[step_id] = set()

            # Invalidate cached computations
            self._invalidate_cache()

            self.logger.debug(f"Added step to workflow graph: {step_id}")
        except Exception as e:
            handle_error(e, f"WorkflowGraph.add_step for {step_id}")

    def add_link(self, link_id: str, link: LinkBase, source_id: str, target_id: str) -> None:
        """Add a link edge to the graph."""
        try:
            if link_id in self.edges:
                raise ValueError(f"Link {link_id} already exists in workflow graph")

            if source_id not in self.nodes:
                raise ValueError(f"Source step {source_id} not found in workflow graph")

            if target_id not in self.nodes:
                raise ValueError(f"Target step {target_id} not found in workflow graph")

            # Store link with source/target IDs for validation
            self.edges[link_id] = {
                'link': link,
                'source_id': source_id,
                'target_id': target_id
            }
            self.adjacency[source_id].add(target_id)
            self.reverse_adjacency[target_id].add(source_id)

            # Invalidate cached computations
            self._invalidate_cache()

            self.logger.debug(f"Added link to workflow graph: {link_id} ({source_id} -> {target_id})")
        except Exception as e:
            handle_error(e, f"WorkflowGraph.add_link for {link_id}")

    def remove_step(self, step_id: str) -> None:
        """Remove a step and all its connections from the graph."""
        try:
            if step_id not in self.nodes:
                raise ValueError(f"Step {step_id} not found in workflow graph")

            # Remove all edges involving this step
            edges_to_remove = []
            for link_id, link_info in self.edges.items():
                if link_info['source_id'] == step_id or link_info['target_id'] == step_id:
                    edges_to_remove.append(link_id)

            for link_id in edges_to_remove:
                self.remove_link(link_id)

            # Remove from adjacency lists
            for connected_id in self.adjacency[step_id]:
                self.reverse_adjacency[connected_id].discard(step_id)

            for predecessor_id in self.reverse_adjacency[step_id]:
                self.adjacency[predecessor_id].discard(step_id)

            # Remove the step
            del self.nodes[step_id]
            del self.adjacency[step_id]
            del self.reverse_adjacency[step_id]

            self._invalidate_cache()
            self.logger.debug(f"Removed step from workflow graph: {step_id}")
        except Exception as e:
            handle_error(e, f"WorkflowGraph.remove_step for {step_id}")

    def remove_link(self, link_id: str) -> None:
        """Remove a link from the graph."""
        try:
            if link_id not in self.edges:
                raise ValueError(f"Link {link_id} not found in workflow graph")

            link_info = self.edges[link_id]

            # Get source and target step IDs from stored info
            source_id = link_info['source_id']
            target_id = link_info['target_id']

            if source_id and target_id:
                self.adjacency[source_id].discard(target_id)
                self.reverse_adjacency[target_id].discard(source_id)

            del self.edges[link_id]
            self._invalidate_cache()
            self.logger.debug(f"Removed link from workflow graph: {link_id}")
        except Exception as e:
            handle_error(e, f"WorkflowGraph.remove_link for {link_id}")

    def get_step(self, step_id: str) -> Optional[BaseStep]:
        """Get a step by ID."""
        try:
            return self.nodes.get(step_id)
        except Exception as e:
            return handle_error(e, f"WorkflowGraph.get_step for {step_id}", None)

    def get_link(self, link_id: str) -> Optional[LinkBase]:
        """Get a link by ID."""
        try:
            link_info = self.edges.get(link_id)
            return link_info['link'] if link_info else None
        except Exception as e:
            return handle_error(e, f"WorkflowGraph.get_link for {link_id}", None)

    def get_step_dependencies(self, step_id: str) -> Set[str]:
        """Get all steps that must execute before the given step."""
        try:
            if step_id not in self.nodes:
                raise ValueError(f"Step {step_id} not found in workflow graph")
            return self.reverse_adjacency[step_id].copy()
        except Exception as e:
            return handle_error(e, f"WorkflowGraph.get_step_dependencies for {step_id}", set())

    def get_step_dependents(self, step_id: str) -> Set[str]:
        """Get all steps that depend on the given step."""
        try:
            if step_id not in self.nodes:
                raise ValueError(f"Step {step_id} not found in workflow graph")
            return self.adjacency[step_id].copy()
        except Exception as e:
            return handle_error(e, f"WorkflowGraph.get_step_dependents for {step_id}", set())

    def has_cycles(self) -> bool:
        """Check if the graph contains cycles using DFS."""
        try:
            color = {step_id: 0 for step_id in self.nodes}  # 0: white, 1: gray, 2: black

            def dfs(step_id: str) -> bool:
                if color[step_id] == 1:  # Back edge found - cycle detected
                    return True
                if color[step_id] == 2:  # Already processed
                    return False

                color[step_id] = 1  # Mark as being processed

                for neighbor in self.adjacency[step_id]:
                    if dfs(neighbor):
                        return True

                color[step_id] = 2  # Mark as completely processed
                return False

            for step_id in self.nodes:
                if color[step_id] == 0:
                    if dfs(step_id):
                        return True

            return False
        except Exception as e:
            return handle_error(e, "WorkflowGraph.has_cycles", False)

    # G18 Step 2 — bounded-cycle relaxation through LoopController nodes.
    # See `apecx-mcp-integration/docs/nanobrain_capability_gaps.md G18`.
    # The integrity validator allows declared back-edges that route through
    # at least one node tagged with `COMPONENT_TYPE == "loop_controller"`.
    # Detection uses the COMPONENT_TYPE attribute (already on LoopController)
    # to avoid an import dependency from core → library.

    def _node_is_loop_controller(self, step_id: str) -> bool:
        """True iff the node at ``step_id`` is a LoopController instance.

        Detection uses the ``COMPONENT_TYPE`` class attribute set on
        ``nanobrain.library.steps.LoopController`` rather than an
        isinstance check, to keep workflow_graph.py free of a
        dependency on the library.
        """
        node = self.nodes.get(step_id)
        if node is None:
            return False
        return getattr(node, "COMPONENT_TYPE", None) == "loop_controller"

    def _get_strongly_connected_components(self) -> List[List[str]]:
        """Tarjan's SCC algorithm. Returns SCCs of size >= 2 (single-node
        SCCs without self-loops are reported as singleton SCCs only if
        the node has a self-link). Used by the G18 relaxation logic.
        """
        # Tarjan's algorithm — iterative-ish but recursive on neighbor walks.
        index_counter = [0]
        stack: List[str] = []
        lowlinks: Dict[str, int] = {}
        index: Dict[str, int] = {}
        on_stack: Dict[str, bool] = {}
        result: List[List[str]] = []

        def strongconnect(v: str) -> None:
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in self.adjacency.get(v, set()):
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack.get(w, False):
                    lowlinks[v] = min(lowlinks[v], index[w])

            if lowlinks[v] == index[v]:
                component: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                # Only report multi-node SCCs (cycles) or self-loops.
                if len(component) > 1 or (
                    len(component) == 1 and component[0] in self.adjacency.get(component[0], set())
                ):
                    result.append(component)

        for v in self.nodes:
            if v not in index:
                strongconnect(v)

        return result

    def _all_cycles_pass_through_loop_controller(self) -> bool:
        """G18 Step 2 — return True iff EVERY cycle in the graph contains
        at least one LoopController node.

        Returns True when there are no cycles (vacuous truth — nothing to
        relax). When some cycles do NOT pass through a LoopController,
        returns False; the validator treats those cycles as undeclared
        and rejects the workflow.
        """
        sccs = self._get_strongly_connected_components()
        if not sccs:
            return True
        for scc in sccs:
            if not any(self._node_is_loop_controller(node) for node in scc):
                return False
        return True

    def _undeclared_cycle_nodes(self) -> List[List[str]]:
        """Return the SCCs that do NOT contain a LoopController. Used to
        give the operator a precise error message naming the offending
        nodes — exactly the nodes that must be either restructured to be
        acyclic OR routed through a LoopController."""
        return [
            scc for scc in self._get_strongly_connected_components()
            if not any(self._node_is_loop_controller(node) for node in scc)
        ]

    def get_execution_order(self) -> List[str]:
        """Get topological execution order using Kahn's algorithm."""
        try:
            if self._execution_order is not None:
                return self._execution_order.copy()

            # Kahn's algorithm for topological sorting
            in_degree = {step_id: len(self.reverse_adjacency[step_id]) for step_id in self.nodes}
            queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
            execution_order = []

            while queue:
                current = queue.pop(0)
                execution_order.append(current)

                for neighbor in self.adjacency[current]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if len(execution_order) != len(self.nodes):
                # Cycles detected - return partial order with warning
                self.logger.warning(
                    f"⚠️  Cannot determine complete execution order due to cycles. "
                    f"Returning partial order of {len(execution_order)}/{len(self.nodes)} steps. "
                    f"Dynamic execution will handle cyclic dependencies at runtime."
                )
                # Return the partial order we could determine
                execution_order.extend([step_id for step_id in self.nodes if step_id not in execution_order])

            self._execution_order = execution_order
            return execution_order.copy()
        except Exception as e:
            return handle_error(e, "WorkflowGraph.get_execution_order", [])

    def get_parallel_execution_levels(self) -> List[List[str]]:
        """Get steps grouped by execution level for parallel execution."""
        try:
            execution_order = self.get_execution_order()
            levels = []
            processed = set()

            while processed != set(self.nodes.keys()):
                current_level = []

                for step_id in execution_order:
                    if step_id in processed:
                        continue

                    # Check if all dependencies are satisfied
                    dependencies = self.get_step_dependencies(step_id)
                    if dependencies.issubset(processed):
                        current_level.append(step_id)

                if not current_level:
                    # Circular dependency detected - handle gracefully
                    remaining_steps = set(self.nodes.keys()) - processed
                    self.logger.warning(
                        f"⚠️  Circular dependency detected in parallel execution levels. "
                        f"Adding remaining {len(remaining_steps)} steps to final level: {remaining_steps}"
                    )
                    levels.append(list(remaining_steps))
                    break

                levels.append(current_level)
                processed.update(current_level)

            return levels
        except Exception as e:
            return handle_error(e, "WorkflowGraph.get_parallel_execution_levels", [])

    def validate_graph(self, allow_cycles: bool = False, require_connected: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate the workflow graph structure.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            errors = []
            warnings = []

            # Check for empty graph
            if not self.nodes:
                errors.append("Workflow graph is empty - no steps defined")

            # Check for cycles. Three layers of relaxation, in order:
            #   1. allow_cycles=True (operator opt-in escape hatch — preserved
            #      for backward compat).
            #   2. G18 Step 2 — every cycle passes through a LoopController.
            #   3. Otherwise: warn (legacy v1 behavior).
            if self.has_cycles():
                if allow_cycles:
                    self.logger.debug(
                        f"🔄 Workflow cycles detected and allowed by "
                        f"configuration. Steps: {self._get_cycles_info()}"
                    )
                elif self._all_cycles_pass_through_loop_controller():
                    # G18 Step 2: every cycle is bounded by a LoopController.
                    # Quiet success — log at debug only.
                    self.logger.debug(
                        f"🔄 Workflow cycles detected but each is bounded by "
                        f"a LoopController (G18). Steps: {self._get_cycles_info()}"
                    )
                else:
                    # At least one cycle is undeclared. Name the offending
                    # SCCs so the operator knows exactly which subgraph
                    # to fix.
                    undeclared = self._undeclared_cycle_nodes()
                    cycle_warning = (
                        "⚠️  WORKFLOW CYCLES DETECTED: This workflow contains cycles "
                        "that are NOT bounded by a LoopController.\n"
                        f"   Undeclared cycle node groups: {undeclared}\n"
                        "   To bound a cycle: route the back-edge through a step\n"
                        "   of class nanobrain.library.steps.LoopController (G18).\n"
                        "   To suppress this warning entirely (legacy escape hatch):\n"
                        "   set 'allow_cycles: true' in the workflow configuration."
                    )
                    warnings.append(cycle_warning)
                    self.logger.warning(cycle_warning)

            # Check for disconnected components if required
            if require_connected and len(self.nodes) > 1:
                if not self._is_weakly_connected():
                    errors.append("Workflow graph is not connected - contains isolated components")

            # Check for orphaned steps (no inputs or outputs)
            orphaned_steps = []
            for step_id in self.nodes:
                has_input = len(self.reverse_adjacency[step_id]) > 0
                has_output = len(self.adjacency[step_id]) > 0

                if not has_input and not has_output and len(self.nodes) > 1:
                    orphaned_steps.append(step_id)

            if orphaned_steps:
                errors.append(f"Orphaned steps found (no connections): {orphaned_steps}")

            # Validate that all links have valid source and target steps
            for link_id, link_info in self.edges.items():
                source_id = link_info['source_id']
                target_id = link_info['target_id']

                if source_id not in self.nodes:
                    errors.append(f"Link {link_id} has invalid source step: {source_id}")

                if target_id not in self.nodes:
                    errors.append(f"Link {link_id} has invalid target step: {target_id}")

                # Check for self-referencing links (illegal in workflow architecture)
                if source_id == target_id:
                    errors.append(
                        f"❌ ILLEGAL SELF-REFERENCING LINK: {link_id} connects step '{source_id}' to itself. "
                        f"Self-referencing links are prohibited in the workflow architecture as they create "
                        f"infinite trigger loops and prevent proper workflow execution.")

            # Cycles are not errors: Only fail on true structural problems
            is_valid = len(errors) == 0
            self._is_valid = is_valid

            # Log warnings separately
            if warnings:
                for warning in warnings:
                    self.logger.warning(warning)

            return is_valid, errors
        except Exception as e:
            return handle_error(e, "WorkflowGraph.validate_graph", (False, ["Graph validation failed"]))

    def _invalidate_cache(self) -> None:
        """Invalidate cached computations when graph structure changes."""
        try:
            self._execution_order = None
            self._strongly_connected_components = None
            self._is_valid = False
        except Exception as e:
            handle_error(e, "WorkflowGraph._invalidate_cache")

    def _is_weakly_connected(self) -> bool:
        """Check if the graph is weakly connected (ignoring edge direction)."""
        try:
            if not self.nodes:
                return True

            visited = set()
            start_node = next(iter(self.nodes))
            stack = [start_node]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                visited.add(current)

                # Add both successors and predecessors (treat as undirected)
                stack.extend(self.adjacency[current] - visited)
                stack.extend(self.reverse_adjacency[current] - visited)

            return len(visited) == len(self.nodes)
        except Exception as e:
            handle_error(e, "WorkflowGraph._is_weakly_connected", default_return=True)
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        try:
            return {
                "num_steps": len(self.nodes),
                "num_links": len(self.edges),
                "has_cycles": self.has_cycles(),
                "is_connected": self._is_weakly_connected(),
                "max_depth": self._calculate_max_depth(),
                "avg_branching_factor": self._calculate_avg_branching_factor()
            }
        except Exception as e:
            handle_error(e, "WorkflowGraph.get_stats", default_return={})
            return {}

    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of the graph."""
        try:
            if not self.nodes:
                return 0

            levels = self.get_parallel_execution_levels()
            return len(levels)
        except Exception:
            # Graph has cycles or other issues, return -1
            return -1

    def _calculate_avg_branching_factor(self) -> float:
        """Calculate average branching factor."""
        try:
            if not self.nodes:
                return 0.0

            total_edges = sum(len(neighbors) for neighbors in self.adjacency.values())
            return total_edges / len(self.nodes)
        except Exception as e:
            handle_error(e, "WorkflowGraph._calculate_avg_branching_factor", default_return=0.0)
            return 0.0


