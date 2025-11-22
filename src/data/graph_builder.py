"""
Timing DAG Builder

Constructs heterogeneous directed acyclic graphs (DAGs) from circuit netlists.
Creates two edge types: net edges (fan-out) and cell edges (gate delays).
"""

import networkx as nx
from collections import deque
from typing import Dict, List, Tuple, Set
from loguru import logger


class TimingDAGBuilder:
    """
    Build heterogeneous timing DAG from parsed netlist.
    
    Graph Structure:
        - Nodes: Pins (input/output of each gate)
        - Net Edges: Driver pin → Load pins (interconnect)
        - Cell Edges: Input pins → Output pin within gate (delay arc)
    """
    
    def __init__(self, gates: Dict, nets: Dict, primary_inputs: Set, primary_outputs: Set):
        """
        Initialize DAG builder.
        
        Args:
            gates: Dict[instance_name] = (gate_type, input_nets, output_net)
            nets: Dict[net_name] = (driver_instance, load_instances)
            primary_inputs: Set of primary input net names
            primary_outputs: Set of primary output net names
        """
        self.gates = gates
        self.nets = nets
        self.primary_inputs = primary_inputs
        self.primary_outputs = primary_outputs
        
        self.graph = nx.DiGraph()
        self.pin_to_id: Dict[str, int] = {}
        self.id_to_pin: Dict[int, str] = {}
        self.node_id = 0
        
    def build(self) -> Tuple[nx.DiGraph, Dict[str, int], Dict[int, int]]:
        """
        Build the heterogeneous DAG.
        
        Returns:
            graph: NetworkX DiGraph with node/edge attributes
            pin_to_id: Mapping from pin names to node IDs
            levels: Topological levels for each node
        """
        logger.info("Building heterogeneous timing DAG...")
        
        # Step 1: Create nodes for all pins
        self._create_nodes()
        
        # Step 2: Add net edges (interconnect)
        self._add_net_edges()
        
        # Step 3: Add cell edges (gate delays)
        self._add_cell_edges()
        
        # Step 4: Compute topological levels
        levels = self._compute_levels()
        
        logger.info(
            f"Built DAG: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges, "
            f"{max(levels.values())} levels"
        )
        
        return self.graph, self.pin_to_id, levels
    
    def _create_nodes(self):
        """Create nodes for all pins (gate inputs/outputs + primary ports)."""
        # Add primary input pins
        for pi in self.primary_inputs:
            pin_name = f"PI_{pi}"
            self.graph.add_node(
                self.node_id,
                pin_name=pin_name,
                gate="PRIMARY_INPUT",
                pin_type="output",  # Drives internal circuit
                is_endpoint=False
            )
            self.pin_to_id[pin_name] = self.node_id
            self.id_to_pin[self.node_id] = pin_name
            self.node_id += 1
        
        # Add gate pins
        for gate_name, (gate_type, input_nets, output_net) in self.gates.items():
            # Input pins
            for i, input_net in enumerate(input_nets):
                pin_name = f"{gate_name}_in_{i}"
                self.graph.add_node(
                    self.node_id,
                    pin_name=pin_name,
                    gate=gate_name,
                    gate_type=gate_type,
                    pin_type="input",
                    is_endpoint=False
                )
                self.pin_to_id[pin_name] = self.node_id
                self.id_to_pin[self.node_id] = pin_name
                self.node_id += 1
            
            # Output pin
            pin_name = f"{gate_name}_out"
            is_po = output_net in self.primary_outputs
            self.graph.add_node(
                self.node_id,
                pin_name=pin_name,
                gate=gate_name,
                gate_type=gate_type,
                pin_type="output",
                is_endpoint=is_po  # Endpoints are primary outputs
            )
            self.pin_to_id[pin_name] = self.node_id
            self.id_to_pin[self.node_id] = pin_name
            self.node_id += 1
    
    def _add_net_edges(self):
        """Add net edges from driver pins to load pins."""
        for net_name, (driver_gate, load_gates) in self.nets.items():
            # Find driver pin
            if not driver_gate:
                # Net driven by primary input
                if net_name in self.primary_inputs:
                    driver_pin = f"PI_{net_name}"
                else:
                    continue
            else:
                driver_pin = f"{driver_gate}_out"
            
            if driver_pin not in self.pin_to_id:
                continue
            
            driver_id = self.pin_to_id[driver_pin]
            
            # Add edges to all loads
            for load_gate in load_gates:
                # Find which input pin of load connects to this net
                gate_type, input_nets, _ = self.gates.get(load_gate, (None, [], None))
                for i, input_net in enumerate(input_nets):
                    if input_net == net_name:
                        load_pin = f"{load_gate}_in_{i}"
                        if load_pin in self.pin_to_id:
                            load_id = self.pin_to_id[load_pin]
                            self.graph.add_edge(
                                driver_id,
                                load_id,
                                edge_type="net",
                                net_name=net_name
                            )
    
    def _add_cell_edges(self):
        """Add cell edges from input pins to output pins within gates."""
        for gate_name, (gate_type, input_nets, output_net) in self.gates.items():
            output_pin = f"{gate_name}_out"
            if output_pin not in self.pin_to_id:
                continue
            
            output_id = self.pin_to_id[output_pin]
            
            # Add edge from each input to output
            for i in range(len(input_nets)):
                input_pin = f"{gate_name}_in_{i}"
                if input_pin in self.pin_to_id:
                    input_id = self.pin_to_id[input_pin]
                    self.graph.add_edge(
                        input_id,
                        output_id,
                        edge_type="cell",
                        gate_type=gate_type
                    )
    
    def _compute_levels(self) -> Dict[int, int]:
        """
        Compute topological levels (depth in DAG).
        
        Returns:
            Dictionary mapping node IDs to their levels
        """
        levels = {}
        in_degree = dict(self.graph.in_degree())
        
        # Start with nodes having no predecessors
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        for node in queue:
            levels[node] = 0
        
        # BFS to assign levels
        while queue:
            node = queue.popleft()
            current_level = levels[node]
            
            for successor in self.graph.successors(node):
                in_degree[successor] -= 1
                
                # Update level to max of all predecessors + 1
                if successor in levels:
                    levels[successor] = max(levels[successor], current_level + 1)
                else:
                    levels[successor] = current_level + 1
                
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Handle any nodes with cycles (shouldn't happen in proper netlists)
        for node in self.graph.nodes():
            if node not in levels:
                levels[node] = 0
                logger.warning(f"Node {self.id_to_pin[node]} has cycle or is disconnected")
        
        return levels


if __name__ == "__main__":
    # Example usage
    gates = {
        "U1": ("AND2", ["a", "b"], "n1"),
        "U2": ("INV", ["n1"], "y"),
    }
    nets = {
        "a": ("", ["U1"]),
        "b": ("", ["U1"]),
        "n1": ("U1", ["U2"]),
        "y": ("U2", []),
    }
    primary_inputs = {"a", "b"}
    primary_outputs = {"y"}
    
    builder = TimingDAGBuilder(gates, nets, primary_inputs, primary_outputs)
    graph, pin_to_id, levels = builder.build()
    
    print(f"\nDAG Statistics:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    print(f"  Max Level: {max(levels.values())}")
