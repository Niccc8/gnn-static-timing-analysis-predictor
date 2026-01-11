"""
Timing DAG Builder

Constructs heterogeneous directed acyclic graphs (DAGs) from circuit netlists.
Creates two edge types:
1. Net Edges: Driver pin -> Load pins (Interconnect)
2. Cell Edges: Input pins -> Output pin within gate (Logic delay)
"""

import networkx as nx
from collections import deque
from typing import Dict, List, Tuple, Set, Optional
from loguru import logger


class TimingDAGBuilder:
    """
    Build heterogeneous timing DAG from parsed netlist.
    """
    
    def __init__(
        self, 
        gates: Dict[str, Tuple[str, List[Tuple[str, str]], Optional[Tuple[str, str]]]], 
        nets: Dict[str, Tuple[str, List[str]]], 
        primary_inputs: Set[str], 
        primary_outputs: Set[str]
    ):
        """
        Initialize DAG builder.
        
        Args:
            gates: Dict[instance_name] = (gate_type, input_pins, output_pin)
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
        # Step 1: Create nodes for all pins
        self._create_nodes()
        
        # Step 2: Add net edges (interconnect)
        self._add_net_edges()
        
        # Step 3: Add cell edges (gate delays)
        self._add_cell_edges()
        
        # Step 4: Compute topological levels
        levels = self._compute_levels()
        
        logger.debug(
            f"Built DAG: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return self.graph, self.pin_to_id, levels
    
    def _create_nodes(self):
        """Create nodes for all pins (gate inputs/outputs + primary ports)."""
        # 1. Primary Inputs
        for pi in self.primary_inputs:
            pin_name = f"PI_{pi}"
            self._add_node(pin_name, "PRIMARY_INPUT", "output", is_endpoint=False)
        
        # 2. Gate Pins
        for gate_name, (gate_type, input_pins, output_pin) in self.gates.items():
            # Input pins
            for pin, net in input_pins:
                full_pin_name = f"{gate_name}/{pin}"
                self._add_node(full_pin_name, gate_type, "input", is_endpoint=False, gate_name=gate_name)
            
            # Output pin
            if output_pin:
                pin, net = output_pin
                full_pin_name = f"{gate_name}/{pin}"
                is_po = net in self.primary_outputs
                # Note: Registers are also endpoints, but we handle them via labels usually.
                # Here we mark POs explicitly.
                self._add_node(full_pin_name, gate_type, "output", is_endpoint=is_po, gate_name=gate_name)

    def _add_node(self, pin_name: str, gate_type: str, pin_type: str, is_endpoint: bool, gate_name: str = None):
        """Helper to add a node to the graph."""
        self.graph.add_node(
            self.node_id,
            pin_name=pin_name,
            gate=gate_name if gate_name else gate_type,
            gate_type=gate_type,
            pin_type=pin_type,
            is_endpoint=is_endpoint
        )
        self.pin_to_id[pin_name] = self.node_id
        self.id_to_pin[self.node_id] = pin_name
        self.node_id += 1
    
    def _add_net_edges(self):
        """Add net edges from driver pins to load pins."""
        for net_name, (driver_gate, load_gates) in self.nets.items():
            # Determine driver pin
            driver_pin = self._get_driver_pin(net_name, driver_gate)
            if not driver_pin or driver_pin not in self.pin_to_id:
                continue
            
            driver_id = self.pin_to_id[driver_pin]
            
            # Connect to all load gates
            for load_gate in load_gates:
                self._connect_load_gate(driver_id, load_gate, net_name)

    def _get_driver_pin(self, net_name: str, driver_gate: str) -> Optional[str]:
        """Get the pin name driving the net."""
        if not driver_gate:
            # Driven by Primary Input
            if net_name in self.primary_inputs:
                return f"PI_{net_name}"
            return None
        
        # Driven by Gate Output
        gate_info = self.gates.get(driver_gate)
        if gate_info and gate_info[2]:  # has output pin
            pin, _ = gate_info[2]
            return f"{driver_gate}/{pin}"
        return None

    def _connect_load_gate(self, driver_id: int, load_gate: str, net_name: str):
        """Connect driver to all input pins of load_gate connected to net_name."""
        gate_info = self.gates.get(load_gate)
        if not gate_info:
            return
            
        _, input_pins, _ = gate_info
        for pin, net in input_pins:
            if net == net_name:
                load_pin = f"{load_gate}/{pin}"
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
        for gate_name, (gate_type, input_pins, output_pin) in self.gates.items():
            if not output_pin:
                continue
                
            pin, _ = output_pin
            output_pin_name = f"{gate_name}/{pin}"
            
            if output_pin_name not in self.pin_to_id:
                continue
            
            output_id = self.pin_to_id[output_pin_name]
            
            # Connect all inputs to output
            for pin, _ in input_pins:
                input_pin_name = f"{gate_name}/{pin}"
                if input_pin_name in self.pin_to_id:
                    input_id = self.pin_to_id[input_pin_name]
                    self.graph.add_edge(
                        input_id,
                        output_id,
                        edge_type="cell",
                        gate_type=gate_type
                    )
    
    def _compute_levels(self) -> Dict[int, int]:
        """Compute topological levels (depth in DAG)."""
        levels = {}
        in_degree = dict(self.graph.in_degree())
        
        # Start with nodes having no predecessors
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        for node in queue:
            levels[node] = 0
        
        # BFS
        while queue:
            node = queue.popleft()
            current_level = levels[node]
            
            for successor in self.graph.successors(node):
                in_degree[successor] -= 1
                
                # Level = max(predecessors) + 1
                levels[successor] = max(levels.get(successor, 0), current_level + 1)
                
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Handle disconnected/cyclic nodes
        for node in self.graph.nodes():
            if node not in levels:
                levels[node] = 0
        
        return levels
