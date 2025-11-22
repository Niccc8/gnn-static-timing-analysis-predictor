"""
Verilog Netlist Parser

Parses Verilog netlists to extract gate instances, connections, and topology.
Uses pyverilog library for AST-based parsing.
"""

import os
from typing import Dict, List, Tuple, Set
from pathlib import Path
from loguru import logger
import pyverilog.vparser.parser as vparser
from pyverilog.vparser.ast import *


class VerilogParser:
    """
    Parse Verilog netlists and extract circuit topology.
    
    Attributes:
        gates: Dictionary mapping gate instance names to (type, inputs, outputs)
        nets: Dictionary mapping net names to (driver, loads)
        primary_inputs: Set of primary input port names
        primary_outputs: Set of primary output port names
    """
    
    def __init__(self, verilog_file: str):
        """
        Initialize parser with a Verilog file.
        
        Args:
            verilog_file: Path to Verilog netlist file
        """
        self.verilog_file = Path(verilog_file)
        if not self.verilog_file.exists():
            raise FileNotFoundError(f"Verilog file not found: {verilog_file}")
        
        self.gates: Dict[str, Tuple[str, List[str], str]] = {}
        self.nets: Dict[str, Tuple[str, List[str]]] = {}
        self.primary_inputs: Set[str] = set()
        self.primary_outputs: Set[str] = set()
        self.top_module: str = ""
        
        logger.info(f"Initialized parser for {self.verilog_file.name}")
    
    def parse(self) -> bool:
        """
        Parse the Verilog file and extract circuit information.
        
        Returns:
            True if parsing successful, False otherwise
        """
        try:
            # Parse Verilog file
            ast, _ = vparser.parse([str(self.verilog_file)])
            
            if ast is None:
                logger.error(f"Failed to parse {self.verilog_file}")
                return False
            
            # Extract module definitions
            for item in ast.description.definitions:
                if isinstance(item, ModuleDef):
                    self._parse_module(item)
            
            # Build net connectivity
            self._build_net_connectivity()
            
            logger.info(
                f"Parsed {len(self.gates)} gates, {len(self.nets)} nets, "
                f"{len(self.primary_inputs)} inputs, {len(self.primary_outputs)} outputs"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error parsing Verilog: {e}")
            return False
    
    def _parse_module(self, module: ModuleDef):
        """Extract ports and instances from a module."""
        self.top_module = module.name
        
        # Extract primary inputs and outputs
        for port in module.portlist.ports:
            if isinstance(port, Port):
                port_name = port.first.name if hasattr(port.first, 'name') else str(port.first)
                
                # Determine direction from module items
                for item in module.items:
                    if isinstance(item, Decl):
                        for decl in item.list:
                            if hasattr(decl, 'name') and decl.name == port_name:
                                if isinstance(item, Input):
                                    self.primary_inputs.add(port_name)
                                elif isinstance(item, Output):
                                    self.primary_outputs.add(port_name)
        
        # Extract gate instances
        for item in module.items:
            if isinstance(item, InstanceList):
                self._parse_instance(item)
    
    def _parse_instance(self, instance_list: InstanceList):
        """Parse gate instance and extract connections."""
        gate_type = instance_list.module
        
        for instance in instance_list.instances:
            inst_name = instance.name
            connections = {}
            
            # Extract port connections
            for port_arg in instance.portlist:
                if isinstance(port_arg, PortArg):
                    port_name = port_arg.portname
                    net_name = self._get_identifier(port_arg.argname)
                    connections[port_name] = net_name
            
            # Assume standard cell has 'Y' or 'Z' as output, rest are inputs
            output_net = connections.get('Y') or connections.get('Z') or connections.get('O')
            input_nets = [net for port, net in connections.items() 
                         if port not in ['Y', 'Z', 'O']]
            
            self.gates[inst_name] = (gate_type, input_nets, output_net)
    
    def _get_identifier(self, node) -> str:
        """Extract identifier name from AST node."""
        if isinstance(node, Identifier):
            return node.name
        elif isinstance(node, Pointer):
            return self._get_identifier(node.var)
        else:
            return str(node)
    
    def _build_net_connectivity(self):
        """Build net-to-driver and net-to-load mappings."""
        for gate_name, (gate_type, inputs, output) in self.gates.items():
            # Output net driven by this gate
            if output:
                if output not in self.nets:
                    self.nets[output] = (gate_name, [])
                else:
                    # Update driver
                    self.nets[output] = (gate_name, self.nets[output][1])
            
            # Input nets loaded by this gate
            for input_net in inputs:
                if input_net in self.nets:
                    driver, loads = self.nets[input_net]
                    loads.append(gate_name)
                    self.nets[input_net] = (driver, loads)
                else:
                    # Net driven by primary input or previous gate
                    self.nets[input_net] = ("", [gate_name])
    
    def get_statistics(self) -> Dict[str, int]:
        """Return circuit statistics."""
        return {
            "num_gates": len(self.gates),
            "num_nets": len(self.nets),
            "num_primary_inputs": len(self.primary_inputs),
            "num_primary_outputs": len(self.primary_outputs),
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python netlist_parser.py <verilog_file>")
        sys.exit(1)
    
    parser = VerilogParser(sys.argv[1])
    if parser.parse():
        stats = parser.get_statistics()
        print(f"\nCircuit Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to parse netlist")
        sys.exit(1)
