"""
Simple Verilog Parser

A lightweight regex-based parser for basic Verilog netlists.
Works reliably on Windows without external dependencies.
"""

import re
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
from loguru import logger


class SimpleVerilogParser:
    """
    Simple regex-based Verilog parser.
    Handles basic gate-level netlists without complex AST parsing.
    """
    
    def __init__(self, verilog_file: str):
        """Initialize parser with a Verilog file."""
        self.verilog_file = Path(verilog_file)
        if not self.verilog_file.exists():
            raise FileNotFoundError(f"Verilog file not found: {verilog_file}")
        
        self.gates: Dict[str, Tuple[str, List[Tuple[str, str]], Optional[Tuple[str, str]]]] = {}
        self.nets: Dict[str, Tuple[str, List[str]]] = {}
        self.primary_inputs: Set[str] = set()
        self.primary_outputs: Set[str] = set()
        self.top_module: str = ""
        
        logger.debug(f"Initialized simple parser for {self.verilog_file.name}")
    
    def parse(self) -> bool:
        """Parse the Verilog file."""
        try:
            with open(self.verilog_file, 'r') as f:
                content = f.read()
            
            # Remove comments
            content = re.sub(r'//.*?\n', '\n', content)
            content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            
            # Extract module name
            module_match = re.search(r'module\s+(\w+)', content)
            if module_match:
                self.top_module = module_match.group(1)
            
            # Extract inputs
            input_matches = re.findall(r'input\s+(.*?);', content, re.DOTALL)
            for match in input_matches:
                ports = re.findall(r'\w+', match)
                self.primary_inputs.update(ports)
            
            # Extract outputs
            output_matches = re.findall(r'output\s+(.*?);', content, re.DOTALL)
            for match in output_matches:
                ports = re.findall(r'\w+', match)
                self.primary_outputs.update(ports)
            
            # Extract gate instances
            gate_pattern = r'(\w+)\s+(\w+)\s*\((.*?)\)\s*;'
            gate_matches = re.findall(gate_pattern, content, re.DOTALL)
            
            # Heuristics
            output_pins = {'Y', 'X', 'Q', 'QN', 'CO', 'S', 'CONB', 'HI', 'LO', 'ZN', 'Z'}
            power_pins = {'VPWR', 'VGND', 'VPB', 'VNB', 'VDD', 'VSS'}
            
            for gate_type, inst_name, connections in gate_matches:
                if gate_type in ['module', 'wire', 'input', 'output', 'assign']:
                    continue
                
                conn_matches = re.findall(r'\.(\w+)\s*\((.*?)\)', connections)
                
                input_pins = []
                output_pin = None
                
                if conn_matches:
                    for pin, net in conn_matches:
                        pin = pin.strip()
                        net = net.strip()
                        
                        if pin in power_pins:
                            continue
                            
                        if pin in output_pins:
                            output_pin = (pin, net)
                        else:
                            input_pins.append((pin, net))
                else:
                    # Fallback for positional args
                    conn_list = [c.strip() for c in connections.split(',')]
                    if len(conn_list) >= 2:
                        output_pin = ("OUT", conn_list[0])
                        for i, net in enumerate(conn_list[1:]):
                            input_pins.append((f"IN{i}", net))
                
                if output_pin or input_pins:
                    self.gates[inst_name] = (gate_type, input_pins, output_pin)
            
            self._build_net_connectivity()
            
            logger.info(
                f"Parsed {len(self.gates)} gates, {len(self.nets)} nets, "
                f"{len(self.primary_inputs)} inputs, {len(self.primary_outputs)} outputs"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error parsing Verilog: {e}")
            return False
    
    def _build_net_connectivity(self):
        """Build net-to-driver and net-to-load mappings."""
        for gate_name, (gate_type, input_pins, output_pin) in self.gates.items():
            # Driver
            if output_pin:
                _, output_net = output_pin
                if output_net:
                    if output_net not in self.nets:
                        self.nets[output_net] = (gate_name, [])
                    else:
                        self.nets[output_net] = (gate_name, self.nets[output_net][1])
            
            # Loads
            for _, input_net in input_pins:
                if input_net in self.nets:
                    driver, loads = self.nets[input_net]
                    loads.append(gate_name)
                    self.nets[input_net] = (driver, loads)
                else:
                    self.nets[input_net] = ("", [gate_name])
    
    def get_statistics(self) -> Dict[str, int]:
        """Return circuit statistics."""
        return {
            "num_gates": len(self.gates),
            "num_nets": len(self.nets),
            "num_primary_inputs": len(self.primary_inputs),
            "num_primary_outputs": len(self.primary_outputs),
        }
