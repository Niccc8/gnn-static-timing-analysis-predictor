"""
Simple Verilog Parser (Fallback)

A lightweight regex-based parser for basic Verilog netlists.
Works reliably on Windows without external dependencies.
"""

import re
from typing import Dict, List, Tuple, Set
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
        
        self.gates: Dict[str, Tuple[str, List[str], str]] = {}
        self.nets: Dict[str, Tuple[str, List[str]]] = {}
        self.primary_inputs: Set[str] = set()
        self.primary_outputs: Set[str] = set()
        self.top_module: str = ""
        
        logger.info(f"Initialized simple parser for {self.verilog_file.name}")
    
    def parse(self) -> bool:
        """Parse the Verilog file."""
        try:
            # Read file
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
            # Pattern: gate_type instance_name ( .PIN(NET), ... );
            # We use DOTALL to handle multi-line instantiations
            gate_pattern = r'(\w+)\s+(\w+)\s*\((.*?)\)\s*;'
            gate_matches = re.findall(gate_pattern, content, re.DOTALL)
            
            # Heuristic for pin directions (SkyWater 130nm & generic)
            # Common output pin names
            output_pins = {'Y', 'X', 'Q', 'QN', 'CO', 'S', 'CONB', 'HI', 'LO', 'ZN', 'Z'}
            # Power pins to ignore
            power_pins = {'VPWR', 'VGND', 'VPB', 'VNB', 'VDD', 'VSS'}
            
            for gate_type, inst_name, connections in gate_matches:
                # Skip module and wire declarations
                if gate_type in ['module', 'wire', 'input', 'output', 'assign']:
                    continue
                
                # Parse named connections: .PIN(NET)
                # Regex to find all .PIN(NET) pairs
                conn_matches = re.findall(r'\.(\w+)\s*\((.*?)\)', connections)
                
                input_nets = []
                output_net = ""
                
                if conn_matches:
                    # Handle named connections
                    for pin, net in conn_matches:
                        pin = pin.strip()
                        net = net.strip()
                        
                        if pin in power_pins:
                            continue
                            
                        if pin in output_pins:
                            output_net = net
                        else:
                            input_nets.append(net)
                else:
                    # Fallback to positional (legacy support)
                    conn_list = [c.strip() for c in connections.split(',')]
                    if len(conn_list) >= 2:
                        output_net = conn_list[0]
                        input_nets = conn_list[1:]
                
                # Store gate if it has connections
                if output_net or input_nets:
                    self.gates[inst_name] = (gate_type, input_nets, output_net)
            
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
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_parser.py <verilog_file>")
        sys.exit(1)
    
    parser = SimpleVerilogParser(sys.argv[1])
    if parser.parse():
        stats = parser.get_statistics()
        print(f"\nCircuit Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nSample gates:")
        for i, (gate_name, (gate_type, inputs, output)) in enumerate(list(parser.gates.items())[:3]):
            print(f"  {gate_name}: {gate_type} - inputs={inputs}, output={output}")
    else:
        print("Failed to parse netlist")
        sys.exit(1)
