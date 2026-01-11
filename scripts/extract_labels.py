"""
Extract Node-Level Timing Labels

This script runs OpenSTA to extract slack values for every endpoint (registers and outputs)
in the design. It generates CSV files containing node-level labels (slack + violation status)
used for training the GNN.

Usage:
    python scripts/extract_labels.py --data_dir data/raw/timing_predict_data
"""

import argparse
import subprocess
import re
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional, List, Tuple

# ==============================================================================
# CONFIGURATION
# ==============================================================================

TIMEOUT_SMALL = 60    # Seconds (< 2MB)
TIMEOUT_MEDIUM = 300  # Seconds (< 10MB)
TIMEOUT_LARGE = 600   # Seconds (> 10MB)

# ==============================================================================
# CORE LOGIC
# ==============================================================================

def get_timeout(file_size_mb: float) -> int:
    """Determine timeout based on file size."""
    if file_size_mb < 2:
        return TIMEOUT_SMALL
    elif file_size_mb < 10:
        return TIMEOUT_MEDIUM
    return TIMEOUT_LARGE


def generate_tcl_script(design_dir: Path, verilog: Path, sdc: Path, spef: Optional[Path]) -> Path:
    """Generate OpenSTA TCL script for endpoint slack extraction."""
    design_name = design_dir.name
    tcl_path = design_dir / "node_extraction.tcl"
    
    # Base configuration
    tcl_content = [
        "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
        "read_liberty -min ../techlib/sky130_fd_sc_hd__ff_n40C_1v95.lib",
        "read_liberty -max ../techlib/sky130_fd_sc_hd__ss_100C_1v60.lib",
        f"read_verilog {verilog.name}",
        f"link_design {design_name}"
    ]
    
    if spef:
        tcl_content.append(f"read_spef {spef.name}")
        
    tcl_content.append(f"read_sdc {sdc.name}")
    
    # Extraction logic
    tcl_content.append("""
# Report slack for all endpoints
puts "ENDPOINT_SLACK_START"

# Get all register data pins
set registers [all_registers -data_pins]
foreach reg $registers {
    set pin_name [get_full_name $reg]
    puts "ENDPOINT: $pin_name"
    report_checks -to $reg -path_delay max
}

# Get all output ports
set outputs [all_outputs]
foreach out $outputs {
    set port_name [get_full_name $out]
    puts "OUTPUT: $port_name"
    report_checks -to $out -path_delay max
}

puts "ENDPOINT_SLACK_END"
exit
""")
    
    tcl_path.write_text("\n".join(tcl_content))
    return tcl_path


def parse_sta_output(output: str) -> pd.DataFrame:
    """Parse OpenSTA text output into a DataFrame."""
    endpoints = []
    slacks = []
    
    current_endpoint = None
    in_section = False
    
    # Regex for slack lines: "  -0.50  slack (VIOLATED)" or "   1.20  slack (MET)"
    slack_pattern = re.compile(r'^\s*([-\d.eE]+)\s+slack\s+\([A-Z]+\)')
    endpoint_pattern = re.compile(r'^(ENDPOINT|OUTPUT):\s+(\S+)')

    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if 'ENDPOINT_SLACK_START' in line:
            in_section = True
            continue
        if 'ENDPOINT_SLACK_END' in line:
            break
        
        if not in_section:
            continue

        # Check for new endpoint
        ep_match = endpoint_pattern.search(line)
        if ep_match:
            current_endpoint = ep_match.group(2)
            continue
        
        # Check for slack value
        if current_endpoint:
            if "slack (" in line:
                match = slack_pattern.search(line)
                if match:
                    try:
                        slack = float(match.group(1))
                        endpoints.append(current_endpoint)
                        slacks.append(slack)
                        current_endpoint = None  # Reset
                    except ValueError:
                        pass
            elif "No paths found" in line:
                current_endpoint = None

    return pd.DataFrame({
        'endpoint': endpoints,
        'slack': slacks,
        'label': [1 if s < 0 else 0 for s in slacks]
    })


def process_design(design_dir: Path) -> Optional[pd.DataFrame]:
    """Run extraction for a single design."""
    design_name = design_dir.name
    
    # Find files
    verilog = list(design_dir.glob("*.synthesis_preroute.v"))
    sdc = list(design_dir.glob("*.sdc"))
    spef = list(design_dir.glob("*.spef"))
    
    if not verilog or not sdc:
        logger.warning(f"[{design_name}] Missing Verilog or SDC files")
        return None
        
    verilog_file = verilog[0]
    sdc_file = sdc[0]
    spef_file = spef[0] if spef else None
    
    # Setup
    tcl_file = generate_tcl_script(design_dir, verilog_file, sdc_file, spef_file)
    output_file = design_dir / "node_slack_output.txt"
    
    # Timeout calculation
    size_mb = verilog_file.stat().st_size / (1024 * 1024)
    timeout = get_timeout(size_mb)
    
    has_spef = "with SPEF" if spef_file else "NO SPEF"
    logger.info(f"[{design_name}] Extracting... ({size_mb:.1f}MB, {has_spef})")
    
    try:
        # Run OpenSTA
        cmd = f"sta {tcl_file.name} < /dev/null > {output_file.name} 2>&1"
        subprocess.run(cmd, cwd=design_dir, shell=True, timeout=timeout, check=True)
        
        # Parse
        output = output_file.read_text()
        df = parse_sta_output(output)
        
        violation_count = (df['label'] == 1).sum()
        total = len(df)
        rate = (violation_count / total * 100) if total > 0 else 0
        
        logger.success(f"[{design_name}] ✓ {total} endpoints, {violation_count} violations ({rate:.1f}%)")
        return df
        
    except subprocess.TimeoutExpired:
        logger.error(f"[{design_name}] ✗ Timeout after {timeout}s")
        return None
    except subprocess.CalledProcessError:
        logger.error(f"[{design_name}] ✗ OpenSTA failed")
        return None
    except Exception as e:
        logger.error(f"[{design_name}] ✗ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract node-level timing labels")
    parser.add_argument("--data_dir", type=str, default="data/raw/timing_predict_data",
                        help="Directory containing design subdirectories")
    parser.add_argument("--output_dir", type=str, default="data/labels/node_level",
                        help="Directory to save label CSVs")
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    
    data_root = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("NODE-LEVEL LABEL EXTRACTION")
    logger.info("="*60)
    
    if not data_root.exists():
        logger.error(f"Data directory not found: {data_root}")
        return

    # Get designs sorted by size
    designs = []
    for d in data_root.iterdir():
        if d.is_dir() and d.name != "techlib":
            v = list(d.glob("*.synthesis_preroute.v"))
            if v:
                designs.append((d, v[0].stat().st_size))
    
    designs.sort(key=lambda x: x[1])
    logger.info(f"Found {len(designs)} designs")
    
    summary = []
    
    for design_dir, _ in designs:
        df = process_design(design_dir)
        
        if df is not None:
            # Save CSV
            save_path = output_dir / f"{design_dir.name}_node_labels.csv"
            df.to_csv(save_path, index=False)
            
            summary.append({
                'design': design_dir.name,
                'status': 'SUCCESS',
                'endpoints': len(df),
                'violations': (df['label'] == 1).sum(),
                'rate': (df['label'] == 1).mean()
            })
        else:
            summary.append({
                'design': design_dir.name,
                'status': 'FAILED',
                'endpoints': 0,
                'violations': 0,
                'rate': 0
            })
            
    # Save Summary
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / "extraction_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    logger.info("\n" + "="*60)
    logger.info(f"Extraction Complete. Summary saved to {summary_path}")
    
    # Print stats
    total_eps = summary_df['endpoints'].sum()
    total_viols = summary_df['violations'].sum()
    logger.info(f"Total Endpoints: {total_eps:,}")
    logger.info(f"Total Violations: {total_viols:,} ({total_viols/total_eps*100:.2f}%)")


if __name__ == "__main__":
    main()
