"""
Prepare Colab Training Package

Creates a zip file containing all necessary components for Google Colab training:
- Processed dataset
- Source code
- Configuration files
"""

import shutil
from pathlib import Path
from loguru import logger
import sys


def prepare_colab_pack(output_name: str = "gnn_timing_pack") -> Path:
    """
    Create a zip file for Colab training.
    
    Args:
        output_name: Name of the output zip file (without .zip extension)
    
    Returns:
        Path to created zip file
    """
    root_dir = Path(".")
    temp_dir = root_dir / "temp_colab_pack"
    
    try:
        # Clean up any existing temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()
        
        logger.info("Collecting files for Colab package...")
        
        # 1. Copy Dataset
        src_data = root_dir / "data/processed/timing_predict"
        if not src_data.exists():
            logger.error(f"Dataset not found at {src_data}")
            raise FileNotFoundError(f"Dataset directory missing: {src_data}")
        
        dst_data = temp_dir / "data/processed/timing_predict"
        shutil.copytree(src_data, dst_data)
        logger.info(f"✓ Copied dataset ({sum(1 for _ in dst_data.rglob('*'))} files)")
        
        # 2. Copy Source Code
        src_code = root_dir / "src"
        dst_code = temp_dir / "src"
        shutil.copytree(src_code, dst_code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        logger.info(f"✓ Copied source code")
        
        # 3. Copy Configs
        src_config = root_dir / "experiments/configs"
        dst_config = temp_dir / "experiments/configs"
        shutil.copytree(src_config, dst_config)
        logger.info(f"✓ Copied configs")
        
        # 4. Create zip
        logger.info("Creating zip file...")
        output_path = shutil.make_archive(output_name, 'zip', temp_dir)
        
        logger.success(f"✅ Created {output_path}")
        logger.info("Upload this file to Google Drive to start Colab training!")
        
        return Path(output_path)
        
    except Exception as e:
        logger.error(f"Failed to create Colab pack: {e}")
        raise
        
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    
    prepare_colab_pack()
