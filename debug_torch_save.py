
import torch
import os
from pathlib import Path

def test_save():
    output_dir = Path("data/processed/timing_predict")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "test_save.pt"
    print(f"Attempting to save to: {save_path.absolute()}")
    
    # Create dummy data
    data = torch.randn(10, 10)
    
    try:
        torch.save(data, str(save_path))
        print("SUCCESS: Saved file successfully.")
        
        # Clean up
        if save_path.exists():
            os.remove(save_path)
            print("SUCCESS: Cleaned up file.")
            
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_save()
