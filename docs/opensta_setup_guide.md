# OpenSTA Setup Guide for Windows (WSL2)

This guide walks you through installing OpenSTA to extract real timing labels from your circuit designs.

---

## Step 1: Install WSL2 (Windows Subsystem for Linux)

**Open PowerShell as Administrator** and run:

```powershell
wsl --install
```

This will:
- Enable WSL
- Install Ubuntu (default Linux distribution)
- Require a **restart**

**After restart:**
- Ubuntu will open automatically
- Create a username/password (remember these!)

---

## Step 2: Install OpenSTA in WSL

Open **Ubuntu** (search for it in Start menu) and run:

```bash
# Update package manager
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y git cmake build-essential swig bison flex \
    tcl-dev libreadline-dev zlib1g-dev

# Clone OpenSTA
cd ~
git clone https://github.com/The-OpenROAD-Project/OpenSTA.git
cd OpenSTA

# Build OpenSTA
mkdir build
cd build
cmake ..
make -j$(nproc)

# Install
sudo make install

# Verify installation
sta -version
```

**Expected output:** `OpenSTA 2.x.x`

**Time:** ~10-15 minutes

---

## Step 3: Access Your Windows Files from WSL

Your Windows `D:` drive is accessible at `/mnt/d/` in WSL.

```bash
# Navigate to your project
cd /mnt/d/GNN-Based\ Static\ Timing\ Analysis\ Predictor/

# Verify you can see your data
ls data/raw/timing_predict_data/
```

---

## Step 4: Run OpenSTA on One Design (Test)

```bash
# Navigate to a design
cd data/raw/timing_predict_data/cic_decimator/

# Run the existing OpenSTA script
sta cic_decimator.opensta.tcl

# This will output timing analysis
# Look for lines with "slack" values
```

---

## Step 5: Extract Labels (Python Script)

I'll create a Python script that:
1. Runs OpenSTA on all 21 designs
2. Parses timing reports
3. Extracts slack values
4. Converts to binary labels (0=safe, 1=violating)
5. Saves to CSV

**Script location:** `scripts/extract_labels_from_opensta.py`

**Run it:**
```bash
cd /mnt/d/GNN-Based\ Static\ Timing\ Analysis\ Predictor/
python scripts/extract_labels_from_opensta.py
```

---

## Troubleshooting

**Issue:** `sta: command not found`
- **Fix:** Make sure `make install` completed successfully
- Try: `sudo ldconfig`

**Issue:** Can't find Windows files in WSL
- **Fix:** They're at `/mnt/c/` (C drive) or `/mnt/d/` (D drive)

**Issue:** OpenSTA crashes
- **Fix:** Ensure `.sdc` (constraints) and `.spef` (parasitics) files exist
- Check the `.opensta.tcl` script for correct file paths

---

## Next Steps

After running the extraction script:
1. Rebuild dataset with real labels
2. Re-train on GPU
3. Achieve ROC-AUC > 0.95!

