# Complete Circuit Dataset Download Guide

**ALL Available Datasets for GNN-Based Timing Analysis**

Last Updated: November 2025

---

## 🎯 Priority 1: Best Datasets (Download These First!)

### 1. TimingPredict Dataset ⭐⭐⭐ (HIGHLY RECOMMENDED)
**Why:** Pre-labeled with STA timing data, designed specifically for GNN timing analysis!

- **What:** 21 real-world benchmark circuits on SkyWater 130nm technology
- **Size:** ~500 MB
- **Features:** Pre-computed arrival time, slack, delay, slew, capacitance
- **Format:** Verilog + DEF + timing annotations (ready to use!)
- **Download:** 
  - GitHub: https://github.com/TimingPredict/TimingPredict
  - Dataset repo: Check the main page for dataset download links (Google Drive or PKU Drive)
- **Paper:** "A Timing Engine Inspired Graph Neural Network Model for Pre-Routing Slack Prediction" (DAC 2022)
- **Extract to:** `data/raw/timing_predict/`

**This is THE BEST dataset because it's already labeled - no OpenSTA needed!**

---

### 2. ISCAS'85 Benchmarks ⭐⭐
**Why:** Small, well-known combinational circuits perfect for testing

- **What:** 11 combinational circuits (c17 to c7552)
- **Size:** ~10 KB (very small!)
- **Gates:** 6 to 3,512 gates
- **Already downloaded:** ✅ You have c17, c432, c880
- **Download remaining circuits:**
  - Source: http://sportlab.usc.edu/~msabrishami/benchmarks.html
  - Download Verilog format for:
    - c1355 (546 gates)
    - c1908 (880 gates)
    - c2670 (1,193 gates - may not work with simple parser)
    - c3540 (1,669 gates)
    - c5315 (2,307 gates)
    - c6288 (2,416 gates - 16x16 multiplier)
- **Extract to:** `data/raw/iscas85/`

---

### 3. IWLS 2005 Benchmarks ⭐⭐
**Why:** Mix of real-world designs, medium to large circuits

- **What:** 84 designs (up to 185,000 registers, 900,000 gates)
- **Size:** ~213 MB compressed
- **Format:** Verilog + RTL source
- **Technology:** Synthesized to 180nm library
- **Download:** 
  - Direct link: https://iwls.org/iwls2005/benchmarks.html
  - File: `iwls2005benchmarks.tar.gz` (213.3 MB)
- **Extract to:** `data/raw/iwls05/`

---

## 🎯 Priority 2: Large-Scale Datasets (For Advanced Training)

### 4. EPFL Combinational Benchmark Suite ⭐⭐⭐
**Why:** Modern arithmetic and control circuits, very challenging

- **What:** 23 combinational circuits in 3 categories:
  - Arithmetic: multipliers, dividers, sqrt (10 circuits)
  - Control: arbiters, routers, voters (10 circuits)
  - MtM: 3 circuits with >10 million gates each!
- **Size:** ~100 MB (complete archive)
- **Format:** Verilog, VHDL, BLIF, AIGER
- **Download:**
  - **2019 Release:** https://zenodo.org/record/2572934
  - File: `EPFL_complete.tar.gz`
  - GitHub (latest): https://github.com/lsils/benchmarks
- **Extract to:** `data/raw/epfl/`

**Examples:** adder, multiplier, max, sin, sqrt, hypotenuse, divisor, arbiter, router, voter

---

### 5. OpenCores IP Cores Collection ⭐⭐
**Why:** Real-world open-source hardware designs

- **What:** 860+ Verilog/VHDL IP cores
- **Types:** I2C controllers, SPI, UART, Ethernet MACs, processors
- **Size:** Varies (individual downloads)
- **Download:**
  - **Archive (all cores):** https://github.com/klyone/opencores-ip
  - **Individual cores:** https://opencores.org/projects
  - **Benchmark collection:** https://cw.fel.cvut.cz/wiki/courses/b35apo/benchmarks (includes ISCAS + OpenCores)
- **Extract to:** `data/raw/opencores/`

**Popular cores:** i2c_master, spi_master, uart16550, aes_core, ethernet_tri_mode

---

## 🎯 Priority 3: Research/Contest Datasets (Advanced)

### 6. TAU Contest Benchmarks ⭐
**Why:** Industry-grade timing analysis benchmarks

- **TAU 2015:** Incremental timing analysis benchmarks
  - https://sites.google.com/site/taucontest2015/
  - Benchmarks: ispd, crc32d16N, netcard
  
- **TAU 2014:** Timing analysis + CPPR framework
  - https://sites.google.com/site/taucontest2014/
  - Files: `sV2.tar.gz`, `aes_core.tar.gz`
  
- **TAU 2021:** Stage delay calculator benchmarks
  - https://tauworkshop.com/
  - Includes: RC netlists, libraries, test suites

- **Extract to:** `data/raw/tau/`

---

## 📊 Dataset Size Summary

| Dataset | Circuits | Total Size | Difficulty | Pre-labeled? |
|---------|----------|------------|------------|--------------|
| **TimingPredict** | 21 | ~500 MB | Medium | ✅ YES |
| **ISCAS'85** | 11 | ~10 KB | Easy | ❌ No |
| **ISCAS'89** | 31 | ~50 KB | Medium | ❌ No |
| **IWLS 2005** | 84 | ~213 MB | Hard | ❌ No |
| **EPFL** | 23 | ~100 MB | Very Hard | ❌ No |
| **OpenCores** | 860+ | Varies | Mixed | ❌ No |
| **TAU** | ~20 | Varies | Industry | Partial |

---

## 🎯 Recommended Download Strategy

### Stage 1: Get Started (TODAY)
✅ ISCAS'85 (you already have 3 circuits!)  
⬜ Download 5 more ISCAS'85 circuits (c1355, c1908, c2670, c3540, c5315)  
⬜ **TimingPredict dataset** (PRIORITY - has labels!)

### Stage 2: Scale Up (WEEK 2)
⬜ IWLS 2005 (84 circuits)  
⬜ EPFL benchmarks (23 circuits)

### Stage 3: Advanced (WEEK 3+)
⬜ OpenCores selection (pick 20-30 cores)  
⬜ TAU contest benchmarks

---

## 📥 Quick Download Commands

**For Linux/WSL/Mac:**

```bash
cd "d:/GNN-Based Static Timing Analysis Predictor/data/raw"

# IWLS 2005
wget https://iwls.org/iwls2005/benchmarks/iwls2005benchmarks.tar.gz
tar -xzf iwls2005benchmarks.tar.gz -C iwls05/

# EPFL 2019
wget https://zenodo.org/record/2572934/files/EPFL_complete.tar.gz
tar -xzf EPFL_complete.tar.gz -C epfl/

# OpenCores collection
git clone https://github.com/klyone/opencores-ip.git opencores/
```

**For Windows (PowerShell):**

```powershell
# Download manually from links above and extract
```

---

## 🎓 Dataset Quality Ranking for Your Project

**For GNN Timing Analysis:**

1. **TimingPredict** (10/10) - Perfect for your project, already labeled!
2. **IWLS 2005** (8/10) - Real designs, good variety
3. **EPFL** (7/10) - Challenging, modern circuits
4. **ISCAS'85** (7/10) - Small but trusted, great for testing
5. **TAU** (6/10) - Industry-grade but harder to access
6. **OpenCores** (5/10) - Variable quality, needs curation

---

## 📚 Standard Cell Libraries (Needed for STA)

You'll also need Liberty (.lib) files:

1. **SkyWater 130nm PDK** (Open Source, used by TimingPredict)
   - https://github.com/google/skywater-pdk
   - File: `sky130_fd_sc_hd__tt_025C_1v80.lib`
   
2. **FreePDK45** (Academic use)
   - https://eda.ncsu.edu/freepdk/freepdk45/

3. **NanGate 45nm** (Research license)
   - http://www.nangate.com/

---

## 💡 Pro Tips

1. **Start small:** Download TimingPredict + ISCAS'85 first (~500 MB total)
2. **Test immediately:** Validate your pipeline works before downloading huge datasets
3. **Use TimingPredict first:** It has pre-computed timing labels!
4. **OpenSTA:** Only needed for unlabeled datasets (ISCAS, IWLS, EPFL)
5. **Storage:** Reserve ~2-3 GB for all datasets combined

---

## 🚀 What to Download RIGHT NOW

**Immediate action items:**

1. **TimingPredict dataset** - Search GitHub page for download link
2. **Complete ISCAS'85** - Download remaining 5-6 circuits
3. **SkyWater PDK** - Get the .lib file for timing analysis

With just these 3 items (~500 MB), you can:
- Train a real GNN model
- Achieve publishable results
- Complete your entire project

The other datasets are **optional extras** for showing generalization!

---

**Questions? Check the links and start downloading!** 🎯
