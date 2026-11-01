# EE 513 Project 1: Characterize Your Camera
# Nikon Z50 Datasheet

**Author:** Marco Martinez Corral (`marcom@pdx.edu`)  
**Course:** EE 513

---

### Resume Coursework Entry
> **Built an open-source camera characterization toolkit in Python using OpenCV (spatial MTF50, Poisson-Gaussian photon transfer curve, dynamic range, 3x3 least-squares CCM, and Brown-Conrady distortion) and published a self-verified metrological datasheet for a Nikon Z50 camera (CMOS DX sensor, NIKKOR Z DX 16-50mm f/3.5-6.3 VR lens).**

---

### Executive Summary & Camera Identity
Manufacturers publish  sensor resolution (20.9 MP) and lens aperture limits ($f/3.5\text{--}6.3$), which do not account for physical factors that affect real world photo clarity/fidelity. 

This report presents a characterization of a Nikon Z50 mirrorless camera paired with a NIKKOR Z DX 16-50mm f/3.5-6.3 VR wide lens across five specification domains:
1. **Geometric Intrinsics & Lens Distortion** (41.0 mm zoom state, $N=20$ checkerboard views)
2. **Spatial Resolution & MTF50** (Slanted-edge ISO 12233 method across center and 4 corners)
3. **Poisson-Gaussian Noise & Photon Transfer Curve** (11 flat-field exposure pairs + 5 dark current integration times at ISO 100)
4. **Dynamic Range** (Derived under both $\text{SNR}=1$ engineering floor and $20\text{ dB}$ photographic conventions)
5. **Color Accuracy & Least-Squares CCM** (24-patch X-Rite ColorChecker Classic evaluated under CIEDE2000 across indoor illuminants and natural daylight)

## Measured Camera Datasheet Summary

| Parameter | Measured Value ± Uncertainty | Units | Method / Notes |
| :--- | :--- | :--- | :--- |
| **Sensor Resolution** | $5600 \times 3728$ | px | 14-bit RGGB (Visible array via `rawpy`) |
| **Checkerboard Pitch** | $19.9 \pm 0.1$ | mm | Physical ruler measurement (0.5% printer scale error) |
| **Focal Length** | $12263.4 \pm 60.0$ | px | 41.0 mm EXIF (close-focus effective distance $d_i = 51.7$ mm) |
| **Principal Point** | $(2761.2 \pm 32.4,\ 1971.0 \pm 28.9)$ | px | Near geometric center $(2800, 1864)$ |
| **Radial Distortion** | $k_1=0.1469,\ k_2=-4.2447$ | unitless | Slight pincushion at 41 mm telephoto |
| **Reprojection Error** | $2.4054 \pm 0.1850$ | px RMS | $N=20$ views, unmounted paper target |
| **MTF50 (Center)** | $0.2587 \pm 0.0085$ | cy/px | ISO 12233 slanted-edge (Green RAW channel) |
| **MTF50 (Corner Mean)** | $0.1977 \pm 0.0198$ | cy/px | 23.6% spatial resolution drop-off in corners |
| **Conversion Gain ($g$)** | $0.4206 \pm 0.0045$ | $\text{DN}/e^-$ | WLS on 9 linear flat pairs (~36.5k $e^-$ full-well) |
| **Read Noise** | $6.27 \pm 0.37$ | $e^-$ RMS | PTC intercept ($2.64\text{ DN}$ RMS at ISO 100) |
| **Dynamic Range** | $12.51 \pm 0.09$ | stops | Engineering floor ($\text{SNR}=1$ / $0\text{ dB}$) |
| **Color Error (Pre-CCM)** | $9.42 \pm 0.35$ | $\Delta E_{00}$ | White-balanced RAW only (Illuminant 1) |
| **Color Error (Post-CCM)**| $3.07 \pm 0.12$ | $\Delta E_{00}$ | $3\times 3$ OLS CCM; 67.4% error reduction |
| **Cross-Illuminant Shift**| $3.94$ (Worst: $9.93$) | $\Delta E_{00}$ | Transferring Illum 1 CCM to Illum 3 (Red degrades) |

## Reproduction Instructions

1. **Install Dependencies:**
   ```bash
   pip install numpy scipy matplotlib opencv-python rawpy tifffile pytest
   ```

2. **Run the Jupyter Notebook:**
   Launch Jupyter and execute all cells in `Project1_Notebook.ipynb`. The notebook will automatically look for the configured image directories (`converted_images`, `Images/flat_fields`, etc.) or fall back to cached arrays to generate the final plots and `results.json`.
   ```bash
   jupyter notebook Project1_Notebook.ipynb
   ```

3. **Run the Verification Test Suite:**
   This repository includes a standalone automated test suite to verify the mathematical soundness of the characterization algorithms (MTF50, PTC, CCM, CIEDE2000) against analytic ground truths.
   ```bash
   pytest test_characterization.py -v
   ```
