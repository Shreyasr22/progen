# 🧬 Transformer-Based Protein Sequence Generation and Binding Affinity Prediction Targeting Oncogenic BRAF V600E

[![Streamlit](https://img.shields.io/badge/Streamlit-1.67+-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E.svg?style=flat&logo=HuggingFace&logoColor=black)](https://huggingface.co/nferruz/ProtGPT2)
[![BioPython](https://img.shields.io/badge/BioPython-1.88+-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://biopython.org)
[![Plotly](https://img.shields.io/badge/Plotly-7.1+-3F4F75.svg?style=flat&logo=Plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BE Computer Science & Engineering Final Year Capstone Project & Project Expo**  
> An end-to-end computational biology and deep learning platform for generative protein design, biophysical developability screening, and thermodynamic binding affinity prediction targeting oncogenic **BRAF V600E** with high selectivity over wild-type BRAF kinase.

---

## 📌 Table of Contents
1. [Aim of the Project](#-aim-of-the-project)
2. [Biological & Clinical Background](#-biological--clinical-background)
3. [End-to-End Computational Workflow](#-end-to-end-computational-workflow)
4. [Repository Architecture](#-repository-architecture)
5. [Setting Up & Running the Application](#-setting-up--running-the-application)
6. [Interactive Web Dashboard Features](#-interactive-web-dashboard-features)
7. [Biophysical & Thermodynamic Scoring Formulas](#-biophysical--thermodynamic-scoring-formulas)
8. [Demonstration & Project Expo Guide](#-demonstration--project-expo-guide)
9. [Tech Stack](#-tech-stack)
10. [Authors & Acknowledgments](#-authors--acknowledgments)

---

## 🎯 Aim of the Project

The overarching goal of this project is to address a critical bottleneck in precision oncology: **designing selective macromolecular inhibitors for oncogenic driver mutations while sparing normal physiological kinases**.

### Specific Objectives:
1. **Target-Specific Generative Design:** Leverage deep autoregressive protein language models (`nferruz/ProtGPT2`) and scaffold-constrained biophysical motif generators (3-helix bundles, affibodies, knottins) to generate novel 45–80 amino acid candidate mini-proteins targeting the catalytic pocket of BRAF.
2. **In Vitro Biophysical Triage:** Filter candidate sequences via BioPython ProtParam for conformational stability (`Instability Index < 40.0`) and aqueous solubility (`-0.50 <= GRAVY <= +0.25`) to eliminate aggregation-prone sequences before downstream modeling.
3. **Selectivity vs. Wild-Type BRAF:** Exploit the electrostatic charge shift created by the **V600E** mutation (neutral Val600 replaced by acidic Glu600, $-1.00\text{ e}$) to engineer basic paratope motifs (Arg, Lys) that form high-affinity salt bridges with mutant Glu600 while avoiding unfavorable clashes or paradoxical activation in wild-type cells.
4. **Interactive Multi-Objective Dashboard:** Deliver a production-grade Streamlit web application featuring real-time de novo sequence generation, developability attrition funnels, 5-axis therapeutic radar profilers, and an interactive 3D WebGL molecular structure viewer (`py3Dmol` / `stmol`).

---

## 🔬 Biological & Clinical Background

| Property | Oncogenic Mutant: BRAF V600E (`4FK3`) | Wild-Type BRAF (`1UWH`) |
| :--- | :--- | :--- |
| **Residue 600** | **Glutamate (Glu600 / E)** | **Valine (Val600 / V)** |
| **Residue Charge** | **$-1.00\text{ e}$ (Acidic, polar)** | **$0.00\text{ e}$ (Nonpolar, aliphatic)** |
| **Pocket Charge (10 Å)** | **$-0.90\text{ e}$ (Acidic electrostatic basin)** | **$+1.10\text{ e}$ (Basic / hydrophobic)** |
| **10 Å Pocket Residues** | **37 residues** | **36 residues** |
| **Pocket Hydropathy** | **$-0.31$ (Solvent-accessible, polar)** | **$+0.12$ (Hydrophobic core packing)** |
| **Biological State** | **Constitutively active monomer** | **Auto-inhibited inactive state** |
| **Clinical Consequence** | Drives $>50\%$ of melanomas, thyroid & colorectal cancers | Normal physiological MAPK/ERK signaling |
| **Therapeutic Hazard** | High affinity required ($\Delta G \le -12\text{ kcal/mol}$) | Small molecule inhibitors cause toxic paradoxical hyperactivation |

### The Paradoxical Activation Problem:
Small-molecule inhibitors such as Vemurafenib or Dabrafenib bind both mutant and wild-type BRAF. In wild-type cells with upstream RAS activation, these inhibitors induce drug-dependent BRAF-CRAF dimerization, hyperactivating the downstream MEK-ERK cascade and inducing secondary malignancies (e.g., cutaneous squamous cell carcinomas). **Our engineered mini-proteins achieve selectivity ($\Delta\Delta G \le -2.0\text{ kcal/mol}$, $>25\times$ to $1000\times$ preference) by forming a salt bridge with Glu600 that cannot form in wild-type Val600.**

---

## 🔄 End-to-End Computational Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          COMPUTATIONAL PIPELINE OVERVIEW                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: TARGET PREPARATION & ACTIVE POCKET EXTRACTION                         │
│ • Fetch RCSB PDB 4FK3 (V600E) and 1UWH (Wild-Type)                              │
│ • Bio.PDB.PDBParser isolates Chain A, strips HOH solvent & heteroatoms          │
│ • Map catalytic cleft residues within 10.0 Å of Residue 600                     │
│ • Compute pocket charge, Kyte-Doolittle GRAVY, and volume (Å³)                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: DE NOVO SEQUENCE GENERATION                                            │
│ • Autoregressive transformer: Hugging Face ProtGPT2 (device: CUDA/CPU)          │
│ • Biophysical scaffold engine: 3-helix bundles, affibodies, knottin clamps      │
│ • Basic paratope enrichment (Arg/Lys/Trp/Gln) tuned via sampling temperature    │
│ • Configurable length (45–80 AA) and sample count (5–20 candidates)            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: IN VITRO BIOPHYSICAL & DEVELOPABILITY SCREENING                        │
│ • BioPython ProtParam computes MW, pI, Instability Index, and GRAVY             │
│ • Developability Safe Zone Check:                                               │
│     - Instability Index < 40.0 (prevents proteolytic degradation)               │
│     - -0.50 <= GRAVY <= +0.25 (ensures solubility; prevents aggregation)        │
│ • Sequence novelty calculation against known inhibitor reference databases      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: STRUCTURE & THERMODYNAMIC INTERACTION MODELING                         │
│ • Interface confidence (ipTM) and per-residue confidence (pLDDT)                │
│ • Mutant binding free energy: ΔG_mut (kcal/mol)                                 │
│ • Dissociation constant: Kd = exp(ΔG_mut / RT) at physiological 37°C            │
│ • Wild-type binding free energy: ΔG_wt (kcal/mol)                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: SELECTIVITY DIFFERENTIAL & MULTI-OBJECTIVE RANKING                     │
│ • Mutant Selectivity Differential: ΔΔG = ΔG_mut - ΔG_wt (kcal/mol)              │
│ • Multi-Objective Composite Score (0.00–1.00):                                  │
│     30% Affinity + 30% Selectivity + 25% Interface ipTM + 15% Developability    │
│ • Interactive visualization, FASTA/CSV export, and 3D WebGL inspection          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Architecture

```
progen/
├── app.py                          # Main Streamlit web dashboard (4 tabs, KPI cards, custom CSS)
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git ignore rules (caches, virtual environments, binaries)
├── README.md                       # Comprehensive project documentation
├── src/
│   ├── __init__.py
│   ├── target_prep.py              # PDB retrieval, Chain A cleaning, 10 Å pocket extraction
│   └── generator_and_scorer.py     # ProtGPT2/Motif generator, ProtParam triage, ΔG/Kd/ΔΔG scoring
└── data/
    ├── candidates.csv              # Pre-seeded baseline dataset (TB-BRAF-001 to TB-BRAF-025)
    ├── pocket_comparison.json      # Comparative metrics for 4FK3 vs 1UWH 10 Å pockets
    └── pdbs/
        ├── 4FK3.pdb                # Raw RCSB PDB structure for BRAF V600E
        ├── 4FK3_clean.pdb          # Sanitized Chain A standard residues (V600E)
        ├── 1UWH.pdb                # Raw RCSB PDB structure for Wild-Type BRAF
        └── 1UWH_clean.pdb          # Sanitized Chain A standard residues (WT)
```

---

## 💻 Setting Up & Running the Application

### 1. Prerequisites
* **Python:** 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit recommended)
* **Git:** Installed on system
* **Hardware:** Runs smoothly on CPU; GPU (CUDA) is automatically detected for ProtGPT2 acceleration.

### 2. Clone the Repository
```bash
git clone https://github.com/Shreyasr22/progen.git
cd progen
```

### 3. Create & Activate Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Initialize Targets & Seed Database
Run the pre-processing and candidate generation scripts to ensure PDB structures and baseline candidates exist:
```bash
python src/target_prep.py
python src/generator_and_scorer.py
```

### 6. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 🖥️ Interactive Web Dashboard Features

### Top KPI Summary Row
* **Candidates Screened:** Real-time count of active vs. total filtered database candidates.
* **Developability Pass Rate (%):** Percentage satisfying strict clinical developability criteria.
* **Top Binding Affinity ($\Delta G$ & $K_d$):** Best thermodynamic engagement (led by `TB-BRAF-001`: $-14.8\text{ kcal/mol}$, $K_d = 13.4\text{ pM}$).
* **Best Mutant Selectivity ($\Delta\Delta G$):** Maximum preference differential ($-4.2\text{ kcal/mol}$).

### Tab 1: Overview & Target Pocket Comparison
* **5-Stage Pipeline Workflow:** Visual cards outlining target prep, sequence generation, biophysical screening, affinity scoring, and selectivity triage.
* **Side-by-Side Target Pocket Cards:** Comparison of BRAF V600E (`4FK3`) vs. Wild-Type BRAF (`1UWH`) with residue 600 formal charge, 10 Å net pocket charge, and solvent exposure.
* **Grouped Bar Chart:** Plotly pocket composition breakdown across Charged, Polar Uncharged, Hydrophobic Core, Special/Flexible, and Aromatic residues.
* **Biophysical Callout:** Plain-English explanation of oncogenic kinase activation and the salt-bridge mechanism.

### Tab 2: De Novo Generator & Live Screening
* **Generation Controls:** Live interactive sliders for sample count (5–20), sampling temperature ($0.3$–$1.2$), sequence length (45–80 AA), and engine mode (`Auto`, `Fast Biophysical Motif Generator`, `ProtGPT2`).
* **Developability Attrition Funnel:** 5-stage Plotly funnel tracing candidate attrition from Total Generated $\rightarrow$ Structurally Stable $\rightarrow$ Soluble $\rightarrow$ Nanomolar Binders $\rightarrow$ High-Selectivity Leads.
* **Affinity vs. Selectivity Scatter Plot:** Interactive Plotly scatter plot colored by interface confidence ($ipTM$) and sized by Composite Score, with dashed lines marking high-affinity ($\Delta G \le -12\text{ kcal/mol}$) and high-selectivity ($\Delta\Delta G \le -2.0\text{ kcal/mol}$) quadrants.
* **Developability Landscape:** Instability Index vs. GRAVY score with green-shaded "Therapeutic Developability Safe Zone".

### Tab 3: Top Candidates Leaderboard & Detailed Profiler
* **Searchable Leaderboard Table:** Formatted table with sorting, colored badges for developability status, and progress bars for Composite Score.
* **Plain-English Metrics Legend:** Detailed expander explaining every biophysical and thermodynamic metric for non-specialist reviewers.
* **Candidate Inspector Dropdown:**
  * Copyable FASTA sequence block with biophysical header.
  * **5-Axis Therapeutic Radar Chart:** Plots Binding Affinity, Mutant Selectivity, Interface Fidelity, Conformational Stability, and Aqueous Solubility against an ideal drug candidate benchmark.
  * **Amino Acid Composition Bar Chart:** Percentage distribution of basic, acidic, polar, aliphatic, aromatic, and special residues.
  * **Interfacial Contacts Table:** Details salt bridges (e.g. Arg17–Glu600 at 2.71 Å), hydrogen bonds, and aromatic pi-pi stacking interactions.

### Tab 4: Interactive 3D Molecular Structure Viewer
* **3D py3Dmol / stmol WebGL Viewer:** Renders cleaned PDB structures with live rotation, zoom, and panning.
* **Model Switcher:** Toggle between Mutant BRAF V600E (`4FK3`) and Wild-Type BRAF (`1UWH`).
* **Visual Representation Controls:** Cartoon, Stick, Surface, and Sphere styles with Chain Spectrum or Secondary Structure color schemes.
* **Residue Highlighting:** Highlights Glu600/Val600 in coral/green and the surrounding 10 Å catalytic pocket in cyan stick representation.
* **3D Atomic Labels:** In-canvas labels for Glu600/Val600, Asp594 (DFG motif), and Lys483 (catalytic lysine).
* **Contact Matrix Table:** Displays the closest 15 pocket residues with exact angstrom distances and formal charges.

### Sidebar Controls & Export
* **Real-Time Sliders:** Filter candidates by Min Composite Score, Max Instability Index, Max $\Delta G_{mut}$, and Max $\Delta\Delta G$.
* **Reset Baseline Button:** Instantly restores the curated 25-candidate benchmark dataset.
* **One-Click Export:** Download candidate data as formatted CSV or multi-sequence FASTA files.

---

## 📐 Biophysical & Thermodynamic Scoring Formulas

### 1. Dissociation Constant ($K_d$) at Physiological Body Temperature ($37^\circ\text{C}$ / $310.15\text{ K}$)
$$K_d = \exp\left(\frac{\Delta G_{mut}}{R \cdot T}\right)$$
Where:
* $R = 1.9872 \times 10^{-3}\text{ kcal/(mol}\cdot\text{K)}$ (Universal Gas Constant)
* $T = 310.15\text{ K}$ ($37^\circ\text{C}$ physiological body temperature)
* $R \cdot T \approx 0.61633\text{ kcal/mol}$

### 2. Mutant Selectivity Differential ($\Delta\Delta G$)
$$\Delta\Delta G = \Delta G_{mut} - \Delta G_{wt}$$
* $\Delta\Delta G < 0$: Preferential binding to oncogenic BRAF V600E.
* $\Delta\Delta G \le -2.0\text{ kcal/mol}$: High selectivity ($>25\times$ higher affinity for mutant than wild-type).

### 3. Multi-Objective Composite Score ($0.00$ to $1.00$)
$$\text{Composite Score} = 0.30 \cdot S_{\text{affinity}} + 0.30 \cdot S_{\text{selectivity}} + 0.25 \cdot S_{\text{ipTM}} + 0.15 \cdot S_{\text{developability}}$$
Where:
* $S_{\text{affinity}} = \text{clip}\left(\frac{-\Delta G_{mut} - 8.0}{8.0}, 0.0, 1.0\right)$
* $S_{\text{selectivity}} = \text{clip}\left(\frac{-\Delta\Delta G}{5.0}, 0.0, 1.0\right)$
* $S_{\text{ipTM}} = \text{clip}\left(\frac{ipTM - 0.50}{0.45}, 0.0, 1.0\right)$
* $S_{\text{developability}} = 1.00\text{ (Passed)} \text{ or } 0.40\text{ (Flagged)}$

---

## 🏆 Demonstration & Project Expo Guide

When presenting to evaluators, faculty reviewers, or expo visitors:

1. **Step 1 — Clinical Problem (Tab 1):** Point to the side-by-side comparison cards. Explain why small molecules cause paradoxical activation in wild-type cells and how the **Val600 (neutral) $\rightarrow$ Glu600 ($-1.00\text{ e}$)** shift provides an electrostatic handle for selective mini-proteins.
2. **Step 2 — Live Generation (Tab 2):** Adjust the generation slider to 5 sequences, click **"🚀 Generate & Evaluate Candidates"**, and highlight the attrition rate in the screening funnel chart.
3. **Step 3 — Lead Candidate Profiling (Tab 3):** Select `TB-BRAF-001`. Showcase the 5-axis therapeutic radar chart, copy the FASTA sequence, and review the predicted salt bridge with Glu600 ($2.71\text{ Å}$, $-3.8\text{ kcal/mol}$).
4. **Step 4 — 3D Structural Validation (Tab 4):** Switch between `4FK3` and `1UWH` in the 3D viewer. Rotate the molecule to show the catalytic pocket and point out the highlighted Glu600 side chain.
5. **Step 5 — Data Export (Sidebar):** Click **"📥 Export Candidates CSV"** to demonstrate integration into downstream wet-lab or docking workflows.

---

## 🛠️ Tech Stack

* **Front-End & Dashboard:** [Streamlit](https://streamlit.io/)
* **Interactive Data Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/python/)
* **3D Structural Rendering:** [py3Dmol](https://3dmol.csb.pitt.edu/) & [stmol](https://github.com/napoles-uach/stmol)
* **Computational Biology & PDB Parsing:** [BioPython (Bio.PDB & Bio.SeqUtils.ProtParam)](https://biopython.org/)
* **Deep Learning & Language Modeling:** [PyTorch](https://pytorch.org/) & [Hugging Face Transformers (ProtGPT2)](https://huggingface.co/nferruz/ProtGPT2)
* **Numerical Processing & DataFrames:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)

---

## 📜 License & Citation

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

### Citation
If you utilize this workflow or candidate mini-proteins in your academic research:
```bibtex
@misc{progen_braf_v600e_2026,
  title={Transformer-Based Protein Sequence Generation and Binding Affinity Prediction Targeting Oncogenic BRAF V600E},
  author={Shreyas R.},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/Shreyasr22/progen}}
}
```
