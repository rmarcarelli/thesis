# Aspects of Charged Lepton Flavor Violation in High-Energy Physics

This repository contains the computational framework, analysis code, and complete LaTeX document for my PhD thesis on aspects of charged lepton flavor violation in high-energy physics.

## Overview

This thesis explores lepton flavor violating processes and their implications for new physics beyond the Standard Model. The work includes:

- **LFV Observables**: Calculations of radiative decays, trilepton decays, and dipole moments
- **Lepton-Nucleus Collisions**: Cross section calculations for the beam-dump experiment SLAC E137, the upcoming Electron Ion Collider (EIC), a hypothetical Muon (Synchrotron)-Ion Collider (MuSIC), and a hypothetical Muon Beam Dump (MuBeD) experiment
- **Higgs Decay Searches**: Searches for LFV ALPs in Higgs decays from CMS, ATLAS, and MATHUSLA
- **Form Factor Calculations**: Exact and approximate form factors for LFV processes

## Repository Structure

```
thesis/
├── src/                          # LaTeX source files
│   ├── main.tex                  # Main thesis document
│   ├── macros.tex                # Custom LaTeX macros
│   ├── chapters/                 # Chapter files
│   │   ├── chapter1.tex
│   │   ├── chapter2.tex
│   │   ├── chapter3.tex
│   │   ├── chapter4.tex
│   │   ├── chapter5.tex
│   │   ├── chapter6.tex
│   │   └── chapter7.tex
│   ├── appendices/               # Appendix files
│   │   ├── appendixA.tex
│   │   └── appendixB.tex
│   ├── styles/                   # LaTeX style files
│   │   └── thesis.cls
│   └── bibliography/             # Bibliography
│       └── refs.bib
├── figures/                      # Generated plots and figures
│   ├── chapter1/
│   ├── chapter2/
│   ├── chapter3/
│   ├── chapter4/
│   ├── chapter5/
│   └── chapter6/
├── build/                        # Build artifacts (auto-generated)
├── thesis_code/                  # Main computational framework
│   ├── phys/                     # Physics constants and utilities
│   ├── lfv_lepton_observables/   # LFV decay rates and dipole moments
│   ├── lfv_higgs_decays/        # Higgs decay signal calculations
│   ├── lepton_nucleus_collisions/ # Cross section calculations
│   └── displaced_vectors/        # Displaced vertex analyses
├── notebooks/                    # Jupyter notebooks for analysis
├── environment.yml               # Conda environment specification
├── build.sh                     # Build script for LaTeX compilation
├── thesis.pdf                   # Compiled thesis (auto-generated)
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda or Miniconda
- LaTeX distribution (TeX Live or MiKTeX)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd thesis
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate thesis
```

3. Install the thesis code as a package:
```bash
pip install -e .
```

## Usage

### Building the Thesis

To compile the LaTeX thesis:

```bash
./build.sh
```

This will:
- Compile the thesis from `src/main.tex`
- Generate `thesis.pdf` in the main directory
- Move all build artifacts to the `build/` directory

The build process includes:
1. First pdflatex pass (creates auxiliary files)
2. BibTeX (processes bibliography)
3. Second pdflatex pass (resolves references)
4. Third pdflatex pass (final formatting)

### Running Notebooks

The analysis is primarily conducted through Jupyter notebooks in the `notebooks/` directory:

```bash
jupyter lab
# or
jupyter notebook
```

Key notebooks include:
- `LFV Decay Limits.ipynb` - Lepton flavor violating decay rate limits
- `Electric and Magnetic Dipole Moment Limits.ipynb` - g-2 and EDM calculations
- `Limits on LFV ALPs and Scalars at Lepton Nucleus Colliders.ipynb` - Collider limits on LFV scalars and ALPs
- `Higgs ALP Decays.ipynb` - Higgs decay signal calculations
- `Displaced Vectors at Lepton Nucleus Colliders.ipynb` - Limits on hidden vectors from displaced signals at lepton-nucleus collision experiments

### Using the Code Directly

```python
import thesis_code as tc

# Calculate LFV decay rates
from tc.lfv_lepton_observables import radiative_decay_rate
rate = radiative_decay_rate(m=100, i=0, j=1, g=[[1]*3]*3)

# Calculate cross sections
from tc.lepton_nucleus_collisions.experiments import Experiment, FinalState
exp = Experiment.from_card('path/to/experiment.yaml')
final_state = FinalState(*final_state_params)
production_cross_section = exp.production_cross_section(final_state)
```

## Physics Content

### Lepton Flavor Violation

The code implements calculations for:
- Radiative decays: ℓᵢ → ℓⱼγ
- Trilepton decays: ℓᵢ → ℓⱼℓₖℓₗ
- Magnetic dipole moments: Δaᵢ
- Electric dipole moments: dᵢ

### Form Factors

Both exact and approximate form factors are implemented:
- Exact calculations using special functions (Li₂, C₀, etc.)
- Approximate forms valid in different mass hierarchies
- Automatic selection of appropriate approximations

### Experimental Limits

Support for multiple experiments:
- **EIC**: Electron-ion collider searches for bosons 
- **MuSIC**: Hypothetical 1 TeV muon-ion collider experiment
- **MuBeD**: Hypothetical 1 TeV muon beam dump experiment
- **CMS/ATLAS**: Existing searches for prompt and displaced new physics from Higgs decays
- **MATHUSLA**: Future surface detector for long-lived particles at CERN

## Key Features

- **Modular Design**: Clean separation of physics domains
- **Comprehensive Documentation**: Detailed docstrings for all functions
- **Vectorized Calculations**: Efficient NumPy-based computations
- **Caching System**: Automatic caching of expensive calculations
- **Experiment Framework**: Flexible experiment configuration system
- **Organized LaTeX Structure**: Clean separation of source files, figures, and build artifacts
- **Automated Build System**: One-command thesis compilation

## Dependencies

Core dependencies include:
- `numpy` - Numerical computations
- `scipy` - Special functions and optimization
- `matplotlib` - Plotting
- `h5py` - Data caching
- `mpmath` - High-precision calculations
- `yaml` - Configuration files
- LaTeX distribution - For thesis compilation

See `environment.yml` for complete dependency list.

## License

This work is part of my PhD thesis. Please cite appropriately if using any of the physics calculations or results.

## Contact

For questions about the physics or code, please contact [roman.marcarelli@colorado.edu].

---

*This repository contains the computational framework and complete LaTeX document for my PhD thesis "Aspects of Charged Lepton Flavor Violation in High-Energy Physics". The code implements theoretical calculations for various experimental searches and provides tools for analyzing the sensitivity of different experiments to new physics.*
