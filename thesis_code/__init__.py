import os
import sys

# Get the absolute path to the thesis_code directory (this file's location)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the project root (parent of thesis_code directory)
PROJECT_ROOT = os.path.dirname(CODE_DIR)

# Add thesis_code/ to sys.path if not already present
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Set the working directory to the project root
if os.getcwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

# Import all submodules to make them available as thesis_code.submodule
try:
    import lfv_lepton_observables
    import lfv_higgs_decays
    import lepton_nucleus_collisions
    import displaced_vectors
    import phys
except ImportError as e:
    print(f"Warning: Could not import all submodules: {e}")

# Make submodules available at the top level
__all__ = ['lfv_lepton_observables', 'lfv_higgs_decays', 'lepton_nucleus_collisions', 'displaced_vectors', 'phys'] 