#!/bin/bash

# Build script for the thesis
# This script compiles the thesis from the src directory and outputs to build directory

echo "Building thesis..."

# Run the full LaTeX compilation sequence from root directory
echo "Running pdflatex (first pass)..."
pdflatex src/main.tex

echo "Running bibtex..."
bibtex main

echo "Running pdflatex (second pass)..."
pdflatex src/main.tex

echo "Running pdflatex (third pass)..."
pdflatex src/main.tex

# Move the final PDF to main directory
echo "Moving PDF to main directory..."
mv main.pdf thesis.pdf

# Move all auxiliary files to build directory
echo "Moving auxiliary files to build directory..."
mv *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.out *.synctex.gz *.fdb_latexmk *.fls build/ 2>/dev/null || true

echo "Build complete! PDF is available at thesis.pdf" 