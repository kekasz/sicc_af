# sicc_af
Structure Integrity pre-Correction Control for AlphaFold predicted proteins

This script takes protonated .pdb files (using pdb2pqr, propka) and returns errors found. The errors are divided into 3 categories:
  1. Clashes between residues
  2. Outstretched bonds between atoms
  3. Hydrogens put at invalid places

How to use:
  Run main.py as a regular Python script with argument being the path to the .pdb file.
  Exemple: $ python main.py <path>
