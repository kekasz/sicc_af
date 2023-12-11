## sicc_af
Structure Integrity Control and Correction for AlphaFold2 predicted proteins

This script takes protonated .pdb files (using [PDB2PQR](https://academic.oup.com/nar/article/35/suppl_2/W522/2920806)) taken from [AlphaFoldDB](https://academic.oup.com/nar/article/50/D1/D439/6430488?login=false) database and returns found errors as an output dictionary. The errors are divided into 3 categories:
  1. Clashes between residues
  2. Overextended bonds between atoms
  3. Hydrogens put at invalid places

The output dictionary's structure is following:
```python
 errors
   |--- CLASHING_PLACEMENT:
           |--- entries []
                   |--- [RESIDUE NUMBER]: atoms' to be removed NEF names []
   |--- BOND_OVEREXTENSION:
           |--- --//--
   |--- H_ATM_MISPLACEMENT:
           |--- --//--
```

# how to use:

Requirements:
In order to run this script on your machine, make sure you have [Python 3.10](https://www.python.org/downloads/) with [RDKit 2023.03.2](https://www.rdkit.org/docs/Install.html) library installed.

Instructions:
Clone the project and run then sicc_af.py as a regular Python script with facultative --pdb_file argument being the path to the .pdb file.

```bash
$ mkdir sicc_af
$ cd sicc_af
$ git clone https://github.com/kekasz/sicc_af
```
```bash
$ python sicc_af.py --pdb_file /path/to/pdb/file
```
