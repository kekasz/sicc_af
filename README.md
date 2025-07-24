# sicc_αf

sicc_αf - **S**tructure **I**ntegrity **C**heck and **C**orrection for **AlphaF**old2 predicted protein structures

sicc_αf is a workflow providing repair of amino acid side chains in protein structures predicted by [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2)
algorithm. It can be either run locally as a command line application or integrated as a regular Python library.
It uses following libraries: [Biopython](https://doi.org/10.1093/bioinformatics/btp163), [RDKit](https://www.rdkit.org/),
 [scikit-learn](https://dl.acm.org/doi/10.5555/1953048.2078195); and the [PDB2PQR](https://doi.org/10.1093/nar/gkh381)
suite.

## Command line use
setup:

```bash
$ git clone https://github.com/kekasz/sicc_af
$ cd sicc_af/
$ sudo python3.12 -m venv sicc_af
$ source sicc_af/bin/activate
$ pip install -r requirements.txt
```

running:

```
$ python sicc_af.py <input> [output] [-l log] [-d -s]
```

input: path to PDB file to proccess with sicc_αf (mandatory)\
output: path for the corrected file (optional)\
log: path for the log file (optional, must be preceded by -l)\
-d: delete auxilliary files (also deletes the log file, if its path has not been specified)\
-s: silent mode

## Python library integration
Import the StructureIntegrityCheckerAndCorrector into your Python script. All options listed in Command line use are
available within the constructor parameters. The logger parameter is used for processing large sets of PDB files using a
custom script. Ignore for casual use.
