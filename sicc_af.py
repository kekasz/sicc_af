from rdkit import Chem
# from Bio.PDB.PDBParser import PDBParser
from os.path import basename
from typing import Union
from archetypes import archetypes


class BasicAtom:
    def __init__(self, rdkit_atom : Chem.rdchem.Atom):

        self._res_idx = rdkit_atom.GetPDBResidueInfo().GetResidueNumber()
        self._res_name = rdkit_atom.GetPDBResidueInfo().GetResidueName()
        self._nef_name = rdkit_atom.GetPDBResidueInfo().GetName().strip()
        self._idx = rdkit_atom.GetIdx()

    def get_res_idx(self) -> int:
        return self._res_idx

    def get_res_name(self) -> str:
        return self._res_name

    def get_idx(self) -> int:
        return self._idx

    def get_nef_name(self) -> str:
        return self._nef_name


class BasicProtein:
    def __init__(self):
        self._residues = {}

    def get_residues(self) -> dict.values:
        return self._residues.values()


class Atom(BasicAtom):
    def __init__(self, rdkit_atom : Chem.rdchem.Atom):
        super().__init__(rdkit_atom)

        self.bonded_ats = {}
        self.clashing_ress_ats = {}

    def add_bonded_at(self, bond_at : BasicAtom):
        nef_name = bond_at.get_nef_name()
        self.bonded_ats[nef_name] = bond_at

#1 vcelku robustný zápis. nezjednodušiť defaultdict-om? aj za cenu prepisovania nemenných údajov?
    def add_clashing_atom(self, clash_at):
        res_idx = clash_at.get_res_idx()
        if res_idx in self.clashing_ress_ats.keys():
            self.clashing_ress_ats[res_idx].add_atom(clash_at)
        else:
            self.clashing_ress_ats[res_idx] = Residue(clash_at)

    def get_clashing_ress_idxs(self) -> dict.keys:
        return self.clashing_ress_ats.keys()

    def get_bond_ats_nef_names(self):
        return self.bonded_ats.keys()


class Residue:
    def __init__(self, init_atom : Union[BasicAtom, Atom]):

        self._idx = init_atom.get_res_idx()
        self._name = init_atom.get_res_name()
        self._atoms = {init_atom.get_idx(): init_atom}

    def add_atom(self, atom : Atom):
        idx = atom.get_idx()
        self._atoms[idx] = atom

    def get_name(self) -> str:
        return self._name

    def get_atoms(self) -> dict.values:
        return self._atoms.values()

    def get_idx(self) -> int:
        return self._idx


class Protein(BasicProtein):
    def __init__(self, path: str, residues: dict):
        super().__init__()
        self._path = path
        self._name = basename(path)[3:-16]
        self._residues = residues


class ErrorProtein(BasicProtein):
    def add_atom(self, atom: Union[BasicAtom, Atom]):
        res_idx = atom.get_res_idx()

        if res_idx in self._residues.keys():
            self._residues[res_idx].add_atom(atom)
        else:
            self._residues[res_idx] = Residue(atom)


class RDKitHandler:
    def __init__(self):
        self._path = None
        self._clashing_atoms = None

    def _load_rdkit_molecule(self):

        try:
            self._rdkit_molecule = Chem.MolFromPDBFile(self._path,
                                       removeHs=False,
                                       sanitize=False)
        except KeyError:
            print(f"ERROR! File at {self._path} does is not a valid PDB file.\n")
            exit()

    def _make_atoms_dict(self):

        rdkit_atoms = self._rdkit_molecule.GetAtoms()
        self._atoms_dict = {}

        for rdkit_atom in rdkit_atoms:
            print(type(rdkit_atom))
            atom = Atom(rdkit_atom)
            atom_idx = atom.get_idx()
            self._atoms_dict[atom_idx] = atom

    def _add_bonded_ats(self):
        rdkit_bonds = self._rdkit_molecule.GetBonds()
        for bond in rdkit_bonds:

            rdkit_atom1 = bond.GetBeginAtom()
            rdkit_atom2 = bond.GetEndAtom()
            atom1 = BasicAtom(rdkit_atom1)
            atom2 = BasicAtom(rdkit_atom2)
            a1_res_idx = atom1.get_res_idx()
            a2_res_idx = atom2.get_res_idx()
            a1_nef_name = atom1.get_nef_name()
            a2_nef_name = atom2.get_nef_name()
            a1_idx = atom1.get_idx()
            a2_idx = atom2.get_idx()

            if a1_res_idx == a2_res_idx or (abs(a1_res_idx - a2_res_idx) == 1 and {a1_nef_name, a2_nef_name} == {'N', 'C'}):

                self._atoms_dict[a1_idx].add_bonded_at(atom2)
                self._atoms_dict[a2_idx].add_bonded_at(atom1)

            elif {a1_nef_name, a2_nef_name} == {'SG', 'SG'}:
                pass

            else:
                self._atoms_dict[a1_idx].add_clashing_atom(atom2)
                self._atoms_dict[a2_idx].add_clashing_atom(atom1)

                if self._clashing_atoms is None:
                    self._clashing_atoms = ErrorProtein()

                self._clashing_atoms.add_atom(atom1)
                self._clashing_atoms.add_atom(atom2)

    def _make_residues_dict(self):

        self._residues_dict = {}

        for atom in self._atoms_dict.values():
            res_idx = atom.get_res_idx()

#1 -//-
            if res_idx not in self._residues_dict.keys():
                self._residues_dict[res_idx] = Residue(atom)
            else:
                self._residues_dict[res_idx].add_atom(atom)

    def _make_protein(self):
        self._protein = Protein(self._path, self._residues_dict)

    def get_protein(self, path: str) -> (Protein, ErrorProtein):
        self._path = path

        self._load_rdkit_molecule()
        self._make_atoms_dict()
        self._add_bonded_ats()
        self._make_residues_dict()
        self._make_protein()

        return self._protein


class IntegrityChecker:
    def __init__(self):
        self._archetypes = archetypes
        self._protein = None
        self._protein_errors = None

    def _check_atom(self):
        self._atom_invalid = True
        res_idx = self._atom.get_res_idx()

        res_name = self._atom.get_res_name()
        at_nef_name = self._atom.get_nef_name()
        bonded_ats_nef_names = self._atom.get_bond_ats_nef_names()
        archetype = self._archetypes[res_name][at_nef_name]

        if bonded_ats_nef_names == archetype:
                self._atom_invalid = False

        elif res_idx == 1:

            if at_nef_name == 'N':
                archetype = archetype - {'C'}

                if bonded_ats_nef_names == archetype:
                    self._atom_invalid = False

        elif res_idx == len(self._residues):\

            if at_nef_name == 'C':
                archetype = archetype - {'N', 'O'}
                archetype.add('OXT')

                if bonded_ats_nef_names == archetype:
                    self._atom_invalid = False

            elif at_nef_name == 'OXT':
                archetype = archetypes[res_name]['O']

                if bonded_ats_nef_names == archetype:
                    self._atom_invalid = False

    def _check_protein(self):
        self._residues = self._protein.get_residues()

        for residue in self._residues:
            atoms = residue.get_atoms()

            for atom in atoms:

                self._atom = atom
                self._check_atom()

                if self._atom_invalid:

                    if self._protein_errors is None:
                        self._protein_errors = ErrorProtein()

                    self._protein_errors.add_atom(atom)

    def check_integrity(self, path):

        self._protein, self._protein_errors = RDKitHandler().get_protein(path)

        self._check_protein()

        return self._protein_errors


class BioHandler:
    def __init__(self):
        ...


IntegrityChecker().check_integrity('/home/l/AF-Q9Y7W4-F1-model_v4.pdb')