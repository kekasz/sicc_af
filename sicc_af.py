from rdkit.Chem import MolFromPDBFile, rdchem
from Bio.PDB import PDBIO, PDBParser, Select
from os.path import basename, dirname, isdir
from os import mkdir
from collections import defaultdict
# from shutil import rmtree
from typing import Union
from sicc_af_data import archetypes, distances, final_statuses


class Atom:
    def __init__(self, rdkit_atom : rdchem.Atom):
        self._res_id = rdkit_atom.GetPDBResidueInfo().GetResidueNumber()
        self._res_name = rdkit_atom.GetPDBResidueInfo().GetResidueName()
        self._nef_name = rdkit_atom.GetPDBResidueInfo().GetName().strip()
        self._id = rdkit_atom.GetPDBResidueInfo().GetSerialNumber()
        self.bonded_ats_nef_names = set()

    def get_res_id(self) -> int:
        return self._res_id

    def get_res_name(self) -> str:
        return self._res_name

    def get_id(self) -> int:
        return self._id

    def get_nef_name(self) -> str:
        return self._nef_name

    def add_bonded_at_nef_name(self, nef_name : str):
        self.bonded_ats_nef_names.add(nef_name)

    def get_bond_ats_nef_names(self) -> set[str]:
        return self.bonded_ats_nef_names


class Residue:
    def __init__(self, atoms_dict: dict[int, Atom]):
        super().__init__()
        (self._id,) = {atom.get_id() for atom in atoms_dict.values()}
        (self._name,) = {atom.get_res_name() for atom in atoms_dict.values()}
        self._atoms_dict = atoms_dict

    def get_id(self) -> int:
        return self._id

    def get_atoms(self) -> set[Atom]:
        return set(self._atoms_dict.values())

    def add_atom(self, atom : Atom):
        idx = atom.get_id()
        self._atoms_dict[idx] = atom

    def get_name(self) -> str:
        return self._name

    def get_min_at_distance(self) -> int:
        return min({distances[atom.get_nef_name()] for atom in self.get_atoms() if len(atom.get_nef_name()) > 1})

    def get_ats_nef_names(self) -> set[str]:
        return {atom.get_nef_name() for atom in self.get_atoms()}

    def get_ats_ids(self) -> set[int]:
        return set(self._atoms_dict.keys())


class Protein:
    def __init__(self, residues: dict[int, Residue], path: str = None):
        self._residues = residues
        if path:
            self._path = path
            self._name = basename(path)[3:-16]

    def get_atoms(self) -> set[Atom]:
        return {atom for residue in self.get_residues() for atom in residue.get_atoms()}

    def get_residues(self) -> set[Residue]:
        return set(self._residues.values())

    def get_residue(self, res_id: int) -> Residue:
        return self._residues[res_id]

    def get_ress_ids(self) -> set[int]:
        return set(self._residues.keys())

    def get_res_err_distance(self, res_id: int) -> int:
        return self._residues[res_id].get_min_at_distance()

    def get_path(self) -> str:
        return self._path

    def get_res_ats_nef_names(self, res_id) -> set[str]:
        return self._residues[res_id].get_ats_nef_names()


class ClusterErroneousResidue(BasicResidue):
    def __init__(self, residue: Residue):
        super().__init__()
        self._id = residue.get_id()
        self._error_distance = int()
        self._status = 'ERRONEOUS'
        self._atoms = dict()
        self._original_atoms = dict()
        self._make_atoms_dicts(residue)

    def _make_atoms_dicts(self, residue: Residue):
        self._atoms = self._original_atoms = {atom.get_nef_name(): atom for atom in residue.get_atoms()}

    def _remove_atoms_layer(self, err_oxygens: set[str]):
        # ABSENCE OF _error_distance MEANS ONLY CHAIN OXYGEN ERRORS
        if self._error_distance:
            self._atoms = {nef_name: atom for nef_name, atom in self._atoms.items() if len(nef_name) == 1 or distances[nef_name[1]] < self._error_distance}

        if err_oxygens:
            for oxygen_nef_name in err_oxygens:
                del self._atoms[oxygen_nef_name]

    def _get_ats_nef_names(self) -> set[str]:
        return set(self._atoms.keys())

    def _get_original_atoms(self) -> set[Union[Atom]]:
        return set(self._original_atoms.values())

    def make_error_distance(self, protein_errors: Protein):
            err_ats_nef_names = protein_errors.get_res_ats_nef_names(self._id)

            # CHECK FOR UNSOLVABLE CHAIN ERROR
            if err_ats_nef_names == {'N', 'C'}:
                self._status = "CHAIN ERROR"
                return

            # ERRONEOUS CHAIN OXYGENS MUST BE TREATED SEPARATEDLY
            chain_oxygens_nef_names = {nef_name for nef_name in err_ats_nef_names if nef_name in {'O', 'OXT'}}

            # RESIDUES WITH CHAIN OXYGEN ERRORS ONLY MUST BE TREATED SEPARATEDLY
            if err_ats_nef_names != {'O', 'OXT'}:
                self._error_distance = protein_errors.get_res_err_distance(self._id)

            self._remove_atoms_layer(chain_oxygens_nef_names)

    def get_error_distance(self) -> int:
        return self._error_distance

    def get_status(self) -> str:
        return self._status

    def get_ats_ids(self) -> set[int]:
        if self._status in final_statuses:
            return {atom.get_id() for atom in self._get_original_atoms()}
        else:
            return {atom.get_id() for atom in self.get_atoms()}


class ErrorCluster:
    def __init__(self):
        self._residues = dict()
        self._err_ress = dict()
        self._max_error_distance = int()
        self._status = 'ERRONEOUS'

    def get_err_ress(self) -> set[ClusterErroneousResidue]:
        return set(self._err_ress.values())

    def add_err_res(self, residue: Residue):
        res_id = residue.get_id()
        self._err_ress[res_id] = ClusterErroneousResidue(residue)

    def add_residue(self, residue: Residue):
        res_id = residue.get_id()
        self._residues[res_id] = residue

    def make_error_distances(self, protein_errors: Protein):
        """Prepares cluster for the first iteration of correction."""
        for err_res in self._err_ress.values():
            err_res.make_error_distance(protein_errors)

        # CHECK FOR UNSOLVABLE CHAIN ERROR
        statuses = {res.get_status() for res in self.get_err_ress()}
        if statuses == {'CHAIN ERROR'}:
            self._status = 'CHAIN ERROR'
            return

        max_error_distance = max(res.get_error_distance() for res in self.get_err_ress())
        if max_error_distance:
            self._max_error_distance = max_error_distance

    def get_status(self) -> str:
        return self._status

    def get_residues(self) -> set[Residue]:
        return set(self._residues.values())

    def get_ats_ids(self) -> set[int]:
        if self._status in final_statuses:
            return set()

        ats_ids = set()
        for residue in (self.get_residues() | self.get_err_ress()):
            ats_ids.update(residue.get_ats_ids())
        return ats_ids


class ErrorClusters:
    def __init__(self, protein_errors : Protein, protein : Protein):
        self._path = protein_errors.get_path()
        self._name = basename(self._path)[3:-16]
        self._correction_dir_path = f'{dirname(self._path)}/{self._name}_correction'
        self._protein_errors = protein_errors
        self._protein = protein
        self._err_ress_ids = protein_errors.get_ress_ids()
        self._correction_level = 1
        self._status = 'ERRONEOUS'

        self._clusters = dict()
        self._ats_set = set()
        self._make_clusters()
        self._make_io()

    def _clusterise_res_ids(self):
        ids_clusters = []
        bio_residues = BioHandler().get_residues(self._path)

        # CLUSTERISE ERRONEOUS RESIDUES
        left = self._err_ress_ids
        i = 0
        for er_idx in self._err_ress_ids:
            if er_idx in left:
                taken = {er_idx}
                last_taken = set()

                ids_clusters.append(frozenset())
                while taken != last_taken:

                    for er_idx_1 in taken - last_taken:
                        for er_idx_2 in left:
                            if bio_residues[er_idx_1]['CA'] - bio_residues[er_idx_2]['CA'] < 10 and er_idx_1 != er_idx_2:
                                taken.add(er_idx_2)

                    last_taken = taken

                ids_clusters[i] = frozenset(taken)
                i += 1

        # ADD SURROUNDING RESIDUES
        for cluster in ids_clusters:
            self._clusters[cluster] = ErrorCluster()

            for err_res_id in cluster:
                err_res = self._protein.get_residue(err_res_id)
                self._clusters[cluster].add_err_res(err_res)

                bio_err_res = bio_residues[err_res_id]
                for bio_res in bio_residues:
                    if bio_err_res['CA'] - bio_res['CA'] < 10:
                        res_id = bio_res.get_id()[1]

                        if res_id in cluster:
                            continue
                        else:
                            residue = self._protein.get_residue(res_id)
                            self._clusters[cluster].add_residue(residue)

    def _get_clusters(self) -> set[ErrorCluster]:
        return set(self._clusters.values())

    def _make_errs_distances(self):
        for cluster in self._get_clusters():
            cluster.make_error_distances(self._protein_errors)

            statuses = {cluster.get_status() for cluster in self._get_clusters()}
            if statuses == {'CHAIN ERROR'}:
                self._status = 'CHAIN ERROR'

    def _make_ats_set(self):
        self._ats_set = {at_id for cluster in self._get_clusters() for at_id in cluster.get_ats_ids()}

    def _make_clusters(self):
        self._clusterise_res_ids()
        self._make_errs_distances()
        self._make_ats_set()

        statuses = {cluster.get_status() for cluster in self._get_clusters()}
        if statuses == {'CHAIN ERROR'}:
            self._status = 'CHAIN ERROR'

    def _update_ats_set(self):
        for cluster in self._clusters.values():
            ...

    def _make_io(self):
        self._io = BioHandler().get_io(self._path)

    def _update_status(self):
        if "ERRONEOUS" not in set([cluster.get_status() for cluster in self._clusters.values()]):
            self._status = "CORRECTION FAILED"

    def get_io(self) -> PDBIO:
        return self._io

    def get_status(self) -> str:
        self._update_status()
        return self._status

    def get_ats_ids(self) -> set:
        return {at_id for cluster in self._get_clusters() for at_id in cluster.get_ats_ids()}

    def get_correction_dir_path(self) -> str:
        return self._correction_dir_path

    def get_correction_level(self) -> int:
        return self._correction_level

    def increment_correction_level(self):
        self._update_ats_set()
        self._correction_level += 1


class RDKitHandler:
    def __init__(self):
        self._path = str()

    def _load_rdkit_molecule(self):
        try:
            self._rdkit_molecule = MolFromPDBFile(self._path,
                                                  removeHs=False,
                                                  sanitize=False)
        except KeyError:
            print(f"ERROR! File at {self._path} does is not a valid PDB file.\n")
            exit()

    def _make_atoms_dict(self):
        self._atoms_dict = {rdkit_atom.GetPDBResidueInfo().GetSerialNumber(): Atom(rdkit_atom) for rdkit_atom in self._rdkit_molecule.GetAtoms()}
        self._clashing_ats_set = set()

    def _add_bonded_ats(self):
        rdkit_bonds = self._rdkit_molecule.GetBonds()

        for bond in rdkit_bonds:
            a1_id = bond.GetBeginAtom().GetPDBResidueInfo().GetSerialNumber()
            a2_id = bond.GetEndAtom().GetPDBResidueInfo().GetSerialNumber()
            atom1 = self._atoms_dict[a1_id]
            atom2 = self._atoms_dict[a2_id]
            a1_res_id = atom1.get_res_id()
            a2_res_id = atom2.get_res_id()
            a1_nef_name = atom1.get_nef_name()
            a2_nef_name = atom2.get_nef_name()

            if {a1_nef_name, a2_nef_name} == {'SG', 'SG'}:
                continue

            if a1_res_id == a2_res_id or (abs(a1_res_id - a2_res_id) == 1 and {a1_nef_name, a2_nef_name} == {'N', 'C'}):

                self._atoms_dict[a1_id].add_bonded_at_nef_name(a2_nef_name)
                self._atoms_dict[a2_id].add_bonded_at_nef_name(a1_nef_name)

            else:
                self._clashing_ats_set.add(atom1)
                self._clashing_ats_set.add(atom2)

    def _make_protein(self):
        self._protein = make_protein(set(self._atoms_dict.values()), self._path)

    def get_protein(self, path: str) -> (Protein, set[Atom]):
        self._path = path
        self._load_rdkit_molecule()
        self._make_atoms_dict()
        self._add_bonded_ats()
        self._make_protein()

        return self._protein, self._clashing_ats_set


class IntegrityChecker:
    def __init__(self):
        self._path = None
        self._protein = None
        self._err_ats_set = None
        self._first_check = True

    def _check_atom(self):
        self._atom_invalid = True
        res_id = self._atom.get_res_id()
        res_name = self._atom.get_res_name()
        at_nef_name = self._atom.get_nef_name()
        bonded_ats_nef_names = self._atom.get_bond_ats_nef_names()
        archetype = archetypes[res_name][at_nef_name]

        if res_id == 1 and at_nef_name == 'N':
            archetype = archetype - {'C'}

        elif res_id == len(self._protein.get_residues()):
            if at_nef_name == 'C':
                archetype = archetype - {'N'}
                archetype.add('OXT')

            if at_nef_name == 'OXT':
                archetype = archetypes[res_name]['O']

        elif not self._first_check:
            if at_nef_name in {'N', 'C'}:
                archetype = archetype - {'N', 'C'}

        if bonded_ats_nef_names == archetype:
            self._atom_invalid = False

    def _make_protein_errors(self):
        self._protein_errors = make_protein(self._err_ats_set)

    def _check_protein(self):
        for atom in self._protein.get_atoms():
            self._atom = atom
            self._check_atom()

            if self._atom_invalid:
                self._err_ats_set.add(atom)

        self._make_protein_errors()

    def get_errors_and_protein(self, path) -> (Protein, Protein):
        self._path = path

        self._protein, self._err_ats_set = RDKitHandler().get_protein(path)
        self._check_protein()

        return self._protein_errors, self._protein

    def get_errors(self, path) -> Protein:
        self._first_check = False
        self._path = path
        self._protein, self._err_ats_set = RDKitHandler().get_protein(path)

        self._check_protein()

        return self._err_ats_set # ale ErrorProtein


class BioHandler:
    def __init__(self):
        self._path = None
        self._clusters = None

    def _make_structure(self):
        self._structure = PDBParser(QUIET=True).get_structure('protein', self._path)

    def get_residues(self, path: str):
        self._path = path
        self._make_structure()
        return self._structure[0]['A']

    def get_io(self, path) -> PDBIO:
        self._path = path
        self._make_structure()
        io = PDBIO()
        io.set_structure(self._structure)

        return io

    def _get_selector(self):

        class SelectIndexedAtoms(Select):
            def __init__(self, indices):
                super().__init__()
                self.indices = indices
            def accept_atom(self, atom):
                if atom.get_serial_number() in self.indices:
                    return 1
                else:
                    return 0

        ats_ids = self._clusters.get_ats_ids()
        self._selector = SelectIndexedAtoms(ats_ids)

    def get_selection_file_path(self, clusters: ErrorClusters) -> str:
        self._clusters = clusters
        self._get_selector()

        # CHECK OR MAKE CORRECTION DIRECTORY
        correction_dir_path = clusters.get_correction_dir_path()
        if not isdir(correction_dir_path):
            mkdir(correction_dir_path)

        # SAVE FILE WITH CUT-OUT CLUSTERS
        correction_level = clusters.get_correction_level()
        selection_file_path = f'{correction_dir_path}/level{correction_level}.pdb'
        io = self._clusters.get_io()
        io.save(selection_file_path, self._selector)

        return selection_file_path


class StructureCorrector:
    def __init__(self):
        self._path = None
        self._protein_errors = None
        self._protein = None

    def _get_clusters(self):
        self._clusters = ErrorClusters(self._protein_errors, self._protein)

    def _try_correcting(self):
        selection_file_path = BioHandler().get_selection_file_path(self._clusters)
        #2 add run propka
        self._protein_errors = IntegrityChecker().get_errors(selection_file_path)
        ...

    def _execute_correction(self):
        self._get_clusters()

        i = 0
        while self._protein_errors and self._clusters.get_status() != "CORRECTION FAILED" and i == 0:
            self._try_correcting()
            self._clusters.increment_correction_level() #4 dopracovať
            i = 1

        # correction_dir_path = f"{dirname(self._path)}/{basename(self._path)[3:-16]}_correction"
        # rmtree(correction_dir_path)

    def correct_structure(self, path: str):
        self._path = path
        self._protein_errors, self._protein = IntegrityChecker().get_errors_and_protein(path)

        if self._protein_errors is None :
            name = basename(path)[3:-16]
            print(f'OK: No error found in {name}.')

        else:
            self._execute_correction()


def make_protein(atoms: set[Atom], path: str = None) -> Protein:
        ress_ats_dicts_dict = defaultdict(dict)

        for atom in atoms:
            at_id = atom.get_id()
            res_id = atom.get_res_id()
            ress_ats_dicts_dict[res_id][at_id] = atom

        residues_dict = {}

        for res_id, res_ats_dict in ress_ats_dicts_dict.items():
            residues_dict[res_id] = Residue(res_ats_dict)

        return Protein(residues_dict, path)

StructureCorrector().correct_structure('/home/l/pycharmprojects/sicc_af/bordel/AF-Q9Y7W4-F1-model_v4.pdb') # rozstrelený H bez clashu
# StructureCorrector().correct_structure('/home/l/pycharmprojects/sicc_af/bordel/AF-A0A1D8PTL3-F1-model_v4.pdb') # clash 2 W