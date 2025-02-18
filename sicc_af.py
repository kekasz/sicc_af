from collections import defaultdict
from os import mkdir, system
from os.path import basename, dirname, isdir
from re import sub

from Bio.PDB import PDBIO, PDBParser, Select
from rdkit.Chem import MolFromPDBFile, rdchem

# from shutil import rmtree
from sicc_af_data import archetypes, distance, proline_distance


class Atom:
    def __init__(self,
                 rdkit_atom:    rdchem.Atom):
        rdkit_pdb_info = rdkit_atom.GetPDBResidueInfo()
        self.res_id = rdkit_pdb_info.GetResidueNumber()
        self.res_name = rdkit_pdb_info.GetResidueName()
        self.nef_name = rdkit_pdb_info.GetName().strip()
        self.id = rdkit_pdb_info.GetSerialNumber()
        self.bonded_ats_nef_names = set()
        self.status = 'CORRECT'  # correction status


class Residue:
    def __init__(self,
                 atoms_dict:        dict[int, Atom],
                 err_atoms_dict:    dict[int, Atom] = None):
        init_atom = list(atoms_dict.values())[0]
        self._id = init_atom.res_id
        self._name = init_atom.res_name
        self._err_atoms_dict = err_atoms_dict
        self.atoms = atoms_dict
        self.ox_errs_ats_dict = {}

        # process errors
        if err_atoms_dict:
            self.status = 'ERRONEOUS'
            self.min_err_distance = None

            # separate chain oxygen errors (they need to be treated separatedly)
            for at_id, atom in err_atoms_dict.items():
                if atom.nef_name[0] == 'O':
                    self.ox_errs_ats_dict[at_id] = atom
            for at_id in self.ox_errs_ats_dict.keys():
                    del self._err_atoms_dict[at_id]

            # calculate "mininal error distance" (distance of the closest residual chain erroneous atom to the chain)
            if err_atoms_dict:
                # purvey proline particularities
                if self._name == 'PRO':
                    self.min_err_distance = min({proline_distance[atom.nef_name[1]] for atom in err_atoms_dict.values()})
                else:
                    self.min_err_distance = min({distance[atom.nef_name[1]] for atom in err_atoms_dict.values()})
        else:
            self.status = 'CORRECT'

    def get_kept_ats_ids(self,
                         correction_level:  int) -> set[int]:
        """Returns ids of atoms to be copied into a correction file."""

        # select atoms to be kept
        ats_ids = set()
        for atom in self.atoms.values():
            # leave out erroneous atoms
            if atom.status != 'ERRONEOUS' or atom.nef_name[0] == 'O':
                # leave out atoms further in residue than any of the erroneous ones
                if len(sub(r'[^a-zA-Z]', '', atom.nef_name)) == 2:
                    # purvey proline particularities
                    if (self._name == 'PRO'
                            and proline_distance[atom.nef_name[1]] < self.min_err_distance - correction_level):
                        ats_ids.add(atom.id)
                    elif distance[atom.nef_name[1]] < self.min_err_distance - correction_level:
                        ats_ids.add(atom.id)
                else:
                    ats_ids.add(atom.id)

        return ats_ids

    def get_ats_nef_names(self) -> set[str]:
        return {atom.nef_name for atom in self.atoms.values()}


class Protein:
    def __init__(self,
                 path: str):
        """The protein is loaded from a PDB file. Error bonds are detected and corresponding atoms and residues are
        noted."""

        self.status = 'CORRECT'
        self.name = basename(path)[3:-16]
        self._residues = {}
        self._path = path
        self._error_residues = {}
        self._chain_errors = False
        self._correction_dir_path = f'{dirname(self._path)}/{self.name}_correction'
        self._correction_level = 0

        # load RDKit molecule from PDB file
        try:
            rdkit_molecule = MolFromPDBFile(self._path,
                                            removeHs=False,
                                            sanitize=False)
        except KeyError:
            exit(f"ERROR! File at {self._path} is not a valid PDB file.\n")

        # make a dictionary of all protein's atoms {atom_id : Atom}
        atoms_dict = {rdkit_atom.GetPDBResidueInfo().GetSerialNumber(): Atom(rdkit_atom)
                      for rdkit_atom in rdkit_molecule.GetAtoms()}

        # make a set of bonded atoms' NEF names for each atom & check for interresidual clashes
        # (check bonds detected by RDKit for atoms pairs)
        err_ats_set = set()
        chain_err_ats_set = set()
        for bond in rdkit_molecule.GetBonds():
            a1_id = bond.GetBeginAtom().GetPDBResidueInfo().GetSerialNumber()
            a2_id = bond.GetEndAtom().GetPDBResidueInfo().GetSerialNumber()
            atom1 = atoms_dict[a1_id]
            atom2 = atoms_dict[a2_id]
            a1_nef_name = atom1.nef_name
            a2_nef_name = atom2.nef_name

            # purvey actual disulphide bonds between cysteins' sulphurs
            if {a1_nef_name, a2_nef_name} == {'SG', 'SG'}:
                continue

            # if the bond is within the same residue or is eupeptidic, add to the set
            a1_res_id = atom1.res_id
            a2_res_id = atom2.res_id
            if a1_res_id == a2_res_id or (abs(a1_res_id - a2_res_id) == 1 and {a1_nef_name, a2_nef_name} == {'N', 'C'}):
                atoms_dict[a1_id].bonded_ats_nef_names.add(a2_nef_name)
                atoms_dict[a2_id].bonded_ats_nef_names.add(a1_nef_name)

            # purvey interresidual clashes of chain atoms
            elif {a1_nef_name, a2_nef_name} < {'N', 'C', 'CA'}:
                self._chain_errors = True
                chain_err_ats_set.add(atom1)
                chain_err_ats_set.add(atom2)
                atom1.status = 'CHAIN ERROR'
                atom2.status = 'CHAIN ERROR'

            # else mark atoms as erroneous, if they are not chain atoms
            else:
                self.status = 'ERRONEOUS'
                if a1_nef_name not in {'N', 'C', 'CA'}:
                    err_ats_set.add(atom1)
                    atom1.status = 'ERRONEOUS'
                if a2_nef_name not in {'N', 'C', 'CA'}:
                    err_ats_set.add(atom2)
                    atom2.status = 'ERRONEOUS'

        # check if the atom is bonded to expected atoms
        ress_ids = {atom.res_id for atom in atoms_dict.values()}
        last_res_id = max(ress_ids)
        for atom in atoms_dict.values():
            if atom.id == 2882:
                pass
            res_id = atom.res_id
            res_name = atom.res_name
            at_nef_name = atom.nef_name
            bonded_ats_nef_names = atom.bonded_ats_nef_names
            archetype = archetypes[res_name][at_nef_name]
            erroneous = False

            # modify the archetype for the atom's bonded atoms
            # purvey the initial nitrogen
            if res_id == 1 and at_nef_name == 'N':
                archetype = archetype - {'C'}

            # purvey the terminal carbon and oxygen
            elif res_id == last_res_id:
                if at_nef_name == 'C':
                    archetype = archetype - {'N'}
                    archetype.add('OXT')
                if at_nef_name == 'OXT':
                    archetype = archetypes[res_name]['O']

            # purvey ends of local chains in correction files
            elif at_nef_name in {'N', 'C'}:
                if ((at_nef_name == 'N' and res_id-1 not in ress_ids) or
                        (at_nef_name == 'C' and res_id+1 not in ress_ids)):
                    archetype = archetype - {'N', 'C'}

            # select atoms to be marked as erroneous
            if bonded_ats_nef_names != archetype:
                # ignore chain atoms in non-chain errors only
                if at_nef_name in {'N', 'C', 'CA', 'O', 'OXT'}:
                    if {'N', 'C', 'CA'} & archetype != {'N', 'C', 'CA'} & bonded_ats_nef_names:
                        if at_nef_name in {'O', 'OXT'}:
                            erroneous = True
                        else:
                            self._chain_errors = True
                            chain_err_ats_set.add(atom)
                            atom.status = 'CHAIN ERROR'
                # if there are any extra atoms bonded, mark erronous
                if not bonded_ats_nef_names < archetype:
                    erroneous = True

                # purvey proline particularities
                elif res_name == 'PRO' and at_nef_name in {'CD', 'CG'}:
                    if at_nef_name == 'CD' and 'N' not in bonded_ats_nef_names:
                        erroneous = True
                    if at_nef_name == 'CG':
                        erroneous = True

                # ignore the atoms that are the "very last correct" in the residue
                elif not all([distance[missing_at_nef_name[1]] > distance[at_nef_name[1]]
                              for missing_at_nef_name in archetype - bonded_ats_nef_names
                              if len(sub(r'[^a-zA-Z]', '', missing_at_nef_name)) == 2]): # possible number at the end of the nef name (e.g. CE1 and NE2 in HIS)
                    erroneous = True
            if erroneous:
                self.status = 'ERRONEOUS'
                atom.status = 'ERRONEOUS'
                err_ats_set.add(atom)

        # clusterize atoms and erroneous atoms into residues and erroneous residues dictionaries {residue_id : Residue}
        ress_ats_dicts_dict = self._make_ress_ats_dicts_dict(set(atoms_dict.values()))
        err_ress_ats_dicts_dict = self._make_ress_ats_dicts_dict(err_ats_set)

        self._residues = {res_id: Residue(atoms_dict =      res_ats_dict,
                                          err_atoms_dict =  err_ress_ats_dicts_dict[res_id] if res_id in err_ress_ats_dicts_dict.keys()
                                                                                            else None)
                          for res_id, res_ats_dict in ress_ats_dicts_dict.items()}
        self._error_residues = {res_id: self._residues[res_id]
                                for res_id in err_ress_ats_dicts_dict.keys()}
        self._chain_error_ress = {atom.res_id for atom in chain_err_ats_set}

    @staticmethod
    def _make_ress_ats_dicts_dict(atoms: set[Atom]) -> dict[int, dict[int, Atom]]:
        ress_ats_dicts_dict = defaultdict(dict)
        for atom in atoms:
            ress_ats_dicts_dict[atom.res_id][atom.id] = atom
        return dict(ress_ats_dicts_dict)

    def execute_correction(self):
        """Makes clusters of error residues and their neighbouring residues."""

        # clusterize erroneous residues into a list of sets of their ids

        # load Biopython structure from the PDB file
        try:
            bio_structure = PDBParser(QUIET=True).get_structure('protein', self._path)
        except KeyError:
            exit(f"ERROR! File at {self._path} is not a valid PDB file.\n")

        bio_residues = {residue.id[1]: residue for residue in bio_structure.get_residues()}
        err_ress_ids_clusters = set()
        left = set(self._error_residues.keys()).copy() # error resiudes not already added to a cluster

        for err_res_id in self._error_residues.keys():
            if err_res_id in left:
                taken = {err_res_id}
                left.remove(err_res_id)
                growth = taken.copy()

                # clusterize until the cluster stops growing
                while growth:
                    old_taken = taken.copy()

                    # try to find nearby residues. consider left residues and the newest taken residues only
                    for err_res_id_1 in growth:
                        new_taken = set()
                        for err_res_id_2 in left:
                            # compare the distance between C-αs of the residues pair
                            if bio_residues[err_res_id_1]['CA'] - bio_residues[err_res_id_2]['CA'] < 10:
                                taken.add(err_res_id_2)
                                new_taken.add(err_res_id_2)

                        left = left - new_taken

                    growth = taken - old_taken

                err_ress_ids_clusters.add(frozenset(taken))

        # execute correction over clusters
        class SelectIndexedAtoms(Select):
            def __init__(self, indices):
                super().__init__()
                self.indices = indices
            def accept_atom(self, atom):
                if atom.get_serial_number() in self.indices:
                    return 1
                else:
                    return 0
        for cluster_err_ids in err_ress_ids_clusters:

            # find surrounding residues and list their atoms' ids
            surr_ats_ids = set()
            left_bio_residues = {residue for residue in bio_structure.get_residues()
                                 if residue.get_id()[1] not in self._error_residues.keys()}
            for err_res_id in cluster_err_ids:
                bio_err_res = bio_residues[err_res_id]
                new_surr = set()

                # select residues with CA-distance under 10 Å from any of the erroneous residues in the cluster
                for bio_res in left_bio_residues:
                    if bio_err_res['CA'] - bio_res['CA'] < 10:
                        surr_res_id = bio_res.get_id()[1]
                        surr_ats_ids.update(self._residues[surr_res_id].atoms.keys())
                        new_surr.add(bio_res)

                left_bio_residues = left_bio_residues - new_surr

            # try iterations of correction
            cluster_err_ress = {res_id: self._residues[res_id] for res_id in cluster_err_ids}
            max_error_distance = max({res.min_err_distance if res.min_err_distance else 1
                                      for res in cluster_err_ress.values()})
            last_err_ress = cluster_err_ress
            correction_level = 0
            kept_ats_ids = set()
            io = PDBIO()

            # ensure existence of a directory for the protein's correction
            if not isdir(self._correction_dir_path):
                mkdir(self._correction_dir_path)


            while not all([residue.status != 'ERRONEOUS' for residue in cluster_err_ress.values()]):

                if correction_level == max_error_distance:
                    print('Correction unsuccessful!')
                    break

                io.set_structure(bio_structure)

                for res_id, residue in last_err_ress.items():
                    if residue.status == 'ERRONEOUS':
                        kept_ats_ids = self._residues[res_id].get_kept_ats_ids(correction_level)
                    for oxygen_id in residue.ox_errs_ats_dict.keys():
                        kept_ats_ids.remove(oxygen_id)

                selector = SelectIndexedAtoms(kept_ats_ids | surr_ats_ids)
                correction_file_path = f'{self._correction_dir_path}/level{correction_level}'
                io.save(correction_file_path + '.pdb', selector)
                system(f'pdb2pqr30 --noopt --nodebump --pdb-output {correction_file_path}out.pdb '
                       f'{correction_file_path}.pdb {self._correction_dir_path}/delete.pqr '
                       f'--titration-state-method propka --with-ph 7.2;'
                       f'rm {self._correction_dir_path}/delete.pqr; rm {self._correction_dir_path}/delete.log')

                correction_attempt = Protein(correction_file_path + '.pdb')

                ...
                correction_level += 1


class StructureCorrector:
    def __init__(self, path: str):
        self._path = path
        self._protein = Protein(path)

        if self._protein.status == 'CORRECT':
            print(f'OK: No error found in {self._protein.name}.')
        else:
            self._protein.execute_correction()

        # correction_dir_path = f"{dirname(self._path)}/{basename(self._path)[3:-16]}_correction"
        # rmtree(correction_dir_path)


StructureCorrector('/home/l/pycharmprojects/sicc_af/bordel/AF-Q9Y7W4-F1-model_v4.pdb') # rozstrelený H bez clashu
# StructureCorrector('/home/l/pycharmprojects/sicc_af/bordel/AF-A0A1D8PTL3-F1-model_v4.pdb') # clash 2 W