from collections import defaultdict
from os import mkdir, system
from os.path import basename, dirname, isdir
from re import sub

from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select
from rdkit.Chem import MolFromPDBFile, rdchem
from sklearn.cluster import AgglomerativeClustering

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
        self.correct = True
        self._chain_error = False


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
            self.correct = False
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
            self.correct = True

    def get_kept_ats_ids(self,
                         correction_level:  int) -> set[int]:
        """Returns ids of atoms to be copied into a correction file."""

        # select atoms to be kept
        ats_ids = set()
        for atom in self.atoms.values():
            # leave out erroneous atoms
            if atom.correct:
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


class Cluster:
    def __init__(self,
                 cluster: list):
        self.ids = cluster
        self.correct = False
        self.size = len(cluster)


class Protein:
    def __init__(self,
                 path: str):
        """The protein is loaded from a PDB file. Error bonds are detected and corresponding atoms and residues are
        noted."""

        self.correct = True
        self.clusters = None
        self.name = basename(path)[3:-16]
        self._residues = {}
        self._path = path
        self._error_residues = {}
        self._chain_errors = False
        self._correction_dir_path = f'{dirname(self._path)}/{self.name}_correction'
        self._correction_level = 0
        self._CLUSTERING_DISTANCE = 20

        # load RDKit molecule from PDB file
        try:
            rdkit_molecule = MolFromPDBFile(self._path,
                                            removeHs=False,
                                            sanitize=False)
        except KeyError:
            exit(f"ERROR! File at {self._path} is not a valid PDB file.\n")

        # make a dictionary of all protein's atoms {atom_id : Atom}, ignoring hydrogens
        atoms_dict = {rdkit_atom.GetPDBResidueInfo().GetSerialNumber(): Atom(rdkit_atom)
                      for rdkit_atom in rdkit_molecule.GetAtoms()
                      if rdkit_atom.GetPDBResidueInfo().GetName().strip()[0] != 'H'}

        # make a set of bonded atoms' NEF names for each atom & check for interresidual clashes
        # (check bonds detected by RDKit for atoms pairs)
        err_ats_set = set()
        chain_err_ats_set = set()
        for bond in rdkit_molecule.GetBonds():
            # ignore bonds containing hydrogen
            a1_nef_name = bond.GetBeginAtom().GetPDBResidueInfo().GetName().strip()
            a2_nef_name = bond.GetEndAtom().GetPDBResidueInfo().GetName().strip()
            if a1_nef_name[0] == 'H' or a2_nef_name[0] == 'H':
                continue

            a1_id = bond.GetBeginAtom().GetPDBResidueInfo().GetSerialNumber()
            a2_id = bond.GetEndAtom().GetPDBResidueInfo().GetSerialNumber()
            atom1 = atoms_dict[a1_id]
            atom2 = atoms_dict[a2_id]

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
                atom1._chain_error = True
                atom2._chain_error = True

            # else mark atoms as erroneous, if they are not chain atoms
            else:
                self.correct = False
                if a1_nef_name not in {'N', 'C', 'CA'}:
                    err_ats_set.add(atom1)
                    atom1.correct = False
                if a2_nef_name not in {'N', 'C', 'CA'}:
                    err_ats_set.add(atom2)
                    atom2.correct = False

        # check if the atom is bonded to expected atoms
        ress_ids = {atom.res_id for atom in atoms_dict.values()}
        last_res_id = max(ress_ids)
        for atom in atoms_dict.values():
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
                            atom._chain_error = True
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
                self.correct = False
                atom.correct = False
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
        print(f'correction of {self.name}...')
        # load Biopython structure from the PDB file
        try:
            bio_structure = PDBParser(QUIET=True).get_structure('protein', self._path)
        except KeyError:
            exit(f"ERROR! File at {self._path} is not a valid PDB file.\n")

        # this class will be used for cutting out clusters for correction into new files
        class SelectIndexedAtoms(Select):
            """Stores atom ids and uses them within the PDBIO class to select atoms to be copied to the new file
            based on their id."""

            def __init__(self):
                super().__init__()
                self.indices = set()

            def accept_atom(self, atom):
                """Overriding the original method (see BioPython documentation)."""
                if atom.get_serial_number() in self.indices:
                    return 1
                else:
                    return 0

            def update_indices(self, indices):
                self.indices = indices

        # cluster residues with regard to their centres of geometry
        bio_residues = {residue.id[1]: residue for residue in bio_structure.get_residues()}
        if len(self._error_residues) == 1:
            err_ress_ids_clusters = list([self._error_residues.keys()])
        else:
            err_ress_centres = [bio_residues[res_id].center_of_mass(geometric=True)
                                for res_id in sorted(self._error_residues.keys())]
            clustering_engine = AgglomerativeClustering(n_clusters=None,
                                                        distance_threshold=self._CLUSTERING_DISTANCE).fit(err_ress_centres)
            err_ress_ids_clusters = [[] for _ in range(clustering_engine.n_clusters_)]
            for cluster_id, res_id in zip(clustering_engine.labels_,
                                          sorted(self._error_residues.keys())):
                err_ress_ids_clusters[cluster_id].append(res_id)
        self.clusters = [Cluster([*cluster]) for cluster in err_ress_ids_clusters]

        # execute correction over individual clusters
        kdtree = NeighborSearch(list(bio_structure.get_atoms()))
        surr_ats_ids = set()
        for cluster in self.clusters:
            # get surrounding residues' atoms' ids
            for err_res_id in cluster.ids:
                for residue in kdtree.search(center=bio_residues[err_res_id].center_of_mass(geometric=True),
                                             radius=self._CLUSTERING_DISTANCE,
                                             level='R'):
                    res_id = residue.id[1]
                    if res_id not in self._error_residues.keys():
                        surr_ats_ids.update(self._residues[res_id].atoms.keys())

            # ensure existence of a directory for the protein's correction
            if not isdir(self._correction_dir_path):
                mkdir(self._correction_dir_path)

            # try iterations of correction
            cluster_err_ress = {res_id: self._residues[res_id] for res_id in cluster.ids}
            max_error_distance = max({res.min_err_distance if res.min_err_distance else 1
                                      for res in cluster_err_ress.values()})
            correction_level = 1
            io = PDBIO()
            io.set_structure(bio_structure)
            selector = SelectIndexedAtoms()
            correction_attempt = self
            while not all([correction_attempt._residues[res_id].correct for res_id in cluster_err_ress.keys()]):
                if correction_level > max_error_distance:
                    break

                # cut out the cluster's atoms into a correction file
                kept_ats_ids = set()
                for res_id in cluster.ids:
                    kept_ats_ids.update(self._residues[res_id].get_kept_ats_ids(correction_level))
                selector.update_indices(kept_ats_ids | surr_ats_ids)
                correction_file_path = f'{self._correction_dir_path}/level{correction_level}'
                io.save(correction_file_path + '.pdb', selector)

                # correct by propka
                system(f'pdb2pqr30 --noopt --pdb-output {correction_file_path}out.pdb '
                       f'{correction_file_path}.pdb {self._correction_dir_path}/delete.pqr '
                       f'--titration-state-method propka --with-ph 7.2  2>/dev/null;'
                       f'rm {self._correction_dir_path}/delete.pqr; rm {self._correction_dir_path}/delete.log')

                # check correction successfulness
                correction_attempt = Protein(correction_file_path + 'out.pdb')

                correction_level += 1

            else:
                for res_id in cluster.ids:
                    self._residues[res_id].correct = True
                cluster.correct = True

        if all([cluster.correct for cluster in self.clusters]):
            self.correct = True

        # print out results
        for cluster in self.clusters:
            for i in range(cluster.size):
                print(f'{cluster.ids[i]}: {'success' if cluster.correct else 'failure'}', end='')
                if i < cluster.size - 1:
                    print(', ', end='')
        print('')



path = '/home/l/pycharmprojects/sicc_af/bordel/AF-Q9Y7W4-F1-model_v4.pdb' # rozstrelený H bez clashu 368
path = '/home/l/pycharmprojects/sicc_af/bordel/AF-A0A1D8PTL3-F1-model_v4.pdb' # clash 2 W 538, 542
protein = Protein(path)

if protein.correct:
    print(f'OK: No error found in {protein.name}.')
else:
    protein.execute_correction()
    if protein.correct:
        print(f'CORRECTION SUCCESSFUL for {protein.name}')

# correction_dir_path = f"{dirname(self._path)}/{basename(self._path)[3:-16]}_correction"
# rmtree(correction_dir_path)



