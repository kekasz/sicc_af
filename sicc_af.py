from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, system, replace
from os.path import basename, dirname, isfile, splitext
from re import sub
from shutil import rmtree
from tabulate import tabulate

from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select
from rdkit.Chem import MolFromPDBFile, rdchem
from sklearn.cluster import AgglomerativeClustering

# from shutil import rmtree
from sicc_af_data import archetypes, distance, proline_distance


def load_arguments():
    parser = ArgumentParser()
    parser.add_argument('input_PDB_file',
                        type    = str,
                        help    = 'PDB file with structure to be corrected.'
                        )
    parser.add_argument('output_PDB_file',
                        type    = str,
                        nargs   = '?',
                        default = None,
                        help    = 'Path for the corrected structure file. (optional)'
                        )
    parser.add_argument('-l', '--logger',
                        type    = str,
                        nargs   = '?',
                        default = None,
                        help    = 'Path for the logging file.')
    parser.add_argument('-d', '--delete_auxiliary_files',
                        action  = 'store_true',
                        help    = 'Delete all auxiliary files. (recommended)'
                        )
    parser.add_argument('-s', '--silent',
                        action  = 'store_true',
                        help    = 'Reduce output. (use only if you know what you are doing)'
                        )

    parameters = parser.parse_args()
    return parameters

def print_output(message, silent):
    if not silent:
        print(message)

def save_log(message, log_file):
    with open(log_file, mode='a') as log_file:
        log_file.write(message)


class Atom:
    def __init__(self,
                 rdkit_atom:    rdchem.Atom):
        rdkit_atom_info = rdkit_atom.GetPDBResidueInfo()
        self.res_id = rdkit_atom_info.GetResidueNumber()
        self.res_name = rdkit_atom_info.GetResidueName()
        self.nef_name = rdkit_atom_info.GetName().strip()
        self.confidence = rdkit_atom_info.GetTempFactor()
        self.id = rdkit_atom_info.GetSerialNumber()
        self.bonded_ats_nef_names = set()
        self.correct = True
        self._chain_correct = True


class Residue:
    def __init__(self,
                 atoms_dict:        dict[int, Atom],
                 err_atoms_dict:    dict[int, Atom] = None):
        init_atom = list(atoms_dict.values())[0]
        self._id = init_atom.res_id
        self._name = init_atom.res_name
        self._err_atoms_dict = err_atoms_dict
        self.confidence = init_atom.confidence
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
                self.min_err_distance = 1
        else:
            self.correct = True

    def get_kept_ats_ids(self,
                         correction_level:  int) -> set[int]:
        """Returns ids of atoms to be copied into a correction file."""

        # select atoms to be kept
        ats_ids = set()
        # (case with only oxygen errors)
        if not self._err_atoms_dict:
            ats_ids = {atom.id for atom in self.atoms.values() if atom.correct}
        else:
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
                 cluster            : dict.keys,
                 surr_ress_ids      : set[int],
                 surr_ats_ids       : set[int],
                 protein_error_ress : dict[int, Residue]):
        self.ids            : list[int] = sorted(list(cluster))
        self.string_ids     : list[str] = list(map(str, self.ids))
        self.correct = False
        self.size = len(cluster)
        self.file = None
        self.surr_ids       : set[int] = surr_ress_ids                      # ids of surrounding residues
        self.surr_ats_ids   : set[int] = surr_ats_ids
        self.confidences = [protein_error_ress[res_id].confidence for res_id in self.ids]


class Protein:
    def __init__(self,
                 path:                      str,
                 corrected_path:            str  = None,
                 logger                     = None,
                 delete_auxiliary_files:    bool = True,
                 silent:                    bool = False):
        """The protein is loaded from a PDB file. Error bonds are detected and corresponding atoms and residues are
        noted."""

        self.chain_correct = True
        self.clusters : list[Cluster] = []
        self.correct = True
        self._delete_auxiliary_files = delete_auxiliary_files
        self.filename = basename(path)
        self._silent = silent
        self._CLUSTERING_DISTANCE = 10
        self._SURROUNDINGS_DISTANCE = 17
        self.path = path
        self._corrected_path = corrected_path
        self._name = basename(path)[3:-16]
        self._correction_dir = f'{dirname(self.path)}/sicc_af/{self._name}_correction'
        if not logger:
            self.logger = self._correction_dir + '/log.txt'
        else:
            self.logger = logger
        self._correction_level = 0
        self._error_residues = {}
        self._residues = {}

        # load RDKit molecule from PDB file
        rdkit_molecule = MolFromPDBFile(self.path,
                                        sanitize    = False,
                                        removeHs    = False)
        if rdkit_molecule is None:
            print(f'ERROR! File at {self.path} cannot be loaded by RDKit (possibly not a valid PDB file).')
            exit(4)

        # make a dictionary of all protein's atoms {atom_id : Atom}, ignoring hydrogens
        atoms_dict = {rdkit_atom.GetPDBResidueInfo().GetSerialNumber(): Atom(rdkit_atom)
                      for rdkit_atom in rdkit_molecule.GetAtoms()
                      if rdkit_atom.GetPDBResidueInfo().GetName().strip()[0] != 'H'}

        # make a set of bonded atoms' NEF names for each atom & check for interresidual clashes
        # (check bonds detected by RDKit for atoms pairs)
        err_ats_set = set()
        chain_err_ats_set = set()
        for bond in rdkit_molecule.GetBonds():
            a1_rdkit_atom_info = bond.GetBeginAtom().GetPDBResidueInfo()
            a2_rdkit_atom_info = bond.GetEndAtom().GetPDBResidueInfo()

            # ignore bonds containing hydrogen
            a1_nef_name = a1_rdkit_atom_info.GetName().strip()
            a2_nef_name = a2_rdkit_atom_info.GetName().strip()
            if a1_nef_name[0] == 'H' or a2_nef_name[0] == 'H':
                continue

            # load additional information
            a1_id = a1_rdkit_atom_info.GetSerialNumber()
            a2_id = a2_rdkit_atom_info.GetSerialNumber()
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
                self.chain_correct = False
                chain_err_ats_set.add(atom1)
                chain_err_ats_set.add(atom2)
                atom1._chain_correct = False
                atom2._chain_correct = False

            # else mark atoms as erroneous, if they are not chain atoms
            else:
                self.correct = False
                if a1_nef_name not in {'N', 'C', 'CA'}:
                    err_ats_set.add(atom1)
                    atom1.correct = False
                if a2_nef_name not in {'N', 'C', 'CA'}:
                    err_ats_set.add(atom2)
                    atom2.correct = False

        # check if atoms are bonded to expected atoms
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
                # if oxygen, mark as erroneous
                if at_nef_name in {'O', 'OXT'}:
                    erroneous = True

                else:
                    # mark chain errors
                    if (at_nef_name in {'N', 'C', 'CA'} and
                            {'N', 'C', 'CA'} & archetype != {'N', 'C', 'CA'} & bonded_ats_nef_names):
                        self.chain_correct = False
                        chain_err_ats_set.add(atom)
                        atom._chain_correct = False

                    # find side chain errors, thus ignore (main) chain atoms
                    if at_nef_name not in {'N', 'C', 'CA'}:
                        # if there are any extra atoms bonded, mark erronous
                        if bonded_ats_nef_names - archetype:
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

        self._residues = {res_id: Residue(atoms_dict        = res_ats_dict,
                                          err_atoms_dict    = err_ress_ats_dicts_dict[res_id] if res_id in err_ress_ats_dicts_dict.keys()
                                                                                              else None)
                          for res_id, res_ats_dict in ress_ats_dicts_dict.items()}
        self._error_residues = {res_id: self._residues[res_id]
                                for res_id in err_ress_ats_dicts_dict.keys()}
        self.chain_error_ress = {atom.res_id for atom in chain_err_ats_set}

    @staticmethod
    def _make_ress_ats_dicts_dict(atoms: set[Atom]) -> dict[int, dict[int, Atom]]:
        ress_ats_dicts_dict = defaultdict(dict)
        for atom in atoms:
            ress_ats_dicts_dict[atom.res_id][atom.id] = atom
        return dict(ress_ats_dicts_dict)

    def prepare_log_directory(self):
        makedirs(name       = dirname(self.logger),
                 exist_ok   = True)

    def execute_correction(self):
        if not self._corrected_path:
            self._corrected_path = f'{dirname(self.path)}/sicc_af/{splitext(self.filename)[0]}_corrected.pdb'
        else:
            self._corrected_path = f'{dirname(self.path)}/sicc_af/{self._corrected_path}'

        # load Biopython structure from the PDB file
        bio_structure = PDBParser(QUIET=True).get_structure('protein', self.path)

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

        # cluster erroneous residues with regard to their centres of geometry
        bio_residues = {residue.id[1]: residue for residue in bio_structure.get_residues()}
        if len(self._error_residues) == 1:
            err_ress_ids_clusters = [list(self._error_residues.keys())]
        else:
            err_ress_centres = [bio_residues[res_id].center_of_mass(geometric=True)
                                for res_id in sorted(self._error_residues.keys())]
            clustering_engine = AgglomerativeClustering(n_clusters=None,
                                                        distance_threshold=self._CLUSTERING_DISTANCE).fit(err_ress_centres)
            err_ress_ids_clusters : list[list[int]] = [[] for _ in range(clustering_engine.n_clusters_)]
            for cluster_id, res_id in zip(clustering_engine.labels_,
                                          sorted(self._error_residues.keys())):
                err_ress_ids_clusters[cluster_id].append(res_id)

        # get surrounding residues for each cluster
        kdtree = NeighborSearch(list(bio_structure.get_atoms()))
        cluster_surr_ress = []
        cluster_surr_ats  = []
        for cluster in err_ress_ids_clusters:
            surr_ress_ids = set()
            surr_ats_ids = set()
            for err_res_id in cluster:
                for residue in set(kdtree.search(center=bio_residues[err_res_id].center_of_mass(geometric=True),
                                                 radius=self._SURROUNDINGS_DISTANCE,
                                                 level='R')):
                    res_id = residue.id[1]
                    if res_id not in self._error_residues.keys():
                        surr_ress_ids.add(res_id)
                        surr_ats_ids.update(self._residues[res_id].atoms.keys())

            cluster_surr_ress.append(surr_ress_ids)
            cluster_surr_ats.append(surr_ats_ids)

        # make cluster objects
        self.clusters = [Cluster(cluster, surr_ress, surr_ats, self._error_residues) for cluster, surr_ress, surr_ats
                         in zip(err_ress_ids_clusters, cluster_surr_ress, cluster_surr_ats)]

        # execute correction over individual clusters
        for i, cluster in enumerate(self.clusters):
            print(f'correcting cluster {i}: residues {'+'.join(cluster.string_ids)} {cluster.confidences}')


            # ensure existence of a directory for the protein's correction
            correction_dir = f'{self._correction_dir}/cluster_{i}'
            makedirs(name       = correction_dir,
                     exist_ok   = True)

            # try iterations of correction
            cluster_err_ress = {res_id: self._residues[res_id] for res_id in cluster.ids}
            max_error_distance = max({res.min_err_distance if res.min_err_distance else 1
                                      for res in cluster_err_ress.values()})
            correction_level = 1
            io = PDBIO()
            io.set_structure(bio_structure)
            selector = SelectIndexedAtoms()
            correction_attempt = self
            output_file = ''
            pdb2pqr_problem = False
            pdb2pqr_nodebump = '--nodebump'
            debump_flag = ''
            while not all([correction_attempt._residues[res_id].correct for res_id in cluster_err_ress.keys()]):
                if correction_level > max_error_distance or pdb2pqr_problem:
                    if not pdb2pqr_problem and pdb2pqr_nodebump:
                        print_output('trying with debump...', self._silent)
                        correction_level = 1
                        pdb2pqr_nodebump = ''
                        debump_flag = '_d'
                    else:
                        print_output('failure', not self._silent)
                        break

                cutout_file = f'{correction_dir}/level{correction_level}.pdb'
                output_file = f'{correction_dir}/level{correction_level}{debump_flag}_out.pdb'
                pdb2pqr_log = f'{correction_dir}/level{correction_level}{debump_flag}_pdb2pqr.log'

                if pdb2pqr_nodebump:
                    # cut out the cluster's atoms into a correction file
                    kept_ats_ids = set()
                    for res_id in cluster.ids:
                        kept_ats_ids.update(self._residues[res_id].get_kept_ats_ids(correction_level))
                    surr_ats_ids = cluster.surr_ats_ids
                    selector.update_indices(kept_ats_ids | surr_ats_ids)
                    io.save(cutout_file, selector)

                # correct by propka
                system(f'pdb2pqr30 {pdb2pqr_nodebump} --noopt --pdb-output {output_file} '
                       f'{cutout_file} {correction_dir}/delete.pqr '
                       f'--titration-state-method propka --with-ph 7.2 2>{pdb2pqr_log};'
                       f'rm {correction_dir}/delete.*;')

                with open(pdb2pqr_log, mode='r') as log_file:
                    for line in log_file:
                        if line[0:39] == 'ERROR:This PDB file is missing too many':
                            print_output(f'ERROR: The level {correction_level} cutout file' + line[14:-1], False)
                            save_log(f'ERROR: The level {correction_level} cutout file' + line[14:], self.logger)
                            pdb2pqr_problem = True
                            break

                # check correction successfulness
                if not pdb2pqr_problem:
                    correction_attempt = Protein(output_file)

                correction_level += 1

            else:
                print_output('success', not self._silent)
                for res_id in cluster.ids:
                    self._residues[res_id].correct = True

                cluster.correct = True
                cluster.file = self._correction_dir + f'/cluster_{i}{debump_flag}.pdb'
                replace(src = output_file,
                        dst = cluster.file)

        if all([cluster.correct for cluster in self.clusters]):
            self.correct = True

        # write the new corrected file
        if any([cluster.correct for cluster in self.clusters]):
            # select whole corrected clusters (even with surrounding residues)
            correction_ress = set()
            for cluster in self.clusters:
                if cluster.correct:
                    correction_ress.update(set(cluster.ids))
                    correction_ress.update(cluster.surr_ids)

            written_correction_ress = []
            with open(file  = self.path,
                      mode  = 'r') as original_file:
                with open(file  = self._corrected_path,
                          mode  = 'w') as corrected_file:
                    for line in original_file:
                        if len(line) > 4 and line[0:4] == 'ATOM':
                            res_id = int(line[22:26])
                            if res_id == 62:
                                pass
                            if res_id in correction_ress:
                                if res_id not in written_correction_ress:
                                    at_id = int(line[6:11])
                                    res_written = False
                                    for cluster in (cluster for cluster in self.clusters if cluster.correct):
                                        if res_written:
                                            break
                                        if (res_id in cluster.ids
                                                or res_id in cluster.surr_ids and res_id not in self._error_residues):
                                            if not cluster.file:
                                                pass
                                            with open(file  = cluster.file,
                                                      mode  = 'r') as cluster_file:
                                                for line_c in cluster_file:
                                                    if line_c[0:4] == 'ATOM' and int(line_c[22:26]) == res_id and 'H' not in line_c[12:14]:
                                                        line_c = f'{line_c[0:6]}{at_id:5}{line_c[11:]}'
                                                        corrected_file.write(line_c)
                                                        at_id += 1
                                                written_correction_ress.append(res_id)
                                                res_written = True

                                continue
                        corrected_file.write(line)

        # delete auxiliary files (if demanded)
        if self._delete_auxiliary_files:
            rmtree(self._correction_dir)



class StructureIntegrityCheckerAndCorrector:
    # noinspection PyPep8Naming
    def __init__(self,
                 input_PDB_file         : str,
                 output_PDB_file        : str,
                 logger                 : str,
                 delete_auxiliary_files : bool,
                 silent                 : bool = False):

        # control usability of files
        if not isfile(input_PDB_file):
            print(f'ERROR! File at {input_PDB_file} does not exist!')
            exit(2)
        if input_PDB_file == output_PDB_file:
            print(f'ERROR! Input and output files cannot be the same.')
            exit(3)

        self._input_PDB_file = input_PDB_file
        self._output_PDB_file = output_PDB_file
        self._logger = logger
        self._delete_auxiliary_files = delete_auxiliary_files
        self._silent = silent

    def process_structure(self):
        print_output('loading file...', self._silent)
        protein = Protein(self._input_PDB_file,
                          self._output_PDB_file,
                          self._logger,
                          self._delete_auxiliary_files,
                          self._silent)
        print_output(f'{protein.filename} loaded', self._silent)

        if protein.correct and protein.chain_correct:
            print_output(f'OK: No error found in {protein.filename}.', self._silent)
        else:
            if type(protein.logger) == str:
                protein.prepare_log_directory()
            save_log(f'{protein.path}:\n', protein.logger)
            correction_executed = False
            if not protein.correct:
                protein.execute_correction()
                correction_executed = True
                listed_results = [[', '.join(cluster.string_ids), 'success' if cluster.correct else 'failure']
                                  for cluster in protein.clusters]
                table = tabulate(tabular_data   = listed_results,
                                 headers        = ['Clustered error residues', 'Correction result'],
                                 colalign       = ['left', 'left'])
                print_output(f'RESULTS:\n{table}', self._silent)
                save_log(f'{table}\n', protein.logger)
                if protein.correct:
                    print_output(f'CORRECTION SUCCESSFUL for {protein.filename}', self._silent)
                elif all([not cluster.correct for cluster in protein.clusters]):
                    print_output(f'CORRECTION FAILED for {protein.filename}', self._silent)
                else:
                    print_output(f'CORRECTION partially successful for {protein.filename}', self._silent)
            if not protein.chain_correct:
                print_output(f'In {protein.filename}, chain errors were found in these residues:', self._silent)
                print_output(f'chain:', not self._silent)
                print(', '.join(list(map(str, sorted(list(protein.chain_error_ress))))))
                if correction_executed:
                    save_log('\n', protein.logger)
                save_log(f'Chain errors\n--------------------------\n{', '.join(list(map(str, sorted(list(protein.chain_error_ress)))))}\n', protein.logger)
                print_output('INFO: Correction of chain errors is never performed.', self._silent)

            save_log('\n', protein.logger)


if __name__ == '__main__':
    args = load_arguments()
    StructureIntegrityCheckerAndCorrector(args.input_PDB_file,
                                          args.output_PDB_file,
                                          args.logger,
                                          args.delete_auxiliary_files,
                                          args.silent).process_structure()
