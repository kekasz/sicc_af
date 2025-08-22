from argparse import ArgumentParser
from collections import defaultdict
from numpy import array, float32
from numpy.linalg import norm as euclidean_distance
from numpy.typing import NDArray
from os import makedirs, system, replace
from os.path import abspath, basename, dirname, isfile, splitext
from re import sub
from sys import exit
from shutil import rmtree
from tabulate import tabulate

from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select
from Bio.PDB.Residue import Residue as BioResidue
from rdkit.Chem import MolFromPDBFile
from rdkit.Chem.rdchem import Atom as RDAtom
from sklearn.cluster import AgglomerativeClustering

from data_sicc_af import archetypes, distance, proline_distance, residues_cycles_atoms

# for batch run use
try:
    from logger import Logger
except ModuleNotFoundError:
    pass


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
    parser.add_argument('-l', '--log_file',
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
                 rdkit_atom: RDAtom):
        self.rdkit_atom_info = rdkit_atom.GetPDBResidueInfo()
        self.res_id = self.rdkit_atom_info.GetResidueNumber()
        self.res_name = self.rdkit_atom_info.GetResidueName()
        self.name = self.rdkit_atom_info.GetName().strip()
        self.id = self.rdkit_atom_info.GetSerialNumber()
        self.bonded_ats_names: dict[str, Atom] = {}
        self.correct_in_side_chain = True
        self.correct_in_backbone = True


class Residue:
    def __init__(self,
                 atoms              : dict[int, Atom],
                 missing_bonds      : set[tuple[str, str]],
                 side_chain_err_ats : dict[int, Atom] = None):

        self.atoms = atoms
        self.atoms_by_name          : dict[str, Atom]               = {atom.name: atom for atom in atoms.values()}
        self.cycle_centres          : tuple[NDArray[float32], ...]  = (array((0, 0, 0)),)
        self.side_chain_correct     : bool                          = False if side_chain_err_ats else True
        self.side_chain_err_atoms                                   = side_chain_err_ats if side_chain_err_ats else {}
        self._ox_err_ats            : dict[int, Atom]               = {}

        init_atom = list(atoms.values())[0]
        self.confidence = init_atom.rdkit_atom_info.GetTempFactor()
        self.id = init_atom.res_id
        self.name = init_atom.res_name

        # process missing bond atoms
        self.missing_bonds: dict[tuple[Atom, Atom], NDArray[float32][3]] = {}
        for missing_bond in missing_bonds:
            atom1 = self.atoms_by_name[missing_bond[0]]
            atom2 = self.atoms_by_name[missing_bond[1]]

            self.missing_bonds[(atom1, atom2)] = array((0, 0, 0))

        # detect oxygen errors
        if side_chain_err_ats:
            # separate backbone oxygen errors (they need to be treated separatedly)
            for at_id, atom in side_chain_err_ats.items():
                if atom.name[0] == 'O':
                    self._ox_err_ats[at_id] = atom
            for at_id in self._ox_err_ats.keys():
                    del self.side_chain_err_atoms[at_id]

            self.min_err_distance = None
            self.calculate_minimal_error_distance()

    def get_kept_ats_ids(self, correction_level:  int) -> set[int]:
        """Returns ids of atoms to be copied into a correction file."""

        # select atoms to be kept
        ats_ids = set()
        # for the case there are only oxygen errors
        if not self.side_chain_err_atoms:
            ats_ids = {atom.id for atom in self.atoms.values() if atom.correct_in_side_chain}
        # for every other case
        else:
            for atom in self.atoms.values():
                # leave out erroneous atoms, including erroneous oxygens
                if atom.correct_in_side_chain:
                    # leave out atoms further in residue than any of the erroneous ones
                    if len(sub(r'[^a-zA-Z]', '', atom.name)) == 2 and atom.name != 'CA':
                        # purvey proline particularities
                        if (self.name == 'PRO'
                                and proline_distance[atom.name[1]] <= self.min_err_distance - correction_level):
                            ats_ids.add(atom.id)
                        elif distance[atom.name[1]] <= self.min_err_distance - correction_level:
                            ats_ids.add(atom.id)
                    else:
                        ats_ids.add(atom.id)

        return ats_ids

    def calculate_minimal_error_distance(self):
        """Calculate "mininal error distance", the distance of the residue's closest side chain error atom to the backbone."""
        if self.side_chain_err_atoms:
            # purvey proline particularities
            if self.name == 'PRO':
                self.min_err_distance = min({proline_distance[atom.name[1]]
                                             for atom in self.side_chain_err_atoms.values()})
            else:
                self.min_err_distance = min({distance[atom.name[1]]
                                             for atom in self.side_chain_err_atoms.values()})
        else:
            self.min_err_distance = 1

    def mark_cycle_erroneous(self, cycle_n):
        self.side_chain_correct = False

        for at_name in residues_cycles_atoms[self.name][cycle_n]:
            atom = self.atoms_by_name[at_name]
            if self.name == 'PRO' and at_name in {'N', 'CA'}:
                atom.correct_in_backbone = False
            else:
                atom.correct_in_side_chain = False
                self.side_chain_err_atoms[atom.id] = atom

        self.calculate_minimal_error_distance()


class Cluster:
    def __init__(self,
                 err_ress           : set[Residue],
                 surr_ress          : set[Residue],
                 err_cyclic_ress    : set[Residue]):

        self.correct_side_chain_wise = False
        self.debump = False
        self.file = None
        self.iterations = 0
        self.pdb2pqr_error = False
        self.surr_ats_ids       : set[int] = {atom for res in surr_ress for atom in res.atoms.keys()}

        # determine error type (0 = other error type, 1 = HIS type 1, 2 = HIS type 2, +3 = cycle clash, +6 = proline cycle clash)
        self.error_type = 0
        erroneous_histidines = [res for res in err_ress if res.name == 'HIS']
        if len(erroneous_histidines) == 1:
            err_h = erroneous_histidines[0]
            err_ats_names = {atom.name for atom in err_h.side_chain_err_atoms.values()}
            if err_ats_names == {'CE1', 'NE2'} and all(atom.bonded_ats_names == {} for atom in err_h.side_chain_err_atoms.values()):
                self.error_type = 1
            elif ({(atom.name, tuple(at_name for at_name in atom.bonded_ats_names.keys()))
                  for atom in err_h.side_chain_err_atoms.values()}
                  ==
                  {('CE1', ('ND1',)), ('NE2', ())}):
                self.error_type = 2
        if err_cyclic_ress:
            if any(ress.name == 'PRO' for ress in err_cyclic_ress):
                self.error_type += 6
            else:
                self.error_type += 3

        self.err_ress           : dict[int, Residue]        = {res.id: res for res in err_ress}
        self.confidence_log     : list[tuple[int, float]]   = sorted(((res.id, res.confidence) for res in err_ress), key=lambda x: x[0])
        self.cycle_confid_log   : list[tuple[int, float]]   = sorted(((res.id, res.confidence) for res in err_cyclic_ress), key=lambda x: x[0])
        self.err_ress_ids_strs  : list[str]                 = list(map(str, [tup[0] for tup in self.confidence_log]))


class SelectIndexedAtoms(Select):
    """Stores atom ids and uses them within the PDBIO class to select atoms to be copied to the new file
    based on their id."""

    def __init__(self):
        super().__init__()
        self.indices = set()

    def accept_atom(self, atom_id):
        """Overriding the original method (see BioPython documentation)."""
        if atom_id.get_serial_number() in self.indices:
            return 1
        else:
            return 0

    def update_indices(self, indices):
        self.indices = indices


class Protein:
    def __init__(self,
                 path           : str,
                 corrected_path : str     = None,
                 correction_dir : str     = False,
                 silent         : bool    = False):
        """The protein is loaded from a PDB file. Error bonds are detected and corresponding atoms and residues are
        noted and clustered."""

        self.filename = basename(path)
        self.name = basename(path)[3:-16]
        self._silent = silent

        # load RDKit molecule from PDB file
        rdkit_molecule = MolFromPDBFile(path,
                                        sanitize    = False,
                                        removeHs    = False)
        if rdkit_molecule is None:
            print(f'ERROR: File at {path} cannot be loaded by RDKit (possibly not a valid PDB file).')
            exit(4)

        # make a dictionary of all protein's atoms {atom_id : Atom}, ignoring hydrogens
        atoms : dict[int, Atom] = {rdkit_atom.GetPDBResidueInfo().GetSerialNumber(): Atom(rdkit_atom)
                                   for rdkit_atom in rdkit_molecule.GetAtoms()
                                   if rdkit_atom.GetPDBResidueInfo().GetName().strip()[0] != 'H'}

        # make a set of bonded atoms' NEF names for each atom & check for interresidual clashes
        self.backbone_correct = True
        self.side_chains_correct = True
        backbone_err_ress   : set[tuple[int, ...]]  = set()
        side_chain_err_ats  : set[Atom]             = set()
        for bond in rdkit_molecule.GetBonds():
            a1_name = bond.GetBeginAtom().GetPDBResidueInfo().GetName().strip()
            a2_name = bond.GetEndAtom().GetPDBResidueInfo().GetName().strip()

            # ignore hydrogen atoms (added into correction files by propka)
            if a1_name[0] == 'H' or a2_name[0] == 'H':
                continue

            ats_names = {a1_name, a2_name}
            atom1 = atoms[bond.GetBeginAtom().GetPDBResidueInfo().GetSerialNumber()]
            atom2 = atoms[bond.GetEndAtom().GetPDBResidueInfo().GetSerialNumber()]

            # purvey disulphide bonds between cysteins sulfurs
            if ats_names == {'SG', 'SG'}:
                continue

            # load additional information
            a1_res_id = atom1.res_id
            a2_res_id = atom2.res_id

            # if the bond is within the same residue or is eupeptidic, add to the set, except for intraresidual backbone clashes
            if a1_res_id == a2_res_id:
                if ats_names == {'N', 'C'}:
                    backbone_err_ress.add((a1_res_id,))
                else:
                    atom1.bonded_ats_names[a2_name] = atom2
                    atom2.bonded_ats_names[a1_name] = atom1
                # else:
                #     # validate the bond if it is in a cycle
                #     res_name = atom1.res_name
                #     res_id = atom1.res_id
                #     bond_valid = True
                #     if res_name in cyclic_residues_names and ats_names in cyclic_bonds[res_name]:
                #         a1_bio = self._bio_structure[0]['A'][(' ', res_id, ' ')][a1_name]
                #         a2_bio = self._bio_structure[0]['A'][(' ', res_id, ' ')][a2_name]
                #         bond_length = (a1_bio - a2_bio).item()
                #         if res_name == 'PRO':
                #             if bond_length > 1.7:
                #                 bond_valid = False
                #         elif bond_length > 1.5:
                #             bond_valid = False
                #
                #     if bond_valid:
                #         atom1.bonded_ats_names.add(a2_name)
                #         atom2.bonded_ats_names.add(a1_name)

            elif ((a2_res_id - a1_res_id == 1 and (a1_name, a2_name) == ('C', 'N'))
                    or (a1_res_id - a2_res_id == 1 and (a1_name, a2_name) == ('N', 'C'))):
                atom1.bonded_ats_names[a2_name] = atom2
                atom2.bonded_ats_names[a1_name] = atom1

            # detect interresidual clashes of backbone atoms
            elif ats_names < {'N', 'C', 'CA'}:
                self.backbone_correct = False
                backbone_err_ress.add(tuple(sorted([a1_res_id, a2_res_id])))
                atom1.correct_in_backbone = False
                atom2.correct_in_backbone = False

            # else mark atoms as erroneous, if they are not backbone atoms
            else:
                self.side_chains_correct = False
                if a1_name not in {'N', 'C', 'CA'}:
                    side_chain_err_ats.add(atom1)
                    atom1.correct_in_side_chain = False
                if a2_name not in {'N', 'C', 'CA'}:
                    side_chain_err_ats.add(atom2)
                    atom2.correct_in_side_chain = False

        # check if atoms are bonded to expected atoms
        ress_ids = {atom.res_id for atom in atoms.values()}
        last_res_id = max(ress_ids)
        missing_bonds: set[tuple[int, tuple[str, str]]] = set()
        for atom in atoms.values():
            at_name = atom.name
            res_name = atom.res_name
            archetype: set[str] = archetypes[res_name][at_name]
            res_id = atom.res_id

            # purvey the initial nitrogen
            if res_id == 1 and at_name == 'N':
                archetype = archetype - {'C'}

            # purvey the terminal carbon and oxygen
            elif res_id == last_res_id:
                if at_name == 'C':
                    archetype = archetype - {'N'}
                    archetype.add('OXT')
                if at_name == 'OXT':
                    archetype = archetypes[res_name]['O']

            # purvey ends of local chains in correction files
            elif (at_name == 'N' and res_id-1 not in ress_ids
                  or at_name == 'C' and res_id+1 not in ress_ids):
                archetype = archetype - {'N', 'C'}

            # select atoms to be marked as erroneous
            bonded_ats_names = atom.bonded_ats_names.keys()
            correct_in_side_chain = True
            if bonded_ats_names != archetype:
                # if oxygen, mark as erroneous
                if at_name in {'O', 'OXT'}:
                    correct_in_side_chain = False

                else:
                    # note backbone errors
                    if at_name == 'CA' and not {'N', 'C'} <= bonded_ats_names:
                        self.backbone_correct = False
                        atom.correct_in_backbone = False
                        backbone_err_ress.add((res_id,))
                    elif at_name == 'N' and {'C'} & archetype != {'C'} & bonded_ats_names:
                        self.backbone_correct = False
                        atom.correct_in_backbone = False
                        backbone_err_ress.add((res_id - 1, res_id))

                    # find side chain errors, thus ignore backbone atoms
                    elif at_name not in {'N', 'C', 'CA'}:
                        # if there are any extra atoms bonded, mark erronous
                        if bonded_ats_names - archetype:
                            correct_in_side_chain = False

                        # purvey proline particularities
                        elif res_name == 'PRO' and at_name in {'CD', 'CG'}:
                            if at_name == 'CD' and 'N' not in bonded_ats_names:
                                correct_in_side_chain = False
                            elif at_name == 'CG':
                                correct_in_side_chain = False

                        # ignore the atoms that are the "very last correct" in the residue
                        elif not all(distance[missing_at_nef_name[1]] > distance[at_name[1]]
                                     for missing_at_nef_name in archetype - bonded_ats_names
                                     if len(sub(r'[^a-zA-Z]', '', missing_at_nef_name)) == 2): # possible number at the end of the nef name (e.g. CE1 and NE2 in HIS)
                            correct_in_side_chain = False

                if not correct_in_side_chain:
                    self.side_chains_correct = False
                    atom.correct_in_side_chain = False
                    side_chain_err_ats.add(atom)

                # note missing bonds
                for bonded_at_name in archetype - bonded_ats_names:
                    if (at_name, bonded_at_name) != ('N', 'C'):
                        # noinspection PyTypeChecker
                        missing_bonds.add((res_id, tuple(sorted([at_name, bonded_at_name]))))

        ress_ats = self._make_ress_ats_dicts_dict(set(atoms.values()))
        side_chain_err_ress_ats = self._make_ress_ats_dicts_dict(side_chain_err_ats)
        self.residues: dict[int, Residue] = {res_id: Residue(atoms                = res_ats_dict,
                                                             missing_bonds        = {bond[1] for bond in missing_bonds if bond[0] == res_id},
                                                             side_chain_err_ats   = side_chain_err_ress_ats[res_id] if res_id in side_chain_err_ress_ats.keys()
                                                                                                                    else None)
                                             for res_id, res_ats_dict in ress_ats.items()}

        # prepare for correction if necessary
        if not self.side_chains_correct:
            self._bio_structure = PDBParser(QUIET=True).get_structure('protein', path)
            clustering_distance = 10
            self._correction_dir = correction_dir
            self._final_file_path = corrected_path
            cycle_distance = 1
            self.log = ''
            self.pdb2pqr_errors_log: list[tuple[int, list[tuple[int, float]]]] = []
            surroundings_distance = 15

            # cluster erroneous residues with regard to their centres of geometry
            bio_residues : dict[int, BioResidue] = {residue.id[1]: residue
                                                    for residue in self._bio_structure.get_residues()}
            side_chain_err_ress_clusters: list[set[Residue]]
            if len(side_chain_err_ress_ats) == 1:
                side_chain_err_ress_clusters = [{self.residues[list(side_chain_err_ress_ats.keys())[0]]}]
            else:
                side_chain_err_ress_centres = [bio_residues[res_id].center_of_mass(geometric=True)
                                               for res_id in sorted(side_chain_err_ress_ats.keys())]
                clustering_engine = AgglomerativeClustering(n_clusters          = None,
                                                            distance_threshold  = clustering_distance).fit(side_chain_err_ress_centres)
                side_chain_err_ress_clusters = [set() for _ in range(clustering_engine.n_clusters_)]
                for cluster_id, res_id in zip(clustering_engine.labels_,
                                              sorted(side_chain_err_ress_ats.keys())):
                    side_chain_err_ress_clusters[cluster_id].add(self.residues[res_id])

            # get surrounding residues for each cluster
            kdtree = NeighborSearch(list(self._bio_structure.get_atoms()))
            clusters_surr_ress      : list[set[Residue]]    = []
            for cluster in side_chain_err_ress_clusters:
                surr_ress_ids = set()
                for side_chain_err_res in cluster:
                    for residue in set(kdtree.search(center = bio_residues[side_chain_err_res.id].center_of_mass(geometric=True),
                                                     radius = surroundings_distance,
                                                     level  = 'R')):
                        res_id = residue.id[1]
                        if res_id not in side_chain_err_ress_ats.keys():
                            surr_ress_ids.add(res_id)

                clusters_surr_ress.append({self.residues[res_id] for res_id in surr_ress_ids})

            # detect errors related to bonds stretching through a cycle
            clusters_err_cyclic_ress: list[set[Residue]] = []
            for cluster_err_ress, cluster_surr_ress in zip(side_chain_err_ress_clusters, clusters_surr_ress):
                # calculate the geometric centres of cycles
                for res in cluster_err_ress | cluster_surr_ress:
                    if res.name in residues_cycles_atoms.keys():
                        centres: list[NDArray[float32]] = []
                        for cycle in residues_cycles_atoms[res.name]:
                            bio_cycle_atoms = [self._bio_structure[0]['A'][(' ', res.id, ' ')][at_name]
                                           for at_name in cycle]
                            x = sum(atom.coord[0] for atom in bio_cycle_atoms)/len(bio_cycle_atoms)
                            y = sum(atom.coord[1] for atom in bio_cycle_atoms)/len(bio_cycle_atoms)
                            z = sum(atom.coord[2] for atom in bio_cycle_atoms)/len(bio_cycle_atoms)
                            centres.append(array((x, y, z)))
                        res.cycle_centres = tuple(centre for centre in centres)

                # calculate the geometric centres of missing (= overextended) bonds
                for res in [res for res in cluster_err_ress | cluster_surr_ress if res.missing_bonds]:
                    for at_pair in res.missing_bonds.keys():
                        bio_bond_at1 = self._bio_structure[0]['A'][(' ', at_pair[0].res_id, ' ')][at_pair[0].name]
                        bio_bond_at2 = self._bio_structure[0]['A'][(' ', at_pair[1].res_id, ' ')][at_pair[1].name]
                        x = (bio_bond_at1.coord[0] + bio_bond_at2.coord[0])/2
                        y = (bio_bond_at1.coord[1] + bio_bond_at2.coord[1])/2
                        z = (bio_bond_at1.coord[2] + bio_bond_at2.coord[2])/2
                        res.missing_bonds[at_pair] = array((x, y, z))

                # detect the errors based on coordinates
                cluster_cyclic_ress = {res for res in cluster_err_ress | cluster_surr_ress
                                       if res.name in residues_cycles_atoms.keys()}
                cluster_err_cyclic_ress = set()
                for res in cluster_err_ress | cluster_surr_ress:
                    for atoms, centre in res.missing_bonds.items():
                        for cyclic_res in cluster_cyclic_ress - {res}:
                            for i, cycle_centre in enumerate(cyclic_res.cycle_centres):
                                if euclidean_distance(centre - cycle_centre) < cycle_distance:
                                    cyclic_res.mark_cycle_erroneous(i)
                                    side_chain_err_ress_ats[cyclic_res.id] = cyclic_res.side_chain_err_atoms
                                    cluster_err_ress.add(cyclic_res)
                                    cluster_surr_ress.discard(cyclic_res)
                                    cluster_err_cyclic_ress.add(cyclic_res)
                                    if res.name == 'PRO':
                                        backbone_err_ress.add((cyclic_res.id,))
                clusters_err_cyclic_ress.append(cluster_err_cyclic_ress)

            # make cluster objects
            self.clusters: list[Cluster] = [Cluster(err_ress, surr_ress, err_cyclic_ress)
                                            for err_ress, surr_ress, err_cyclic_ress in zip(side_chain_err_ress_clusters, clusters_surr_ress, clusters_err_cyclic_ress)]

        # arrange errors
        self.backbone_err_ress  : list[int]             = sorted(res_id for err in backbone_err_ress for res_id in err)
        self.backbone_errors    : list[tuple[int, ...]] = sorted(backbone_err_ress, key = lambda x: x[0])
        self.side_chain_err_ress: dict[int, Residue]    = {res_id: self.residues[res_id]
                                                           for res_id in side_chain_err_ress_ats.keys()}

    @staticmethod
    def _make_ress_ats_dicts_dict(atoms: set[Atom]) -> dict[int, dict[int, Atom]]:
        ress_ats_dicts_dict = defaultdict(dict)
        for atom in atoms:
            ress_ats_dicts_dict[atom.res_id][atom.id] = atom
        return dict(ress_ats_dicts_dict)

    def execute_correction(self):
        """Execute correction over individual clusters"""

        io = PDBIO()
        io.set_structure(self._bio_structure)
        selector = SelectIndexedAtoms()
        for cluster_id, cluster in enumerate(self.clusters):
            cluster_id += 1
            print_output(f'INFO: correcting cluster {cluster_id}: residues {'+'.join(cluster.err_ress_ids_strs)}', self._silent)

            # ensure existence of a directory for the current correction level
            correction_dir = f'{self._correction_dir}/cluster_{cluster_id}'
            makedirs(name       = correction_dir,
                     exist_ok   = True)

            # try iterations of correction
            max_error_distance = max({res.min_err_distance if res.min_err_distance else 1
                                      for res in cluster.err_ress.values()})
            correction_level = 1
            correction_attempt = self
            output_file = ''
            pdb2pqr_problem = False
            pdb2pqr_nodebump = '--nodebump'
            debump_flag = ''
            while not all(correction_attempt.residues[res_id].side_chain_correct for res_id in cluster.err_ress.keys()):
                if correction_level > max_error_distance or pdb2pqr_problem:
                    if not pdb2pqr_problem and pdb2pqr_nodebump:
                        print_output('INFO: trying with debump...', self._silent)
                        correction_level = 1
                        pdb2pqr_nodebump = ''
                        debump_flag = '_d'
                        cluster.debump = True
                    else:
                        print_output('INFO: failure', self._silent)
                        break

                cutout_file = f'{correction_dir}/level{correction_level}.pdb'
                output_file = f'{correction_dir}/level{correction_level}{debump_flag}_out.pdb'
                pdb2pqr_log = f'{correction_dir}/level{correction_level}{debump_flag}_pdb2pqr.log'

                if pdb2pqr_nodebump:
                    # cut out the cluster's atoms into a correction file
                    kept_ats_ids = set()
                    for res_id in cluster.err_ress.keys():
                        kept_ats_ids.update(self.residues[res_id].get_kept_ats_ids(correction_level))
                    surr_ats_ids = cluster.surr_ats_ids
                    selector.update_indices(kept_ats_ids | surr_ats_ids)
                    io.save(cutout_file, selector)

                # correct by propka
                system(f'pdb2pqr30 {pdb2pqr_nodebump} --noopt --pdb-output {output_file} '
                       f'{cutout_file} {correction_dir}/delete.pqr '
                       f'2>{pdb2pqr_log};'
                       f'rm {correction_dir}/delete.*;')

                # check if the outcut cluster was not too small for pdb2pqr
                with open(pdb2pqr_log, mode='r') as log_file:
                    for line in log_file:
                        if line[0:39] == 'ERROR:This PDB file is missing too many':
                            self.pdb2pqr_errors_log.append((cluster_id, cluster.confidence_log))
                            cluster.pdb2pqr_error = True
                            print_output(f'ERROR: The level {correction_level} cutout {line[15:108]} {line[108:-1]}', self._silent) # should be a normal print in the user version
                            self.log += f'ERROR: The level {correction_level} cutout {line[15:108]} {line[108:]}'

                            pdb2pqr_problem = True
                            break

                # load the correction attempt output file to check successfulness
                if not pdb2pqr_problem:
                    correction_attempt = Protein(output_file)

                correction_level += 1

            else:
                print_output('INFO: success', self._silent)
                for res_id in cluster.err_ress.keys():
                    self.residues[res_id].side_chain_correct = True

                cluster.correct_side_chain_wise = True
                cluster.iterations = correction_level - 1
                cluster.file = f'{correction_dir}{debump_flag}.pdb'
                replace(src = output_file,
                        dst = cluster.file)

        # check if the whole protein was corrected
        if all(cluster.correct_side_chain_wise for cluster in self.clusters):
            self.side_chains_correct = True

        # write the corrected cluster into the output file
        for i, cluster in enumerate([cluster for cluster in self.clusters if cluster.correct_side_chain_wise], start=1):
            cluster_bio = PDBParser(QUIET=True).get_structure(id    = f'cluster{i}',
                                                              file  = cluster.file)
            for atom in cluster_bio.get_atoms():
                try:
                    self._bio_structure[0]['A'][atom.get_parent().id][atom.name].coord = atom.coord
                except KeyError:
                    continue
        io.save(self._final_file_path)


class StructureIntegrityCheckerAndCorrector:
    def __init__(self,
                 input_pdb_file         : str,
                 output_pdb_file        : str               = None,
                 logger                                     = None,
                 log_file               : str               = None,
                 delete_auxiliary_files : bool              = False,
                 silent                 : bool              = True):

        # control usability of files
        if not isfile(input_pdb_file):
            print(f'ERROR: File at {input_pdb_file} does not exist!')
            exit(2)
        if input_pdb_file == output_pdb_file:
            print(f'ERROR: Input and output files cannot be the same.')
            exit(3)

        self._input_PDB_file = abspath(input_pdb_file)
        defaultdir = f'{dirname(self._input_PDB_file)}/correction_sicc_af'
        self._correction_dir = f'{defaultdir}/{basename(self._input_PDB_file)[3:-16]}_correction'
        if output_pdb_file:
            self._output_PDB_file = abspath(output_pdb_file)
        else:
            self._output_PDB_file = f'{defaultdir}/{splitext(basename(self._input_PDB_file))[0]}_corrected.pdb'
        self._logger = logger
        self._log = ''
        self._log_file = log_file
        self._delete_auxiliary_files = delete_auxiliary_files
        self._silent = silent

    def process_structure(self):
        # load protein into a Protein object, check for errors
        print_output('INFO: loading file...', self._silent)
        protein = Protein(self._input_PDB_file,
                          self._output_PDB_file,
                          self._correction_dir,
                          self._silent)
        print_output(f'INFO: {protein.filename} loaded', self._silent)

        if protein.side_chains_correct and protein.backbone_correct:
            print_output(f'OK: No error found in {protein.filename}.', self._silent)
        else:

            self._log += f'{protein.name}:\n'
            erroneous_correction = False
            if not protein.side_chains_correct:
                # prepare the correction directory
                makedirs(name       = self._correction_dir,
                         exist_ok   = True)

                protein.execute_correction()

                # run a check of the final file
                persistent_side_chain_errors = protein.side_chain_err_ress.keys() - {res_id for cluster in protein.clusters if cluster.correct_side_chain_wise for res_id in cluster.err_ress.keys()}
                final_protein = Protein(self._output_PDB_file)
                if final_protein.side_chain_err_ress.keys() != persistent_side_chain_errors or final_protein.backbone_errors != protein.backbone_errors:
                    print_output('ERROR: pdb2pqr failed to correct the protein. The Correction result column in the following table is irrelevant.', self._silent)
                    for cluster in protein.clusters:
                        cluster.correct_side_chain_wise = False
                    protein.side_chains_correct = False
                    erroneous_correction = True

                # log the results
                self._log += protein.log
                listed_results = [[', '.join(cluster.err_ress_ids_strs), 'success' if cluster.correct_side_chain_wise else 'failure']
                                  for cluster in protein.clusters]
                table = tabulate(tabular_data   = listed_results,
                                 headers        = ['Clustered error residues', 'Correction result'],
                                 colalign       = ['left', 'left'])
                print_output(f'RESULTS:\n{table}', self._silent)
                self._log += f'{table}\n'
                if protein.side_chains_correct:
                    print_output(f'CORRECTION SUCCESSFUL for {protein.filename}', self._silent)
                elif all(not cluster.correct_side_chain_wise for cluster in protein.clusters):
                    print_output(f'CORRECTION FAILED for {protein.filename}', self._silent)
                else:
                    print_output(f'CORRECTION partially successful for {protein.filename}', self._silent)

                if self._logger:
                    self._logger.side_chain_errors[protein.name] = ([(cluster.confidence_log,
                                                                      cluster.error_type,
                                                                      cluster.correct_side_chain_wise,
                                                                      cluster.iterations,
                                                                      cluster.debump,
                                                                      cluster.pdb2pqr_error,
                                                                      cluster.cycle_confid_log) for cluster in protein.clusters],
                                                                    protein.side_chains_correct)
                    if protein.pdb2pqr_errors_log:
                        self._logger.pdb2pqr_error_log[protein.name] = protein.pdb2pqr_errors_log

            if not protein.backbone_correct:
                print_output(f'INFO: In {protein.filename}, backbone errors were found in these residues:', self._silent)
                chain_errors_string = '; '.join(list(map(str, protein.backbone_err_ress)))
                print_output(chain_errors_string, self._silent)
                self._log += (f'--------------------------\n'
                              f'Chain errors\n'
                              f'--------------------------\n'
                              f'{chain_errors_string}\n')
                print_output('WARNING: sicc_af does not provide correction of errors in the backbone.', self._silent)

                if self._logger:
                    self._logger.backbone_errors[protein.name] = [[(res_id, protein.residues[res_id].confidence) for res_id in error]
                                                                  for error in protein.backbone_errors]

            if erroneous_correction:
                self._log += 'ERROR: PDB2PQR failed to correct the protein.\n'

            # delete auxiliary files (if demanded)
            if self._delete_auxiliary_files:
                rmtree(self._correction_dir)

            # write the log file
            self._log += '\n'
            if self._log_file:
                self._log_file = abspath(self._log_file)
                makedirs(name       =dirname(self._log_file),
                         exist_ok   = True)
                with open(self._log_file, mode='a') as log_file:
                    log_file.write(self._log)
            elif not self._delete_auxiliary_files:
                self._log_file = f'{self._correction_dir}/log.txt'
                with open(self._log_file, mode='a') as log_file:
                    log_file.write(self._log)

            if erroneous_correction:
                exit(5)


if __name__ == '__main__':
    args = load_arguments()
    StructureIntegrityCheckerAndCorrector(args.input_PDB_file,
                                          args.output_PDB_file,
                                          None,
                                          args.log_file,
                                          args.delete_auxiliary_files,
                                          args.silent).process_structure()
