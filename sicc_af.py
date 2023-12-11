import argparse
from rdkit import Chem
from archetypes import archetypes
from os.path import isfile


errors = {'CLASHING_PLACEMENT': [],
          'BOND_OVEREXTENSION': [],
          'H_ATM_MISPLACEMENT': []}

# STRUCTURE:
# errors
#   |--- CLASHING_PLACEMENT:
#           |--- entries[]
#                   |--- [RESIDUE NUMBER]: atoms' to be removed nef names[]
#   |--- BOND_OVEREXTENSION:
#           |--- --//--
#   |--- H_ATM_MISPLACEMENT:
#           |--- --//--


def add_to(err_type, residue_no, atom, residue_nm):
    if residue_no in err_type.keys():
        err_type[residue_no][1].add(atom)
    else:
        err_type[residue_no] = [residue_nm, {atom}]

    # STRUCTURE:
    # err_xx
    #   |--- [RESIDUE NUMBER]:
    #           |--- residue type     [0]
    #           |--- [atom nef_names] [1]


def get_errors(path):

    # INTERNAL ERROR DICTIONARIES
    _err_hy = {}     # Errors with hydrogen
    _err_ha = {}     # Errors with heavy atoms
    _err_hc = set()  # Errors with heavy atoms causing clashes between residues

    # MAKE MOLECULE FROM FILE
    molecule = Chem.MolFromPDBFile(path,
                                   removeHs=False,
                                   sanitize=False)
    residues = {}

    # MAKE MULTI-LEVEL ATOMS DICTIONARY
    for atom in molecule.GetAtoms():

        a_info = atom.GetPDBResidueInfo()
        a_res_no = a_info.GetResidueNumber()
        a_res_nm = a_info.GetResidueName().strip()
        a_nef_nm = a_info.GetName().strip()

        if a_res_no in residues.keys():
            residues[a_res_no][1][a_nef_nm] = set()

        else:
            residues[a_res_no] = (
                a_res_nm,
                {a_nef_nm: set()})

    last_res_no = max(residues.keys())

    # STRUCTURE:
    # residues
    #   |--- [RESIDUE NUMBER]:
    #           |--- residue name [0]
    #           |--- atoms        [1]
    #                   |--- [ATOM NEF NAME]: {bonded atoms' nef names}

    # ADD BONDED ATOMS + FIND SOME ERRORS
    for bond in molecule.GetBonds():

        atom_1 = bond.GetBeginAtom()
        atom_2 = bond.GetEndAtom()
        a1_info = atom_1.GetPDBResidueInfo()
        a2_info = atom_2.GetPDBResidueInfo()
        a1_res_no = a1_info.GetResidueNumber()
        a2_res_no = a2_info.GetResidueNumber()
        a1_nef_nm = a1_info.GetName().strip()
        a2_nef_nm = a2_info.GetName().strip()
        a1_res_nm = a1_info.GetResidueName().strip()
        a2_res_nm = a2_info.GetResidueName().strip()

        # CHECK IF BOND IS VALID (WITHIN ITS OWN RESIDUE, IN PEPTIDE BOND OR IN DISULPHIDE BOND)
        if (a1_res_no == a2_res_no
                or abs(a1_res_no - a2_res_no) == 1 and
                (a1_nef_nm == 'C' and a2_nef_nm == 'N' or a1_nef_nm == 'N' and a2_nef_nm == 'C')):

            residues[a1_res_no][1][a1_nef_nm].add(a2_nef_nm)
            residues[a2_res_no][1][a2_nef_nm].add(a1_nef_nm)

        elif a1_nef_nm == 'SG' and a2_nef_nm == 'SG':
            continue

        else:

            # IF AN ATOMS IS H -> H_ATM_MISPLACEMENT ERRORS
            a1_is_h = 1 if atom_1.GetSymbol() == 'H' else 0
            a2_is_h = 1 if atom_2.GetSymbol() == 'H' else 0

            if a1_is_h:
                add_to(_err_hy, a1_res_no, a1_nef_nm, a1_res_nm)
            if a2_is_h:
                add_to(_err_hy, a2_res_no, a2_nef_nm, a2_res_nm)

            # IF NEITHER ATOM IS H -> CLASHING_PLACEMENT
            else:
                _err_hc.add(frozenset({a1_res_no, a2_res_no}))

    # CHECK BONDED ATOMS
    for res_no, res_info in residues.items():
        for a_nef_nm, bonded_ats_nef_nm in res_info[1].items():

            res_nm = res_info[0]
            archetype = archetypes[res_nm][a_nef_nm]

            # CHECK IF VALID
            valid = False
            if type(archetype) == list:
                for variant in archetype:
                    if bonded_ats_nef_nm == variant:
                        valid = True
                    else:
                        continue
            elif archetype == bonded_ats_nef_nm:
                valid = True

            if valid:
                continue
            elif res_no == last_res_no and a_nef_nm == 'C' and bonded_ats_nef_nm == {'O', 'CA', 'OXT'}:
                continue
            elif res_no == 1 and a_nef_nm == 'N' and bonded_ats_nef_nm <= {'CA', 'H3', 'H2', 'H'}:
                continue

            # CHECK WHAT PROBLEM
            else:
                if a_nef_nm[0] == 'H':
                    add_to(_err_hy, res_no, a_nef_nm, res_nm)
                else:
                    add_to(_err_ha, res_no, a_nef_nm, res_nm)

    # TRANSFORM TO OUTPUT
    resds_already_noted = set()
    resds_of_hydrogens_already_noted = set()

    for couple in _err_hc:
        entry = {}

        for res_no in couple:
            res_atoms = residues[res_no][1].keys()
            entry[res_no] = {atom for atom in res_atoms if atom not in {'N', 'C'}}
            resds_already_noted.add(res_no)

        errors['CLASHING_PLACEMENT'].append(entry)

    for res_no, res_info in _err_ha.items():

        if res_no not in resds_already_noted:
            entry = {}
            res_nm = res_info[0]
            err_ats_nef_nm = res_info[1]

            if res_nm == 'HIS' and (err_ats_nef_nm == {'CD2', 'CE1', 'NE2'} or err_ats_nef_nm == {'CD2', 'CE1', 'NE2', 'ND1'}):
                entry[res_no] = err_ats_nef_nm.union({'HD1', 'HD2', 'HE1', 'HE2'})
                resds_of_hydrogens_already_noted.add(res_no)

            elif err_ats_nef_nm <= {'N', 'C', 'CA', 'O'}:
                if 'O' in err_ats_nef_nm:
                    entry[res_no] = {'O'}

            else:
                res_atoms = residues[res_no][1].keys()
                entry[res_no] = {atom for atom in res_atoms if atom not in {'N', 'C'}}
                resds_already_noted.add(res_no)

            errors['BOND_OVEREXTENSION'].append(entry)

    for res_no, res_info in _err_hy.items():
        entry = {}
        res_nm = res_info[0]
        err_ats_nef_nm = res_info[1]

        if res_nm in resds_of_hydrogens_already_noted:
            to_note = err_ats_nef_nm.difference({'HD1', 'HD2', 'HE1', 'HE2'})
            if to_note:
                entry = to_note

        if res_no not in resds_already_noted:
            entry[res_no] = err_ats_nef_nm

        errors['H_ATM_MISPLACEMENT'].append(entry)

    return errors


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pdb-file',
        type=str,
        required=True,
        dest='pdb_file',
        help='Path to .pdb file to be checked.')
    return parser


def main():

    # GET PATH FROM CALLING ARGUMENT
    args = build_parser().parse_args()
    path = args.pdb_file
    if not isfile(path):
        print('Please provide a valid path.')
        exit()

    # GET ERRORS FROM PATH FILE
    return get_errors(path)


if __name__ == '__main__':
    main()
