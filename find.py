from rdkit import Chem
from archetypes import archetypes

output = {'CLASHING_PLACEMENT': [],
          'BOND_OVEREXTENSION': [],
          'H_ATM_MISPLACEMENT': []}

# STRUCTURE:
# errors
#   |--- CLASHING_PLACEMENT:
#           |--- [entries]
#                   |--- [RESIDUE NUMBER]: [atoms' to be removed nef names]
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


def f_invalid(path):
    err_pp = {}
    err_af = {}
    err_0h = set()

    # MAKE MOLECULE FROM FILE
    molecule = Chem.MolFromPDBFile(path,
                                   removeHs=False,
                                   sanitize=False)
    residues = {}

    # MAKE MULTI-LEVEL ATOMS DICTIONARY
    for atom in molecule.GetAtoms():

        atom_res_no = atom.GetPDBResidueInfo().GetResidueNumber()
        atom_res_nm = atom.GetPDBResidueInfo().GetResidueName().strip()
        atom_nef_nm = atom.GetPDBResidueInfo().GetName().strip()

        if atom_res_no in residues.keys():
            residues[atom_res_no][1][atom_nef_nm] = set()

        else:
            residues[atom_res_no] = (
                atom_res_nm,
                {atom_nef_nm: set()})

    last_res_no = max(residues.keys())

    # STRUCTURE:
    # residues
    #   |--- [RESIDUE NUMBER]:
    #           |--- residue name [0]
    #           |--- atoms        [1]
    #                   |--- [ATOM NEF NAME]: {bonded atoms' nef names}

    # ADD BONDED ATOMS + FIND SOME ERRORS
    for bond in molecule.GetBonds():

        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        a1_res_no = atom1.GetPDBResidueInfo().GetResidueNumber()
        a2_res_no = atom2.GetPDBResidueInfo().GetResidueNumber()
        a1_nef_nm = atom1.GetPDBResidueInfo().GetName().strip()
        a2_nef_nm = atom2.GetPDBResidueInfo().GetName().strip()
        a1_res_nm = atom1.GetPDBResidueInfo().GetResidueName().strip()
        a2_res_nm = atom2.GetPDBResidueInfo().GetResidueName().strip()

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
            a1_is_h = 1 if atom1.GetSymbol() == 'H' else 0
            a2_is_h = 1 if atom2.GetSymbol() == 'H' else 0

            if a1_is_h:
                add_to(err_pp, a1_res_no, a1_nef_nm, a1_res_nm)
            if a2_is_h:
                add_to(err_pp, a2_res_no, a2_nef_nm, a2_res_nm)

            # IF NEITHER ATOM IS H -> CLASHING_PLACEMENT
            else:
                err_0h.add(frozenset({a1_res_no, a2_res_no}))

    # CHECK BONDED ATOMS
    for res_no, residue in residues.items():
        for atom_nef, bonded_ats_nef in residue[1].items():

            res_nm = residue[0]
            archetype = archetypes[res_nm][atom_nef]

            # CHECK IF VALID
            valid = False
            if type(archetype) == list:
                for var in archetype:
                    if bonded_ats_nef == var:
                        valid = True
                    else:
                        continue
            elif archetype == bonded_ats_nef:
                valid = True

            if valid:
                continue
            elif res_no == last_res_no and atom_nef == 'C' and bonded_ats_nef == {'O', 'CA', 'OXT'}:
                continue
            elif res_no == 1 and atom_nef == 'N' and bonded_ats_nef <= {'CA', 'H3', 'H2', 'H'}:
                continue

            # CHECK WHAT PROBLEM
            else:
                if atom_nef[0] == 'H':
                    add_to(err_pp, res_no, atom_nef, res_nm)
                else:
                    add_to(err_af, res_no, atom_nef, res_nm)

    # TRANSFORM TO OUTPUT
    residues_already_noted = set()
    residues_of_hydrogens_already_noted = set()

    for couple in err_0h:
        entry = {}

        for res in couple:
            res_atoms = residues[res][1].keys()
            entry[res] = {atom for atom in res_atoms if atom not in {'N', 'C'}}
            residues_already_noted.add(res)

        output['CLASHING_PLACEMENT'].append(entry)

    for resno, resi in err_af.items():

        if resno not in residues_already_noted:
            entry = {}
            resnm = resi[0]
            err_ats = resi[1]

            if resnm == 'HIS' and (err_ats == {'CD2', 'CE1', 'NE2'} or err_ats == {'CD2', 'CE1', 'NE2', 'ND1'}):
                entry[resno] = err_ats.union({'HD1', 'HD2', 'HE1', 'HE2'})
                residues_of_hydrogens_already_noted.add(resno)

            elif err_ats <= {'N', 'C', 'CA', 'O'}:
                if 'O' in err_ats:
                    entry[resno] = {'O'}

            else:
                res_atoms = residues[resno][1].keys()
                entry[resno] = {atom for atom in res_atoms if atom not in {'N', 'C'}}
                residues_already_noted.add(resno)

            output['BOND_OVEREXTENSION'].append(entry)

    for resno, resi in err_pp.items():
        entry = {}
        resnm = resi[0]
        err_ats = resi[1]

        if resnm in residues_of_hydrogens_already_noted:
            to_note = err_ats.difference({'HD1', 'HD2', 'HE1', 'HE2'})
            if to_note:
                entry = to_note

        if resno not in residues_already_noted:
            entry[resno] = err_ats

        output['H_ATM_MISPLACEMENT'].append(entry)

    return output
