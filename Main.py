from Bio.PDB import PDBParser, DSSP, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import numpy as np

interaction_letters = {
    "hydrogen_bonds": "H",
    "disulfide_bridge": "D",
    "ionic_bonds": "I",
    "van_der_waals": "V",
    "pi_pi_stacking": "P",
    "hydrophobic_interaction": "Y"
}

gap_penalty = -2

interaction_thresholds = {
    "hydrogen_bonds": 3.5,
    "disulfide_bridge": 2.2,
    "ionic_bonds": 5.0,
    "van_der_waals": 4.0,
    "pi_pi_stacking": 5.0,
    "hydrophobic_interaction": 4.5
}

hydrophobic_residues = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP"}
polar_residues = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
charged_residues = {"LYS", "ARG", "HIS", "ASP", "GLU"}

class Interaction:
    def __init__(self, type_, strength, distance, angle=None):
        self.type_ = type_
        self.strength = strength
        self.distance = distance
        self.angle = angle

    def assign_letter(self):
        if self.type_ in interaction_letters:
            return interaction_letters[self.type_]
        return "U"

def classify_interaction(res1, res2, distance):
    if distance < interaction_thresholds["hydrogen_bonds"] and (
        res1 in polar_residues or res2 in polar_residues
    ):
        return "hydrogen_bonds"
    elif distance < interaction_thresholds["disulfide_bridge"] and (
        res1 == "CYS" and res2 == "CYS"
    ):
        return "disulfide_bridge"
    elif distance < interaction_thresholds["ionic_bonds"] and (
        res1 in charged_residues and res2 in charged_residues
    ):
        return "ionic_bonds"
    elif distance < interaction_thresholds["van_der_waals"]:
        return "van_der_waals"
    elif distance < interaction_thresholds["pi_pi_stacking"] and (
        res1 in {"PHE", "TYR", "TRP"} and res2 in {"PHE", "TYR", "TRP"}
    ):
        return "pi_pi_stacking"
    elif distance < interaction_thresholds["hydrophobic_interaction"] and (
        res1 in hydrophobic_residues and res2 in hydrophobic_residues
    ):
        return "hydrophobic_interaction"
    return None

def parse_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)

    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinates.append(atom.get_coord())

    return coordinates

def secondaryextract(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    model = next(structure.get_models())

    dssp = DSSP(model, file_path)
    secondary_structures = [dssp[key][2] for key in dssp.keys()]
    return secondary_structures

def needleman_wunsch(seq1, seq2, scoring_matrix, gap_penalty):
    n = len(seq1)
    m = len(seq2)

    score = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap_penalty
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i - 1][j - 1] + scoring_matrix[seq1[i - 1]].get(seq2[j - 1], -1)
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, delete, insert)

    return score[-1][-1]

def extract_interaction_sequence(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)

    atoms = []
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    residues.append(residue)
                    for atom in residue:
                        atoms.append(atom)

    neighbor_search = NeighborSearch(atoms)
    sequence = []

    for i, res1 in enumerate(residues):
        interaction_found = False
        for res2 in residues[i + 1:]:
            distance = neighbor_search.search(res1["CA"].coord, 5.0)
            if distance:
                interaction_type = classify_interaction(
                    res1.get_resname(), res2.get_resname(), distance
                )
                if interaction_type:
                    interaction = Interaction(
                        interaction_type, strength=10, distance=distance
                    )
                    sequence.append(interaction.assign_letter())
                    interaction_found = True
                    break
        if not interaction_found:
            sequence.append("U")

    return "".join(sequence)

def main():
    pdb_file = "example.pdb"

    coords = parse_pdb(pdb_file)
    print("3D Coordinates:")
    for coord in coords:
        print(coord)

    second_struct = secondaryextract(pdb_file)
    print("\nSecondary Structures:")
    for struct in second_struct:
        print(struct)

    interaction_seq = extract_interaction_sequence(pdb_file)
    print("\nInteraction Sequence:", interaction_seq)

    seq1 = "HDIV"
    seq2 = "HDIVY"
    scoring_matrix = {
        "H": {"H": 1, "D": -1, "I": -1, "V": -1, "P": -1, "Y": -1},
        "D": {"H": -1, "D": 1, "I": -1, "V": -1, "P": -1, "Y": -1},
        "I": {"H": -1, "D": -1, "I": 1, "V": -1, "P": -1, "Y": -1},
        "V": {"H": -1, "D": -1, "I": -1, "V": 1, "P": -1, "Y": -1},
        "P": {"H": -1, "D": -1, "I": -1, "V": -1, "P": 1, "Y": -1},
        "Y": {"H": -1, "D": -1, "I": -1, "V": -1, "P": -1, "Y": 1}
    }
    score = needleman_wunsch(seq1, seq2, scoring_matrix, gap_penalty)
    print(f"\nAlignment score: {score}")

if __name__ == "__main__":
    main()