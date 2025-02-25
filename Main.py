from Bio.PDB import PDBParser, DSSP, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.vectors import calc_dihedral
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

rotational_thresholds = {
    "hydrogen_bonds": 30,
    "disulfide_bridge": 20,
    "ionic_bonds": 40,
    "van_der_waals": 45,
    "pi_pi_stacking": 25,
    "hydrophobic_interaction": 35
}

hydrophobic_residues = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP", "GLY"}
polar_residues = {"SER", "THR", "ASN", "GLN", "TYR", "CYS"}
charged_residues = {"LYS", "ARG", "HIS", "ASP", "GLU"}

class Interaction:
    def __init__(self, type_, strength, distance, angle):
        self.type_ = type_
        self.strength = strength
        self.distance = distance
        self.angle = angle

    def assign_letter(self):
        if self.type_ in interaction_letters:
            return interaction_letters[self.type_]
        return "U"

def calculate_torsion_angle(atom1, atom2, atom3, atom4):
    return np.degrees(calc_dihedral(atom1.get_vector(), atom2.get_vector(), atom3.get_vector(), atom4.get_vector()))

def classify_interaction(res1, res2, distance, torsion_angle):
    for interaction_type, threshold in interaction_thresholds.items():
        if distance < threshold and abs(torsion_angle) < rotational_thresholds[interaction_type]:
            return interaction_type
    return None

def extract_interaction_sequence(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    residues.append(residue)

    neighbor_search = NeighborSearch([atom for residue in residues for atom in residue])
    sequence = []

    for i, res1 in enumerate(residues):
        interaction_found = False
        for res2 in residues[i + 1:]:
            distance = res1["CA"] - res2["CA"]
            if distance < max(interaction_thresholds.values()):
                torsion_angle = calculate_torsion_angle(
                    res1["N"], res1["CA"], res2["CA"], res2["C"]
                )
                interaction_type = classify_interaction(
                    res1.get_resname(), res2.get_resname(), distance, torsion_angle
                )
                if interaction_type:
                    interaction = Interaction(interaction_type, strength=10, distance=distance, angle=torsion_angle)
                    sequence.append(interaction.assign_letter())
                    interaction_found = True
                    break
        if not interaction_found:
            sequence.append("U")

    return "".join(sequence)

def main():
    pdb_file = "example.pdb"
    interaction_seq = extract_interaction_sequence(pdb_file)
    print("\nInteraction Sequence:", interaction_seq)

if __name__ == "__main__":
    main()
