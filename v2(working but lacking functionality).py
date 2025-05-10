from Bio.PDB import PDBParser, DSSP, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.vectors import calc_dihedral
import numpy as np
import argparse

interaction_letters = {
    "hydrogen_bonds": "H",
    "disulfide_bridge": "D",
    "ionic_bonds": "I",
    "van_der_waals": "V",
    "pi_pi_stacking": "P",
    "hydrophobic_interaction": "Y"
}

gap_penalty = -2
match_score = 1
mismatch_score = -1

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
aromatic_residues = {"PHE", "TYR", "TRP", "HIS"}

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
    try:
        return np.degrees(calc_dihedral(atom1.get_vector(), atom2.get_vector(), atom3.get_vector(), atom4.get_vector()))
    except Exception:
        return 0.0

def classify_interaction(res1, res2, distance, torsion_angle):
    res1_name = res1.get_resname()
    res2_name = res2.get_resname()
    
    if res1_name == "CYS" and res2_name == "CYS":
        try:
            sg1 = res1["SG"]
            sg2 = res2["SG"]
            sg_distance = sg1 - sg2
            if sg_distance < interaction_thresholds["disulfide_bridge"]:
                return "disulfide_bridge"
        except KeyError:
            pass  
    
    if res1_name in charged_residues and res2_name in charged_residues:
        if distance < interaction_thresholds["ionic_bonds"]:
            return "ionic_bonds"
    
    if (res1_name in polar_residues and res2_name in polar_residues) or \
       (res1_name in polar_residues and res2_name in charged_residues) or \
       (res1_name in charged_residues and res2_name in polar_residues):
        if distance < interaction_thresholds["hydrogen_bonds"]:
            return "hydrogen_bonds"
    
    if res1_name in aromatic_residues and res2_name in aromatic_residues:
        if distance < interaction_thresholds["pi_pi_stacking"] and abs(torsion_angle) < rotational_thresholds["pi_pi_stacking"]:
            return "pi_pi_stacking"
    
    if res1_name in hydrophobic_residues and res2_name in hydrophobic_residues:
        if distance < interaction_thresholds["hydrophobic_interaction"]:
            return "hydrophobic_interaction"
    
    if distance < interaction_thresholds["van_der_waals"]:
        return "van_der_waals"
    
    return None

def has_required_atoms(residue):
    return all(atom in residue for atom in ["N", "CA", "C"])

def extract_interaction_sequence(file_path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", file_path)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return ""
    
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True) and has_required_atoms(residue):
                    residues.append(residue)
    
    if not residues:
        print("No valid amino acid residues found in the structure")
        return ""
    
    atoms = []
    for residue in residues:
        for atom in residue:
            atoms.append(atom)
    
    neighbor_search = NeighborSearch(atoms)
    sequence = []
    
    for i, res1 in enumerate(residues):
        interaction_found = False
        ca1 = res1["CA"]
        nearby_atoms = neighbor_search.search(ca1.get_coord(), max(interaction_thresholds.values()) + 1.0)
        
        nearby_residues = {atom.get_parent() for atom in nearby_atoms if atom.get_parent() != res1}
        
        nearby_residues_with_distance = []
        for res2 in nearby_residues:
            if is_aa(res2, standard=True) and has_required_atoms(res2):
                try:
                    distance = res1["CA"] - res2["CA"]
                    nearby_residues_with_distance.append((res2, distance))
                except KeyError:
                    continue
        
        nearby_residues_with_distance.sort(key=lambda x: x[1])
        
        for res2, distance in nearby_residues_with_distance:
            if residues.index(res2) <= i:
                continue
                
            try:
                torsion_angle = calculate_torsion_angle(
                    res1["N"], res1["CA"], res2["CA"], res2["C"]
                )
                
                interaction_type = classify_interaction(
                    res1, res2, distance, torsion_angle
                )
                
                if interaction_type:
                    interaction = Interaction(interaction_type, strength=10, distance=distance, angle=torsion_angle)
                    sequence.append(interaction.assign_letter())
                    interaction_found = True
                    break
            except KeyError:
                continue
        
        if not interaction_found:
            sequence.append("U")  
    
    return "".join(sequence)

def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-2):
   
    n, m = len(seq1), len(seq2)
    score_matrix = np.zeros((n+1, m+1))
    
    for i in range(n+1):
        score_matrix[i, 0] = i * gap_penalty
    for j in range(m+1):
        score_matrix[0, j] = j * gap_penalty
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = score_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = score_matrix[i-1, j] + gap_penalty
            insert = score_matrix[i, j-1] + gap_penalty
            score_matrix[i, j] = max(match, delete, insert)
    
    aligned_seq1, aligned_seq2 = [], []
    i, j = n, m
    
    while i > 0 and j > 0:
        score_current = score_matrix[i, j]
        score_diagonal = score_matrix[i-1, j-1]
        score_up = score_matrix[i-1, j]
        score_left = score_matrix[i, j-1]
        
        if score_current == score_diagonal + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score):
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append('-')
            i -= 1
        elif score_current == score_left + gap_penalty:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j-1])
            j -= 1
    
    while i > 0:
        aligned_seq1.append(seq1[i-1])
        aligned_seq2.append('-')
        i -= 1
    while j > 0:
        aligned_seq1.append('-')
        aligned_seq2.append(seq2[j-1])
        j -= 1
    
    aligned_seq1 = ''.join(aligned_seq1[::-1])
    aligned_seq2 = ''.join(aligned_seq2[::-1])
    
    return aligned_seq1, aligned_seq2, score_matrix[n, m]

def compare_structures(pdb_file1, pdb_file2):
    """
    Compare two protein structures by aligning their interaction sequences.
    
    Args:
        pdb_file1 (str): Path to first PDB file
        pdb_file2 (str): Path to second PDB file
        
    Returns:
        tuple: Aligned sequences and similarity score
    """
    print(f"Extracting interaction sequence from {pdb_file1}...")
    seq1 = extract_interaction_sequence(pdb_file1)
    if not seq1:
        print(f"Failed to extract interaction sequence from {pdb_file1}")
        return None, None, 0
    
    print(f"Extracting interaction sequence from {pdb_file2}...")
    seq2 = extract_interaction_sequence(pdb_file2)
    if not seq2:
        print(f"Failed to extract interaction sequence from {pdb_file2}")
        return None, None, 0
    
    print(f"\nInteraction sequence 1: {seq1}")
    print(f"Interaction sequence 2: {seq2}")
    
    print("\nAligning sequences...")
    aligned_seq1, aligned_seq2, score = needleman_wunsch(seq1, seq2, match_score, mismatch_score, gap_penalty)
    
    print(f"Alignment score: {score}")
    print(f"Aligned sequence 1: {aligned_seq1}")
    print(f"Aligned sequence 2: {aligned_seq2}")
    
    total_positions = len(aligned_seq1)
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-' and b != '-')
    identical = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
    gaps = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == '-' or b == '-')
    
    similarity = (matches / total_positions) * 100 if total_positions > 0 else 0
    identity = (identical / total_positions) * 100 if total_positions > 0 else 0
    gap_percentage = (gaps / total_positions) * 100 if total_positions > 0 else 0
    
    print(f"\nSequence similarity: {similarity:.2f}%")
    print(f"Sequence identity: {identity:.2f}%")
    print(f"Gap percentage: {gap_percentage:.2f}%")
    
    print("\nAlignment visualization:")
    print(aligned_seq1)
    match_line = ''.join('|' if a == b and a != '-' else ' ' for a, b in zip(aligned_seq1, aligned_seq2))
    print(match_line)
    print(aligned_seq2)
    
    return aligned_seq1, aligned_seq2, similarity

def visualize_alignment(aligned_seq1, aligned_seq2):
    """
    Create a more detailed visualization of the alignment showing interaction types.
    
    Args:
        aligned_seq1 (str): First aligned sequence
        aligned_seq2 (str): Second aligned sequence
    """
    interaction_names = {
        "H": "Hydrogen Bond",
        "D": "Disulfide Bridge",
        "I": "Ionic Bond",
        "V": "Van der Waals",
        "P": "Pi-Pi Stacking",
        "Y": "Hydrophobic",
        "U": "Unknown"
    }
    
    print("\nDetailed Alignment Analysis:")
    print("-" * 80)
    print(f"{'Position':<10}{'Struct 1':<15}{'Struct 2':<15}{'Match':<10}{'Interaction Type':<30}")
    print("-" * 80)
    
    for i, (a, b) in enumerate(zip(aligned_seq1, aligned_seq2)):
        match = "Yes" if a == b and a != '-' else "No"
        interaction1 = interaction_names.get(a, "Gap") if a != '-' else "Gap"
        interaction2 = interaction_names.get(b, "Gap") if b != '-' else "Gap"
        
        print(f"{i+1:<10}{a:<15}{b:<15}{match:<10}{interaction1 if a == b else f'{interaction1}/{interaction2}':<30}")
    
    print("-" * 80)

def analyze_single_structure(pdb_file):
    """
    Analyze a single PDB structure and display its interaction sequence.
    
    Args:
        pdb_file (str): Path to PDB file
    """
    print(f"Analyzing structure: {pdb_file}")
    interaction_seq = extract_interaction_sequence(pdb_file)
    
    if interaction_seq:
        print("\nInteraction Summary:")
        print(f"Sequence length: {len(interaction_seq)}")
        print(f"Interaction sequence: {interaction_seq}")
        
        counts = {}
        for letter in interaction_seq:
            counts[letter] = counts.get(letter, 0) + 1
        
        print("\nInteraction Statistics:")
        interaction_names = {
            "H": "Hydrogen bonds",
            "D": "Disulfide bridges",
            "I": "Ionic bonds",
            "V": "Van der Waals",
            "P": "Pi-Pi stacking",
            "Y": "Hydrophobic interactions",
            "U": "Unknown/no interactions"
        }
        
        for letter, name in interaction_names.items():
            count = counts.get(letter, 0)
            percentage = (count / len(interaction_seq)) * 100 if interaction_seq else 0
            print(f"{name}: {count} ({percentage:.2f}%)")
    else:
        print("Failed to extract interaction sequence")

def main():
    parser = argparse.ArgumentParser(description='Analyze protein interactions and compare structures')
    parser.add_argument('--pdb1', required=True, help='Path to first PDB file')
    parser.add_argument('--pdb2', help='Path to second PDB file for comparison (optional)')
    parser.add_argument('--match', type=float, default=1.0, help='Score for matching interactions (default: 1.0)')
    parser.add_argument('--mismatch', type=float, default=-1.0, help='Score for mismatching interactions (default: -1.0)')
    parser.add_argument('--gap', type=float, default=-2.0, help='Gap penalty (default: -2.0)')
    
    args = parser.parse_args()
    
    global match_score, mismatch_score, gap_penalty
    match_score = args.match
    mismatch_score = args.mismatch
    gap_penalty = args.gap
    
    try:
        if args.pdb2:
            aligned_seq1, aligned_seq2, similarity = compare_structures(args.pdb1, args.pdb2)
            if aligned_seq1 and aligned_seq2:
                visualize_alignment(aligned_seq1, aligned_seq2)
        else:
            analyze_single_structure(args.pdb1)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
