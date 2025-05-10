from Bio.PDB import PDBParser, DSSP, NeighborSearch
from Bio.PDB.Polypeptide import is_aa, three_to_one
from Bio.PDB.vectors import calc_dihedral, calc_angle
import numpy as np
import mdtraj as md
import freesasa
import argparse
import os
import pickle
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import subprocess
import time
import re
from Bio import pairwise2

# Define constants and parameters
# Interaction types with more detailed descriptions
INTERACTION_TYPES = {
    "HB": "Hydrogen bond",
    "SB": "Salt bridge",
    "DS": "Disulfide bond",
    "VW": "Van der Waals",
    "PP": "Pi-Pi stacking",
    "PC": "Pi-Cation",
    "HP": "Hydrophobic",
    "WM": "Water-mediated",
    "MC": "Metal coordination",
    "CR": "Cation-pi",
    "HF": "Halogen bond",
    "BB": "Backbone-backbone",
    "BS": "Backbone-sidechain",
    "SS": "Sidechain-sidechain"
}

# Strength levels (1-5, where 5 is strongest)
STRENGTH_LEVELS = {
    1: "Very weak",
    2: "Weak",
    3: "Moderate",
    4: "Strong",
    5: "Very strong"
}

# Geometry types
GEOMETRY_TYPES = {
    "P": "Parallel",
    "A": "Antiparallel",
    "O": "Orthogonal",
    "B": "Beta sheet",
    "H": "Helical",
    "T": "Turn",
    "L": "Loop",
    "X": "Complex"
}

# Updated distance thresholds based on literature
INTERACTION_DISTANCES = {
    "HB": {"min": 2.5, "max": 3.5},
    "SB": {"min": 2.8, "max": 5.0},
    "DS": {"min": 1.8, "max": 2.5},
    "VW": {"min": 3.0, "max": 5.0},
    "PP": {"min": 3.5, "max": 7.0},
    "PC": {"min": 3.0, "max": 6.0},
    "HP": {"min": 3.0, "max": 5.0},
    "WM": {"min": 2.5, "max": 5.5},
    "MC": {"min": 1.8, "max": 3.0},
    "CR": {"min": 3.0, "max": 6.0},
    "HF": {"min": 2.5, "max": 4.0},
    "BB": {"min": 2.5, "max": 5.0},
    "BS": {"min": 2.5, "max": 5.0},
    "SS": {"min": 2.5, "max": 5.0}
}

# Angular thresholds for different interaction types
ANGLE_THRESHOLDS = {
    "HB": {"min": 120, "max": 180},
    "SB": {"min": 90, "max": 180},
    "PP": {"min": 0, "max": 30},    # For parallel stacking
    "PC": {"min": 0, "max": 60},    # For parallel cation-pi
    "CR": {"min": 0, "max": 60}     # For cation-pi
}

# Residue properties
AA_PROPERTIES = {
    "ALA": {"hydrophobic": True, "aromatic": False, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False},
    "ARG": {"hydrophobic": False, "aromatic": False, "polar": True, "charged": True, "pos_charged": True, "neg_charged": False},
    "ASN": {"hydrophobic": False, "aromatic": False, "polar": True, "charged": False, "pos_charged": False, "neg_charged": False},
    "ASP": {"hydrophobic": False, "aromatic": False, "polar": True, "charged": True, "pos_charged": False, "neg_charged": True},
    "CYS": {"hydrophobic": True, "aromatic": False, "polar": True, "charged": False, "pos_charged": False, "neg_charged": False},
    "GLN": {"hydrophobic": False, "aromatic": False, "polar": True, "charged": False, "pos_charged": False, "neg_charged": False},
    "GLU": {"hydrophobic": False, "aromatic": False, "polar": True, "charged": True, "pos_charged": False, "neg_charged": True},
    "GLY": {"hydrophobic": True, "aromatic": False, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False},
    "HIS": {"hydrophobic": False, "aromatic": True, "polar": True, "charged": True, "pos_charged": True, "neg_charged": False},
    "ILE": {"hydrophobic": True, "aromatic": False, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False},
    "LEU": {"hydrophobic": True, "aromatic": False, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False},
    "LYS": {"hydrophobic": False, "aromatic": False, "polar": True, "charged": True, "pos_charged": True, "neg_charged": False},
    "MET": {"hydrophobic": True, "aromatic": False, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False},
    "PHE": {"hydrophobic": True, "aromatic": True, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False},
    "PRO": {"hydrophobic": True, "aromatic": False, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False},
    "SER": {"hydrophobic": False, "aromatic": False, "polar": True, "charged": False, "pos_charged": False, "neg_charged": False},
    "THR": {"hydrophobic": False, "aromatic": False, "polar": True, "charged": False, "pos_charged": False, "neg_charged": False},
    "TRP": {"hydrophobic": True, "aromatic": True, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False},
    "TYR": {"hydrophobic": False, "aromatic": True, "polar": True, "charged": False, "pos_charged": False, "neg_charged": False},
    "VAL": {"hydrophobic": True, "aromatic": False, "polar": False, "charged": False, "pos_charged": False, "neg_charged": False}
}

# Define important atom groups for interactions
BACKBONE_ATOMS = ["N", "CA", "C", "O"]
AROMATIC_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"]
}
CHARGED_ATOMS = {
    "ARG": ["NH1", "NH2", "NE"],
    "LYS": ["NZ"],
    "HIS": ["ND1", "NE2"],
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"]
}
POLAR_ATOMS = {
    "SER": ["OG"],
    "THR": ["OG1"],
    "ASN": ["OD1", "ND2"],
    "GLN": ["OE1", "NE2"],
    "TYR": ["OH"],
    "CYS": ["SG"]
}

class EnhancedInteraction:
    """
    Class to represent enhanced interaction between residues with detailed properties.
    """
    def __init__(self, res1, res2, structure=None):
        """
        Initialize an interaction between two residues.
        
        Args:
            res1: First residue
            res2: Second residue
            structure: The parent structure (for context)
        """
        self.res1 = res1
        self.res2 = res2
        self.structure = structure
        self.res1_name = res1.get_resname()
        self.res2_name = res2.get_resname()
        self.res1_id = res1.get_id()[1]
        self.res2_id = res2.get_id()[1]
        self.chain1 = res1.get_parent().id
        self.chain2 = res2.get_parent().id
        
        # Calculate basic properties
        self.ca_distance = self._calculate_ca_distance()
        self.cb_distance = self._calculate_cb_distance()
        self.min_atom_distance = self._calculate_min_atom_distance()
        self.centroid_distance = self._calculate_centroid_distance()
        
        # Detect interaction types
        self.interaction_types = self._detect_interaction_types()
        
        # Generate the interaction code
        self.interaction_code = self._generate_interaction_code()
        
        # Calculate interaction strength
        self.interaction_strength = self._calculate_interaction_strength()
        
        # Determine geometric arrangement
        self.geometry = self._determine_geometry()
        
        # Generate the final interaction fingerprint
        self.fingerprint = self._generate_fingerprint()

    def _calculate_ca_distance(self):
        """Calculate distance between CA atoms."""
        try:
            return self.res1["CA"] - self.res2["CA"]
        except KeyError:
            return float('inf')
    
    def _calculate_cb_distance(self):
        """Calculate distance between CB atoms (or CA for GLY)."""
        try:
            atom1 = self.res1["CB"] if "CB" in self.res1 else self.res1["CA"]
            atom2 = self.res2["CB"] if "CB" in self.res2 else self.res2["CA"]
            return atom1 - atom2
        except KeyError:
            return float('inf')
    
    def _calculate_min_atom_distance(self):
        """Calculate minimum distance between any atoms in the two residues."""
        min_dist = float('inf')
        for atom1 in self.res1:
            for atom2 in self.res2:
                if atom1.element != "H" and atom2.element != "H":  # Skip hydrogens if present
                    dist = atom1 - atom2
                    min_dist = min(min_dist, dist)
        return min_dist
    
    def _calculate_centroid_distance(self):
        """Calculate distance between centroids of residues."""
        centroid1 = self._get_residue_centroid(self.res1)
        centroid2 = self._get_residue_centroid(self.res2)
        return np.linalg.norm(centroid1 - centroid2)
    
    def _get_residue_centroid(self, residue):
        """Calculate centroid of a residue."""
        coords = []
        for atom in residue:
            if atom.element != "H":  # Skip hydrogens
                coords.append(atom.get_coord())
        if not coords:
            return np.array([0, 0, 0])
        return np.mean(coords, axis=0)
    
    def _detect_interaction_types(self):
        """Detect all possible interaction types between the two residues."""
        interactions = []
        
        # Check for disulfide bond
        if self._is_disulfide_bond():
            interactions.append("DS")
        
        # Check for hydrogen bond
        if self._is_hydrogen_bond():
            interactions.append("HB")
        
        # Check for salt bridge
        if self._is_salt_bridge():
            interactions.append("SB")
        
        # Check for pi-pi stacking
        if self._is_pi_pi_stacking():
            interactions.append("PP")
        
        # Check for pi-cation interaction
        if self._is_pi_cation():
            interactions.append("PC")
        
        # Check for hydrophobic interaction
        if self._is_hydrophobic():
            interactions.append("HP")
        
        # Check for van der Waals interaction
        if self._is_van_der_waals():
            interactions.append("VW")
        
        # Check for backbone-backbone interaction
        if self._is_backbone_backbone():
            interactions.append("BB")
        
        # Check for backbone-sidechain interaction
        if self._is_backbone_sidechain():
            interactions.append("BS")
        
        # Check for sidechain-sidechain interaction
        if self._is_sidechain_sidechain():
            interactions.append("SS")
        
        return interactions
    
    def _is_disulfide_bond(self):
        """Check if there is a disulfide bond between two CYS residues."""
        if self.res1_name == "CYS" and self.res2_name == "CYS":
            try:
                sg1 = self.res1["SG"]
                sg2 = self.res2["SG"]
                distance = sg1 - sg2
                return distance <= INTERACTION_DISTANCES["DS"]["max"]
            except KeyError:
                return False
        return False
    
    def _is_hydrogen_bond(self):
        """Check for hydrogen bond between donor and acceptor atoms."""
        # Define donor and acceptor atoms
        donors = ["N", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2", "NZ", "OG", "OG1", "OH"]
        acceptors = ["O", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SD", "ND1", "NE2"]
        
        min_dist = float('inf')
        for atom1 in self.res1:
            if atom1.name in donors:
                for atom2 in self.res2:
                    if atom2.name in acceptors:
                        dist = atom1 - atom2
                        if dist < min_dist:
                            min_dist = dist
                            
        for atom2 in self.res2:
            if atom2.name in donors:
                for atom1 in self.res1:
                    if atom1.name in acceptors:
                        dist = atom1 - atom2
                        if dist < min_dist:
                            min_dist = dist
                            
        return INTERACTION_DISTANCES["HB"]["min"] <= min_dist <= INTERACTION_DISTANCES["HB"]["max"]
    
    def _is_salt_bridge(self):
        """Check for salt bridge between charged residues."""
        pos_charged = ["ARG", "LYS", "HIS"]
        neg_charged = ["ASP", "GLU"]
        
        # Basic check if residue types are compatible
        if (self.res1_name in pos_charged and self.res2_name in neg_charged) or \
           (self.res1_name in neg_charged and self.res2_name in pos_charged):
            # Check charged atom distances
            for atom1_name in CHARGED_ATOMS.get(self.res1_name, []):
                for atom2_name in CHARGED_ATOMS.get(self.res2_name, []):
                    try:
                        atom1 = self.res1[atom1_name]
                        atom2 = self.res2[atom2_name]
                        dist = atom1 - atom2
                        if INTERACTION_DISTANCES["SB"]["min"] <= dist <= INTERACTION_DISTANCES["SB"]["max"]:
                            return True
                    except KeyError:
                        continue
        return False
    
    def _is_pi_pi_stacking(self):
        """Check for pi-pi stacking between aromatic residues."""
        aromatic = ["PHE", "TYR", "TRP", "HIS"]
        if self.res1_name in aromatic and self.res2_name in aromatic:
            # Calculate centroids of aromatic rings
            centroid1 = self._get_aromatic_centroid(self.res1)
            centroid2 = self._get_aromatic_centroid(self.res2)
            if centroid1 is not None and centroid2 is not None:
                # Calculate distance between centroids
                distance = np.linalg.norm(centroid1 - centroid2)
                return INTERACTION_DISTANCES["PP"]["min"] <= distance <= INTERACTION_DISTANCES["PP"]["max"]
        return False
    
    def _get_aromatic_centroid(self, residue):
        """Get centroid of aromatic ring."""
        if residue.get_resname() not in AROMATIC_ATOMS:
            return None
        
        coords = []
        for atom_name in AROMATIC_ATOMS[residue.get_resname()]:
            try:
                coords.append(residue[atom_name].get_coord())
            except KeyError:
                pass
        
        if not coords:
            return None
        return np.mean(coords, axis=0)
    
    def _is_pi_cation(self):
        """Check for pi-cation interaction."""
        aromatic = ["PHE", "TYR", "TRP", "HIS"]
        cationic = ["ARG", "LYS"]
        
        # Check if one is aromatic and the other is cationic
        if (self.res1_name in aromatic and self.res2_name in cationic) or \
           (self.res1_name in cationic and self.res2_name in aromatic):
            # Determine which is which
            if self.res1_name in aromatic:
                aromatic_res = self.res1
                cationic_res = self.res2
            else:
                aromatic_res = self.res2
                cationic_res = self.res1
            
            # Get aromatic centroid
            aromatic_centroid = self._get_aromatic_centroid(aromatic_res)
            if aromatic_centroid is None:
                return False
            
            # Check distances to charged atoms
            for atom_name in CHARGED_ATOMS.get(cationic_res.get_resname(), []):
                try:
                    atom = cationic_res[atom_name]
                    dist = np.linalg.norm(aromatic_centroid - atom.get_coord())
                    if INTERACTION_DISTANCES["PC"]["min"] <= dist <= INTERACTION_DISTANCES["PC"]["max"]:
                        return True
                except KeyError:
                    continue
        return False
    
    def _is_hydrophobic(self):
        """Check for hydrophobic interaction."""
        hydrophobic_res = ["ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "GLY"]
        if self.res1_name in hydrophobic_res and self.res2_name in hydrophobic_res:
            return self.min_atom_distance <= INTERACTION_DISTANCES["HP"]["max"]
        return False
    
    def _is_van_der_waals(self):
        """Check for van der Waals interaction."""
        return self.min_atom_distance <= INTERACTION_DISTANCES["VW"]["max"]
    
    def _is_backbone_backbone(self):
        """Check for backbone-backbone interaction."""
        for atom1_name in BACKBONE_ATOMS:
            for atom2_name in BACKBONE_ATOMS:
                try:
                    atom1 = self.res1[atom1_name]
                    atom2 = self.res2[atom2_name]
                    dist = atom1 - atom2
                    if dist <= INTERACTION_DISTANCES["BB"]["max"]:
                        return True
                except KeyError:
                    continue
        return False
    
    def _is_backbone_sidechain(self):
        """Check for backbone-sidechain interaction."""
        for atom1_name in BACKBONE_ATOMS:
            try:
                atom1 = self.res1[atom1_name]
                for atom2 in self.res2:
                    if atom2.name not in BACKBONE_ATOMS:
                        dist = atom1 - atom2
                        if dist <= INTERACTION_DISTANCES["BS"]["max"]:
                            return True
            except KeyError:
                continue
        
        for atom2_name in BACKBONE_ATOMS:
            try:
                atom2 = self.res2[atom2_name]
                for atom1 in self.res1:
                    if atom1.name not in BACKBONE_ATOMS:
                        dist = atom1 - atom2
                        if dist <= INTERACTION_DISTANCES["BS"]["max"]:
                            return True
            except KeyError:
                continue
        
        return False
    
    def _is_sidechain_sidechain(self):
        """Check for sidechain-sidechain interaction."""
        for atom1 in self.res1:
            if atom1.name not in BACKBONE_ATOMS:
                for atom2 in self.res2:
                    if atom2.name not in BACKBONE_ATOMS:
                        dist = atom1 - atom2
                        if dist <= INTERACTION_DISTANCES["SS"]["max"]:
                            return True
        return False
    
    def _calculate_interaction_strength(self):
        """Calculate the strength of the strongest interaction."""
        if not self.interaction_types:
            return 0
        
        # Assign strength based on distance and interaction type
        strengths = []
        
        for interaction_type in self.interaction_types:
            strength = 0
            
            if interaction_type == "DS":
                # Disulfide bond strength based on SG-SG distance
                try:
                    sg_dist = self.res1["SG"] - self.res2["SG"]
                    if sg_dist <= 2.05:
                        strength = 5  # Very strong
                    elif sg_dist <= 2.2:
                        strength = 4  # Strong
                    else:
                        strength = 3  # Moderate
                except KeyError:
                    strength = 3
            
            elif interaction_type == "HB":
                # Hydrogen bond strength
                if self.min_atom_distance <= 2.8:
                    strength = 4  # Strong
                elif self.min_atom_distance <= 3.2:
                    strength = 3  # Moderate
                
