import numpy as np
from Bio import SeqIO
from Bio import PDB
import argparse

def three_letter_to_one(three_letter):
    corresponding_aa = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
        "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V",
    }
    return corresponding_aa.get(three_letter, "unknown")

def read_pdb_file(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("Structure", pdb_file)
    pdb_seq = []
    chain = structure[0]["A"]
    for index, residue in enumerate(chain.get_residues()):
        if PDB.is_aa(residue):
            aa_1_letter = three_letter_to_one(residue.get_resname())
            pdb_seq.append((index, aa_1_letter))
    return pdb_seq

def read_fasta_file(fasta_file):
    fasta_seq = []
    with open(fasta_file, "r") as f:
        record = SeqIO.read(f, "fasta")
        sequence = str(record.seq)
        for index, aa in enumerate(sequence):
            fasta_seq.append((index, aa))
    return fasta_seq

def extract_ca_coordinates(pdb_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure("template", pdb_file)
    ca_coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coord = residue["CA"].coord
                    ca_coordinates.append(ca_coord)
    return np.array(ca_coordinates)

def distance_matrix_between_ca(ca_coordinates_matrix):
    distance_matrix = np.linalg.norm(
        ca_coordinates_matrix[:, np.newaxis, :] - ca_coordinates_matrix, axis=2
    )
    return distance_matrix

def read_score_dope(dope_file):
    dope_score_dist = {}
    increment_value = 0.5
    distance = 0.75
    with open(dope_file, "r") as file:
        for line in file:
            aa = line.split()
            if len(aa) >= 4:
                if aa[1] == "CA" and aa[3] == "CA":
                    aa_1 = three_letter_to_one(aa[0])
                    aa_2 = three_letter_to_one(aa[2])
                    distance_aa1_aa2 = f"{aa_1} {aa_2}"
                    scores = [float(score) for score in aa[4:]]
                    distance_scores = {
                        distance + i * increment_value: -score
                        for i, score in enumerate(scores)
                    }
                    dope_score_dist[distance_aa1_aa2] = distance_scores
    return dope_score_dist

def choose_value(dope_score_aa, target_value):
    chosen_value = min(dope_score_aa, key=lambda key: abs(key - target_value))
    return chosen_value

def initialize_low_lvl_matrix(pos_sequence, pos_template, sequence, distance_matrix):
    pos_sequence += 1
    pos_template += 1
    len_sequence = len(sequence)
    len_template = len(distance_matrix)
    low_lvl_matrix = np.zeros((len_sequence + 1, len_template + 1))
    low_lvl_matrix[pos_sequence, :] = np.inf
    low_lvl_matrix[:, pos_template] = np.inf
    low_lvl_matrix[pos_sequence + 1:, :pos_template] = np.inf
    low_lvl_matrix[:pos_sequence, pos_template + 1:] = np.inf
    low_lvl_matrix[pos_sequence, pos_template] = 0
    return low_lvl_matrix

def update_low_lvl_matrix(
    low_lvl_matrix, i, j, seq_list, pos_sequence, pos_template, dope_score,
    distance_matrix, gap, low_lvl_score
):
    aa = f"{seq_list[i - 1][1]} {seq_list[pos_sequence - 1][1]}"
    val_dist = choose_value(dope_score[aa], distance_matrix[j - 1][pos_template - 1])
    score = dope_score[aa][val_dist]
    delete = low_lvl_matrix[i - 1][j] + gap
    insert = low_lvl_matrix[i][j - 1] + gap
    match = low_lvl_matrix[i - 1][j - 1] + score
    low_lvl_matrix[i][j] = min(delete, insert, match)
    low_score = low_lvl_matrix[i][j]
    return low_score

def score_low_lvl_matrix(
    pos_sequence, pos_template, seq_list, dope_score, distance_matrix, gap
):
    low_lvl_matrix = initialize_low_lvl_matrix(pos_sequence, pos_template, seq_list, distance_matrix)
    low_lvl_score = None
    pos_sequence += 1
    pos_template += 1
    low_lvl_matrix[:pos_sequence, 0] = np.arange(pos_sequence) * gap
    low_lvl_matrix[0, :pos_template] = np.arange(pos_template) * gap
    for i in range(1, pos_sequence):
        for j in range(1, pos_template):
            if low_lvl_matrix[i][j] == 0:
                low_lvl_score = update_low_lvl_matrix(
                    low_lvl_matrix, i, j, seq_list, pos_sequence, pos_template,
                    dope_score, distance_matrix, gap, low_lvl_score
                )
    if (pos_sequence + 1 < low_lvl_matrix.shape[0]) and (pos_template + 1 < low_lvl_matrix.shape[1]):
        aa = f"{seq_list[pos_sequence][1]} {seq_list[pos_sequence][1]}"
        val_dist = choose_value(dope_score[aa], distance_matrix[pos_template][pos_template - 1])
        score = dope_score[aa][val_dist]
        low_lvl_matrix[pos_sequence + 1][pos_template + 1] = low_lvl_matrix[pos_sequence - 1][pos_template - 1] + score
        for i in range(pos_sequence + 1, low_lvl_matrix.shape[0]):
            for j in range(pos_template + 1, low_lvl_matrix.shape[1]):
                if (i == pos_sequence + 1) and (j == pos_template + 1):
                    continue
                if low_lvl_matrix[i][j] == 0:
                    low_lvl_score = update_low_lvl_matrix(
                        low_lvl_matrix, i, j, seq_list, pos_sequence, pos_template,
                        dope_score, distance_matrix, gap, low_lvl_score
                    )
    if low_lvl_score is None:
        low_lvl_score = low_lvl_matrix[pos_sequence - 1][pos_template - 1] + 1
    if pos_sequence == (low_lvl_matrix.shape[0] - 1):
        gap = np.full(low_lvl_matrix.shape[1] - pos_template, gap)
        low_lvl_score += np.sum(gap)
    if pos_template == (low_lvl_matrix.shape[1] - 1):
        gap = np.full(low_lvl_matrix.shape[0] - pos_sequence, gap)
        low_lvl_score += np.sum(gap)
    return low_lvl_score

def score_high_lvl_matrix(seq_list, dist_ca, dope_score, gap):
    len_seq = len(seq_list)
    len_template = len(dist_ca)
    matrix = [
        [
            score_low_lvl_matrix(i, j, seq_list, dope_score, dist_ca, gap)
            for j in range(len_template)
        ]
        for i in range(len_seq)
    ]
    return np.array(matrix)

def global_alignment(high_matrix, seq_list, template_seq):
    final_matrix = np.zeros((high_matrix.shape[0] + 1, high_matrix.shape[1] + 1))
    final_matrix[:, 0] = np.arange(final_matrix.shape[0]) * gap
    final_matrix[0, :] = np.arange(final_matrix.shape[1]) * gap
    for i in range(1, final_matrix.shape[0]):
        for j in range(1, final_matrix.shape[1]):
            score = high_matrix[i - 1][j - 1]
            delete = final_matrix[i - 1][j] + gap
            insert = final_matrix[i][j - 1] + gap
            match = final_matrix[i - 1][j - 1] + score
            final_matrix[i][j] = min(delete, insert, match)
    aligned_seq = []
    aligned_template = []
    i, j = len(seq_list), len(template_seq)
    while i > 0 and j > 0:
        diag = final_matrix[i - 1][j - 1] + high_matrix[i - 1][j - 1]
        haut = final_matrix[i - 1][j] + gap
        gauche = final_matrix[i][j - 1] + gap
        if final_matrix[i][j] == diag:
            aligned_seq.append(seq_list[i - 1][1])
            aligned_template.append(template_seq[j - 1][1])
            i -= 1
            j -= 1
        elif final_matrix[i][j] == haut:
            aligned_seq.append(seq_list[i - 1][1])
            aligned_template.append('-')
            i -= 1
        else:
            aligned_seq.append('-')
            aligned_template.append(template_seq[j - 1][1])
            j -= 1
    aligned_seq = ''.join(reversed(aligned_seq))
    aligned_template = ''.join(reversed(aligned_template))
    return final_matrix, aligned_seq, aligned_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A threading program by double dynamic programming.")
    parser.add_argument("-p", "--pdb_file", type=str, default=0, help="PDB file")
    parser.add_argument("-f", "--fasta_file", type=str, default=0, help="FASTA file")
    parser.add_argument("-d", "--dope_file", type=str, default=0, help="VDOPE SCORE file")
    parser.add_argument("-g", "--gap", type=int, default=0, help="Valeur du gap à utiliser (par défaut : 0)")
    args = parser.parse_args()
    PDB_FILE = args.pdb_file
    FASTA_FILE = args.fasta_file
    DOPE_FILE = args.dope_file
    gap = args.gap
    if not PDB_FILE.endswith('.pdb'):
        print("Please select file with pdb extension")
    if not FASTA_FILE.endswith('.fasta'):
        print("Please select file with fasta extension")
    else:
        template_seq = read_pdb_file(PDB_FILE)
        ca_coordinates = extract_ca_coordinates(PDB_FILE)
        dist_ca = distance_matrix_between_ca(ca_coordinates)
        structure_seq = read_fasta_file(FASTA_FILE)
        dope_score = read_score_dope(DOPE_FILE)
        matrix = score_high_lvl_matrix(structure_seq, dist_ca, dope_score, gap)
        align_matrix, seq, template = global_alignment(matrix, structure_seq, template_seq)
        print("\n---------------------")
        print("| Results Alignment |")
        print("---------------------\n")
        print(f"Sequence : {seq} \n")
        print(f"Template : {template} \n")
