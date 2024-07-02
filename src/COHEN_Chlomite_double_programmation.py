"""Design of a threading program by double dynamic programming.

Usage:
======
    python COHEN_threading_double_progra.py argument1 argument2

    argument1: une chaîne de caractères correspondant a un fichier .pdb
    argument2: une chaîne de caractères correspondant a un fichier .fasta
"""

__authors__ = "Chlomite Cohen"
__contact__ = "chlomitecohen@gmail.com"
__copyright__ = "CHLO"
__date__ = "14-09-2023"
__version__ = "1.0"

import numpy as np
from Bio import SeqIO
from Bio import PDB
import argparse


def three_letter_to_one(three_letter):
    """
    Convert three-letter amino acid code to one-letter code.

    Parameters
    ----------
    three_letter : str
        The three-letter amino acid code to be converted.

    Returns
    -------
    str
        The one-letter amino acid code corresponding to the input.
        If the input code is not found in the dictionary, it returns "unknown".

    Examples
    --------
    >>> three_letter_to_one("ALA")
    'A'

    >>> three_letter_to_one("LEU")
    'L'

    >>> three_letter_to_one("XYZ")
    'unknown'
    """
    # Dictionary for mapping three-letter
    # amino acid codes to one-letter codes
    corresponding_aa = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }

    # Check if the three-letter code is in the mapping dictionary
    if three_letter in corresponding_aa:
        return corresponding_aa[three_letter]
    else:
        # If the three-letter code is not found, return an "unknown" value
        return "unknown"


def read_pdb_file(pdb_file):
    """
    Read a PDB file and extract amino acid sequence information.

    Parameters
    ----------
    pdb_file : str
        The path to the input PDB file.

    Returns
    -------
    list
        A list of tuples, where each tuple contains the position index and
        the one-letter amino acid code extracted from the PDB file.
    """
    # Create a PDBParser with QUIET mode to suppress warnings
    parser = PDB.PDBParser(QUIET=True)

    # Parse the PDB file and create a structure object
    structure = parser.get_structure("Structure", pdb_file)

    # List to store amino acid names
    pdb_seq = []

    # Select the first chain in the structure (e.g., chain 'A')
    chain = structure[0]["A"]

    # Iterate through the structure and extract amino acids
    for index, residue in enumerate(chain.get_residues()):
        if PDB.is_aa(residue):
            # Get the one-letter amino acid code (e.g., 'A', 'L', etc.)
            aa_1_letter = three_letter_to_one(residue.get_resname())
            pdb_seq.append((index, aa_1_letter))
    return pdb_seq


def read_fasta_file(fasta_file):
    """
    Read a FASTA file and extract sequence information.

    Parameters
    ----------
    fasta_file : str
        The path to the input FASTA file.

    Returns
    -------
    list
        A list of tuples, where each tuple contains the position index and
        a one-letter amino acid code extracted from the FASTA file.
    """
    # Create an empty list to store amino acid information
    fasta_seq = []

    # Open the FASTA file for reading ('r' mode) using a context manager
    with open(fasta_file, "r") as f:
        # Read the first record from the FASTA file
        record = SeqIO.read(f, "fasta")

        # Extract the amino acid sequence as a string
        sequence = str(record.seq)

        # Iterate through the sequence and extract amino acid information
        for index, aa in enumerate(sequence):
            # Append a tuple containing the position (index)
            # and one-letter amino acid code
            fasta_seq.append((index, aa))

    # Return the list containing amino acid information
    return fasta_seq


def extract_ca_coordinates(pdb_file):
    """
    Extract the coordinates of alpha carbon (CA) atoms from a PDB file.

    Parameters
    ----------
    pdb_file : str
        The path to the input PDB file.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the CA
        coordinates extracted from the PDB file.
    """
    # Create a PDBParser to parse the PDB file
    parser = PDB.PDBParser()

    # Parse the PDB file and create a structure object
    structure = parser.get_structure("template", pdb_file)

    # Create an empty list to store CA atom coordinates
    ca_coordinates = []

    # Iterate through the structure hierarchy
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue contains a "CA" atom
                if "CA" in residue:
                    # Get the coordinates of the "CA" atom
                    ca_coord = residue["CA"].coord
                    ca_coordinates.append(ca_coord)

    # Convert the list of coordinates into a NumPy array
    # for efficient manipulation
    return np.array(ca_coordinates)


def distance_matrix_between_ca(ca_coordinates_matrix):
    """
    Calculate the distance matrix between CA atoms.

    Parameters
    ----------
    ca_coordinates_matrix : numpy.ndarray
        A NumPy array containing the CA coordinates.

    Returns
    -------
    numpy.ndarray
        A distance matrix representing the pairwise distances between CA atoms.
    """
    # We subtract each CA atom's coordinates from all other
    # CA atoms' coordinates and then calculate
    # the norm (Euclidean distance) along axis 2.
    distance_matrix = np.linalg.norm(
        ca_coordinates_matrix[:, np.newaxis, :] - ca_coordinates_matrix, axis=2
    )
    return distance_matrix


def read_score_dope(dope_file):
    """
    Read a DOPE score file and organize the data for scoring.

    Parameters
    ----------
    dope_file : str
        The path to the input DOPE score file.

    Returns
    -------
    dict
        A dictionary where keys are pairs of one-letter amino acid codes
        (e.g., 'A A' for two alanines) and values are sub-dictionaries mapping
        distance values to corresponding DOPE scores.
    """
    # Initialize an empty dictionary to store DOPE scores
    dope_score_dist = {}

    # Define increment value and distance threshold
    increment_value = 0.5
    distance = 0.75

    # Open the DOPE score file in read mode
    with open(dope_file, "r") as file:
        for line in file:
            # Split the line into a list of tokens
            aa = line.split()

            # Check if the line has at least 4 indices
            if len(aa) >= 4:
                # Check if the second and fourth indices are "CA"
                if aa[1] == "CA" and aa[3] == "CA":
                    # Convert three-letter amino acid codes to one-letter codes
                    aa_1 = three_letter_to_one(aa[0])
                    aa_2 = three_letter_to_one(aa[2])

                    # Create a unique key for the amino acid pair and distance
                    distance_aa1_aa2 = f"{aa_1} {aa_2}"

                    # Extract DOPE scores as floats
                    scores = [float(score) for score in aa[4:]]

                    # Create a dictionary of distance scores
                    # using a comprehension
                    distance_scores = {
                        distance + i * increment_value: -score
                        for i, score in enumerate(scores)
                    }

                    # Add the distance scores to the DOPE score dictionary
                    dope_score_dist[distance_aa1_aa2] = distance_scores

    # Return the organized DOPE score dictionary
    return dope_score_dist


def choose_value(dope_score_aa, target_value):
    """
    Choose the DOPE score value based on a target distance.

    Find the DOPE score key (distance) that minimizes the absolute difference
    between the key and the target value using a lambda function.

    Parameters
    ----------
    dope_score_aa : dict
        A sub-dictionary containing DOPE scores for a specific amino acid pair.
    target_value : float
        The target distance value for which to select a DOPE score.

    Returns
    -------
    float
        The selected DOPE score for the given target distance.
    """
    chosen_value = min(dope_score_aa, key=lambda key: abs(key - target_value))
    return chosen_value


def initialize_low_lvl_matrix(pos_sequence, pos_template, sequence,
                              distance_matrix):
    """
    Initialize a matrix for low-level sequence alignment.

    Parameters
    ----------
    pos_sequence : int
        Position of the sequence residue.
    pos_template : int
        Position of the template residue.
    sequence : list
        A list of tuples containing sequence residue positions
        and one-letter amino acid codes.
    distance_matrix : numpy.ndarray
        A distance matrix representing pairwise distances between CA atoms.

    Returns
    -------
    numpy.ndarray
        An initialized low-level alignment matrix.
    """
    # Add one to positions to account for gap columns and rows
    pos_sequence += 1
    pos_template += 1

    # Get the lengths of the sequences
    len_sequence = len(sequence)
    len_template = len(distance_matrix)

    # Initialize the matrix with zeros
    low_lvl_matrix = np.zeros((len_sequence + 1, len_template + 1))

    # Set the entire row and column to infinity except
    # for the specified residue position
    low_lvl_matrix[pos_sequence, :] = np.inf
    low_lvl_matrix[:, pos_template] = np.inf

    # Set cells i > residue_position and j < residue_position to infinity
    low_lvl_matrix[pos_sequence + 1:, :pos_template] = np.inf

    # Set cells i < residue_position and j > residue_position to infinity
    low_lvl_matrix[:pos_sequence, pos_template + 1:] = np.inf

    # Set the initial score for the specified residue position to 0
    low_lvl_matrix[pos_sequence, pos_template] = 0

    return low_lvl_matrix


def update_low_lvl_matrix(
    low_lvl_matrix, i, j, seq_list, pos_sequence, pos_template, dope_score,
    distance_matrix, gap, low_lvl_score
):
    """
    Update the low-level alignment matrix with gap penalties and scores.

    Parameters
    ----------
    low_lvl_matrix : numpy.ndarray
        The low-level alignment matrix.
    i : int
        Current row index in the matrix.
    j : int
        Current column index in the matrix.
    seq_list : list
        A list of tuples containing sequence residue positions
        and one-letter amino acid codes.
    pos_sequence : int
        Position of the sequence residue.
    pos_template : int
        Position of the template residue.
    dope_score : dict
        A dictionary containing DOPE scores for amino acid pairs and distances.
    distance_matrix : numpy.ndarray
        A distance matrix representing pairwise distances between CA atoms.
    gap : float
        The gap penalty value.
    low_lvl_score : float
        The current low-level alignment score.

    Returns
    -------
    float
        The updated low-level alignment score.
    """
    # Construct a string representing the amino acid pair (e.g., "A L")
    aa = f"{seq_list[i - 1][1]} {seq_list[pos_sequence - 1][1]}"

    # Choose the DOPE score value for the current amino acid pair and distance
    val_dist = choose_value(dope_score[aa],
                            distance_matrix[j - 1][pos_template - 1])

    # Retrieve the DOPE score for the chosen amino acid pair and distance
    score = dope_score[aa][val_dist]

    # Calculate scores for different operations: delete, insert, and match
    delete = low_lvl_matrix[i - 1][j] + gap  # Gap from above
    insert = low_lvl_matrix[i][j - 1] + gap  # Gap from the left
    match = low_lvl_matrix[i - 1][j - 1] + score

    # Update the current cell with the minimum of delete, insert, and match
    low_lvl_matrix[i][j] = min(delete, insert, match)

    # Update the low-level score for the current cell
    low_score = low_lvl_matrix[i][j]

    return low_score


def score_low_lvl_matrix(
    pos_sequence, pos_template, seq_list, dope_score, distance_matrix, gap
):
    """
    Score the low-level alignment matrix for a given sequence position.

    Parameters
    ----------
    pos_sequence : int
        Position of the sequence residue.
    pos_template : int
        Position of the template residue.
    seq_list : list
        A list of tuples containing sequence residue positions
        and one-letter amino acid codes.
    dope_score : dict
        A dictionary containing DOPE scores for amino acid pairs and distances.
    distance_matrix : numpy.ndarray
        A distance matrix representing pairwise distances between CA atoms.

    Returns
    -------
    float
        The low-level alignment score for the specified position.
    """
    # Initialize the low-level matrix
    low_lvl_matrix = initialize_low_lvl_matrix(
        pos_sequence, pos_template, seq_list, distance_matrix)

    # Initialize the low-level score variable to None
    low_lvl_score = None

    # Add one to the positions to account for gap columns and rows
    pos_sequence += 1
    pos_template += 1

    # Fill the first row and the first column with gap penalties
    low_lvl_matrix[:pos_sequence, 0] = np.arange(pos_sequence) * gap
    low_lvl_matrix[0, :pos_template] = np.arange(pos_template) * gap

    # Fill the first sub-matrix
    for i in range(1, pos_sequence):
        for j in range(1, pos_template):
            if low_lvl_matrix[i][j] == 0:
                low_lvl_score = update_low_lvl_matrix(
                    low_lvl_matrix, i, j, seq_list, pos_sequence, pos_template,
                    dope_score,
                    distance_matrix,
                    gap,
                    low_lvl_score
                )

    # Fill the second sub-matrix if it exists
    if (pos_sequence + 1 < low_lvl_matrix.shape[0]) and (
        pos_template + 1 < low_lvl_matrix.shape[1]
    ):
        aa = f"{seq_list[pos_sequence][1]} {seq_list[pos_sequence][1]}"
        val_dist = choose_value(
            dope_score[aa], distance_matrix[pos_template][pos_template - 1]
        )
        score = dope_score[aa][val_dist]
        low_lvl_matrix[pos_sequence + 1][pos_template + 1] = (
            low_lvl_matrix[pos_sequence - 1][pos_template - 1] + score
        )

        for i in range(pos_sequence + 1, low_lvl_matrix.shape[0]):
            for j in range(pos_template + 1, low_lvl_matrix.shape[1]):
                if (i == pos_sequence + 1) and (j == pos_template + 1):
                    continue
                if low_lvl_matrix[i][j] == 0:
                    low_lvl_score = update_low_lvl_matrix(
                        low_lvl_matrix, i, j, seq_list, pos_sequence,
                        pos_template,
                        dope_score,
                        distance_matrix,
                        gap,
                        low_lvl_score
                    )

    # Check if low_lvl_score is still None,
    # indicating that no scores were updated
    if low_lvl_score is None:
        low_lvl_score = low_lvl_matrix[pos_sequence - 1][pos_template - 1] + 1

    # If pos_sequence reaches the end of the matrix, apply gap penalties
    if pos_sequence == (low_lvl_matrix.shape[0] - 1):
        # Create a vector of gap penalties with
        # the same length as the remaining columns
        gap = np.full(low_lvl_matrix.shape[1] - pos_template, gap)
        # Add the gap penalty vector to the low_lvl_score
        low_lvl_score += np.sum(gap)

    # If pos_template reaches the end of the matrix, apply gap penalties
    if pos_template == (low_lvl_matrix.shape[1] - 1):
        # Create a vector of gap penalties with
        # the same length as the remaining rows
        gap = np.full(low_lvl_matrix.shape[0] - pos_sequence, gap)
        # Add the gap penalty vector to the low_lvl_score
        low_lvl_score += np.sum(gap)

    return low_lvl_score


def score_high_lvl_matrix(seq_list, dist_ca, dope_score, gap):
    """
    Score the high-level alignment matrix.

    Parameters
    ----------
    seq_list : list
        A list of tuples containing sequence residue positions
        and one-letter amino acid codes.
    dist_ca : numpy.ndarray
        A NumPy array containing the pairwise distances between CA atoms.
    dope_score : dict
        A dictionary containing DOPE scores for amino acid pairs and distances.
    gap : float
        The gap penalty value.

    Returns
    -------
    numpy.ndarray
        A high-level alignment matrix representing alignment scores
        between sequence and template.
    """
    # Calculate the length of the sequence and template
    len_seq = len(seq_list)
    len_template = len(dist_ca)

    # Use a list comprehension to create the matrix of scores
    matrix = [
        [
            score_low_lvl_matrix(i, j, seq_list, dope_score, dist_ca, gap)
            for j in range(len_template)
        ]
        for i in range(len_seq)
    ]

    # Convert the list of lists into a NumPy array
    matrix = np.array(matrix)

    return matrix


def global_alignment(high_matrix, seq_list, template_seq):
    """
    Perform global sequence alignment using dynamic programming.

    Parameters
    ----------
    high_matrix : numpy.ndarray
        High-level scoring matrix for structural comparison.
    seq_list : list
        List of tuples containing the amino acid sequence to align.
    template_seq : str
        Amino acid sequence of the template.

    Returns
    -------
    numpy.ndarray
        The final dynamic programming matrix used in alignment.
    str
        Aligned sequence based on the global alignment.
    str
        Aligned template sequence based on the global alignment.

    This function initializes a scoring matrix, fills it
    using dynamic programming, and performs traceback to
    obtain the optimal global sequence alignment.

    """
    # Create a matrix to store alignment scores,
    # adding one extra row and column
    final_matrix = np.zeros((high_matrix.shape[0] + 1,
                             high_matrix.shape[1] + 1))

    # Initialize the first column of the matrix with
    # gap penalties based on row index
    final_matrix[:, 0] = np.arange(final_matrix.shape[0]) * gap

    # Initialize the first row of the matrix with
    # gap penalties based on column index
    final_matrix[0, :] = np.arange(final_matrix.shape[1]) * gap

    # Fill in the scoring matrix using dynamic programming
    for i in range(1, final_matrix.shape[0]):
        for j in range(1, final_matrix.shape[1]):
            score = high_matrix[i - 1][j - 1]
            delete = final_matrix[i - 1][j] + gap
            insert = final_matrix[i][j - 1] + gap
            match = final_matrix[i - 1][j - 1] + score

            # Update the current cell in the matrix with
            # the minimum of delete, insert, and match
            final_matrix[i][j] = min(delete, insert, match)

    # Traceback to obtain the optimal alignment
    aligned_seq = []
    aligned_template = []

    # Initialize indices for traceback starting from the end of sequences
    i, j = len(seq_list), len(template_seq)
    while i > 0 and j > 0:
        diag = final_matrix[i - 1][j - 1] + high_matrix[i - 1][j - 1]
        haut = final_matrix[i - 1][j] + gap

        if i > 0 and j > 0 and final_matrix[i][j] == diag:
            # Diagonal movement, indicating a matching residue
            aligned_seq.append(seq_list[i - 1][1])
            aligned_template.append(template_seq[j - 1][1])
            i -= 1
            j -= 1
        elif i > 0 and final_matrix[i][j] == haut:
            # Vertical movement, indicating a gap in the template sequence
            aligned_seq.append(seq_list[i - 1][1])
            aligned_template.append('-')
            i -= 1
        else:
            # Horizontal movement, indicating a gap in the sequence
            aligned_seq.append('-')
            aligned_template.append(template_seq[j - 1][1])
            j -= 1

    # Reverse the aligned sequences to obtain the correct order
    aligned_seq = ''.join(reversed(aligned_seq))
    aligned_template = ''.join(reversed(aligned_template))

    # Return the final alignment matrix, aligned sequence, and aligned template
    return final_matrix, aligned_seq, aligned_template


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A threading program by double dynamic programming."
        )
    parser.add_argument("-p", "--pdb_file", type=str, default=0,
                        help="PDB file")
    parser.add_argument("-f", "--fasta_file", type=str, default=0,
                        help="FASTA file")
    parser.add_argument("-d", "--dope_file", type=str, default=0,
                        help="VDOPE SCORE file")
    parser.add_argument("-g", "--gap", type=int, default=0,
                        help="Valeur du gap à utiliser (par défaut : 0)")
    args = parser.parse_args()
    PDB_FILE = args.pdb_file
    FASTA_FILE = args.fasta_file
    DOPE_FILE = args.dope_file
    gap = args.gap
    if not PDB_FILE.endswith('.pdb'):
        print("Please select file with pdb extension")

    if not FASTA_FILE.endswith('.fasta'):
        print("Please select file with pdb extension")

    else:
        template_seq = read_pdb_file(PDB_FILE)
        ca_coordinates = extract_ca_coordinates(PDB_FILE)
        dist_ca = distance_matrix_between_ca(ca_coordinates)
        structure_seq = read_fasta_file(FASTA_FILE)
        dope_score = read_score_dope(DOPE_FILE)
        matrix = score_high_lvl_matrix(structure_seq, dist_ca, dope_score, gap)
        align_matrix, seq, template = global_alignment(matrix,
                                                       structure_seq,
                                                       template_seq)
        print("\n---------------------")
        print("| Results Alignment |")
        print("---------------------\n")
        print(f"Sequence : {seq} \n")
        print(f"Template : {template} \n")