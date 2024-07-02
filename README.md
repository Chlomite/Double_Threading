# Threading avec Double Programmation Dynamique (DDP_threading)

Le threading (enfilage) est une stratégie pour rechercher des séquences compatibles avec une structure.


## Download this repository

```bash
git clone https://gaufre.informatique.univ-paris-diderot.fr/cohenc/cohen_chlomite_double_programmation.git
cd cohen_chlomite_double_programmation
```

## Install dependencies

### Conda environment

Install [conda](https://docs.conda.io/en/latest/miniconda.html).

Install mamba:

```bash
conda install mamba -n base -c conda-forge
```

Create conda environment and install dependendies:

```bash
mamba env create -f environment.yml
```

Load conda environment:

```bash
conda activate DDP_threading
```


## Ressources

### Find structure and sequence

- HOMSTRAD (https://mizuguchilab.org/homstrad/) Homologous Protein Structure Alignment Database : is a collection of protein families, clustered on the basis of sequence and structural similarity.
- With HOMSTRAD, with the file "malform", we can have the Percentage identities betwee two protein:
- Uniprot (and PyMol) for3D structure data and identify secondary structures. 

### Project tree

```
/cohen_chlomite_double_programmation
├── data
│   ├── .fasta files
│   ├── .pdb files
│   └── dope.txt
│
├── src
│   └── COHEN_Chlomite_double_programmation.py
│
├── README.md
│   
└── environment.yml 
```

## Run DDP_threading


```bash
python src/COHEN_Chlomite_double_programmation.py -p data/[filename].pdb -f data/[filename].fast -d data/dope.txt (with the -g option we can change the gap value).
```


## Get results

The results appear directly on the terminal of the alignment between the sequence extracted from the PDB file and the sequence of the template from the FASTA file.

```bash
---------------------
| Results Alignment |
---------------------

Sequence : ... 

Template : ...
```