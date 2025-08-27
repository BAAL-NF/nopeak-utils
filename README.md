# Motif calling prototypes

This repository contains some code that was developed to investigate the use of NoPeak for motif calling in the `baal-nf` pipeline. It also includes custom functions that are implemented in the `baal-nf` pipeline for mapping SNPs to motifs, characterizing NoPeak motifs, and summarizing information across many motif groups. Motif calling has been done with [NoPeak](https://github.com/menzel/nopeak). 

This repository uses poetry for dependency handling and git-lfs for handling large data files. 
To set up the repository, do the following

```bash
conda create -n nopeak-utils python=3.8
conda activate nopeak-utils
pip install poetry
poetry install
```

The code can be broken down as follows:
 - `nopeak.py` contains classes and functions associated with reading in and processing NoPeak motifs with gimmemotifs.
 - `scores.py` contains classes and functions for scoring each sequence instance (REF and ALT instance for each heterozygous SNP) onto a motif-of-interest.
 - `motifutils.py` contains many functions, those of which are used to run various analyses in the `baal-nf` pipeline, with various pre-processing steps, SNP-motif mapping, motif clustering and many more functionalities.

## Docker container

In order to build the docker container, create a file called `credentials.sh` (it is in .gitignore), and set the following variables to conform with a gitlab access token that has read/write permissions to the package registry
```
export POETRY_HTTP_BASIC_GITLAB_USERNAME=<token-name>
export POETRY_HTTP_BASIC_GITLAB_PASSWORD=<token>
```

The docker container should be viewed as a proof of concept for integrating with the baal-nf pipeline.
Future plans are
- Move nopeak_utils classes into tfomics library
- Move any requirements for processing nopeak motifs into the baal-nf-env container.