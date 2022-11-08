# Motif calling prototypes

This repository contains two notebooks and some code that was developed to investigate the use of NoPeak for motif calling in the `baal-nf` pipeline.
Motif calling has been done with [NoPeak](https://github.com/menzel/nopeak).

The pipeline was initially run on CTCF only, the output of that run can be found on eddie in `/exports/igmm/eddie/ponting-lab/oyvind/work/baal_chip/prototypes/motifs`.
Following that, another pipeline was run for FOXA1, NR3C1 and ESR1 (data in `/exports/igmm/eddie/ponting-lab/oyvind/work/baal_chip/prototypes/motifs_2`). 
Hence this repository has two data folders, one for [CTCF](ctcf) and one for the [other transcription factors](other_tfs.dvc).

This repository uses poetry for dependency handling and git-lfs for handling large data files. 
To set up the repository, do the following

```bash
conda create -n nopeak-utils python=3.8
conda activate nopeak-utils
pip install poetry
poetry install
dvc pull
```

Note that dvc has been configured to work from an eddie wild west node only.
If you wish to run the notebooks on your laptop, you'll have to either mount the datastore in /exports/igmm/datastore or use an SSH tunnel and create a local remote.

## [`motif_and_asb.ipynb`](/motif_and_asb.ipynb)

This contains a comparison of ASB values as called by BaalChIP and the difference in motif score as reported by gimmemotifs.
These two values are plotted as a scatterplot and the spearman correlation is reported.

Roughly, the notebook does the following

- Variant calls from the baal-nf pipeline
- Import motifs that have been downloaded from Jaspar for each of the transcription factors
- Import motif calls made for each sample file that was processed by the pipeline, and select the five motifs that were called based on the largest number of reads for eacht ranscription factor.

The SNPs are then filtered as follows

- SNPs that do not lie within a peak are filtered out
- SNPs are filtered out if they do not match at least one motif for the relevant transcription factor at a false positive rate of 0.05.
- If there is no change in motif score between the two alleles, the SNP is filtered out

## [`prototype.ipynb`](/prototype.ipynb)

This is an early prototype notebook, looking exclusively at the CTCF data and attempting to see if there are consensus motifs that can be extracted from the many different motif calls.
Two things might be interesting in there, namely:

- gimmemotifs built-in clustering algorithm
- gimmemotifs ability to look up the best-matching motif from a database.

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