from collections import defaultdict
from typing import List

import pandas as pd
from gimmemotifs.motif import Motif
from gimmemotifs.fasta import Fasta
from gimmemotifs.scanner import Scanner
from tfomics import ReferenceGenome


class ScoreSet:
    """Utility class for calculating motif scores for a set of ASB sites"""

    def __init__(
        self,
        sites: pd.DataFrame,
        motifs: List[Motif],
        genome: ReferenceGenome,
        genome_name="hg19",
    ):
        self.motifs = motifs
        self.genome = genome
        sequences = sites.apply(self._get_sequences, axis=1)
        self.sites = sites.join(sequences)
        self.ref_matches = None
        self.alt_matches = None
        self.genome_name = genome_name

    def _get_sequences(self, row: pd.Series):
        """Get ref and alt sequences for a given site"""
        sequence = self.genome.get_peak(row.CHROM, row.POS)
        ref_seq = sequence
        alt_seq = (
            sequence[: self.genome.offset]
            + row.ALT
            + sequence[self.genome.offset + 1 :]
        )
        return pd.Series({"ref_seq": ref_seq, "alt_seq": alt_seq})

    def _score_sequences(self, fpr):
        """Scan all reference and alternate sequences and store a list of
        the output scores in self.ref_matches/self.alt_matches"""
        scanner = Scanner()
        scanner.set_motifs(self.motifs)
        scanner.set_genome(self.genome_name)
        scanner.set_threshold(fpr=fpr)
        self.ref_matches = list(scanner.scan(Fasta(fdict=self.sites.ref_seq.to_dict())))
        self.alt_matches = list(scanner.scan(Fasta(fdict=self.sites.alt_seq.to_dict())))

    def get_best_scores(self, fpr=0.05):
        """Get the best motif score for each sequence, at a given false positive rate."""
        self._score_sequences(fpr=fpr)
        scores = defaultdict(list)

        for ref_matches, alt_matches, rsid, ref_seq, alt_seq, ar in zip(
            self.ref_matches,
            self.alt_matches,
            self.sites.ID,
            self.sites.ref_seq,
            self.sites.alt_seq,
            self.sites["Corrected.AR"],
        ):
            for ref_match_l, alt_match_l, motif in zip(
                ref_matches, alt_matches, self.motifs
            ):
                ref_match, alt_match = None, None
                if ref_match_l:
                    ref_match = max(ref_match_l, key=lambda x: x[0])
                if alt_match_l:
                    alt_match = max(alt_match_l, key=lambda x: x[0])

                # FIXME: this can be cleaned up
                if (ref_match is not None) or (alt_match is not None):
                    scores["motif"].append(str(motif))
                    scores["ar"].append(ar)
                    scores["id"].append(rsid)
                    if (alt_match is None) or (
                        (ref_match is not None) and (ref_match[0] > alt_match[0])
                    ):
                        scores["ref_score"].append(ref_match[0])
                        scores["pos"].append(ref_match[1])
                        scores["strand"].append(ref_match[2])
                        scores["alt_score"].append(
                            ScoreSet.re_score(
                                alt_seq, ref_match[1], ref_match[2], motif
                            )
                        )

                    else:
                        scores["alt_score"].append(alt_match[0])
                        scores["pos"].append(alt_match[1])
                        scores["strand"].append(alt_match[2])
                        scores["ref_score"].append(
                            ScoreSet.re_score(
                                ref_seq, alt_match[1], alt_match[2], motif
                            )
                        )
        score_set = pd.DataFrame.from_dict(scores)
        score_set["score_diff"] = score_set.ref_score - score_set.alt_score
        return score_set

    @staticmethod
    def re_score(sequence, pos, strand, motif):
        """Calculate the motif score at a given position in the sequence."""
        length = len(motif.pfm)

        if strand < 0:
            motif = ~motif

        kmer = sequence[pos : pos + length]
        return motif.score_kmer(kmer)
