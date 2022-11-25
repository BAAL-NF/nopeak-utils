from typing import List, Tuple

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
        fasta: str,
    ):
        self.motifs = motifs
        self.genome = genome
        sequences = sites.apply(self._get_sequences, axis=1)
        self.sites = sites.join(sequences)
        self.fasta = fasta

    def _get_sequences(self, row: pd.Series) -> pd.Series:
        """Get ref and alt sequences for a given site"""
        sequence = self.genome.get_peak(row.CHROM, row.POS)
        ref_seq = sequence
        alt_seq = (
            sequence[: self.genome.offset]
            + row.ALT
            + sequence[self.genome.offset + 1 :]
        )
        return pd.Series({"ref_seq": ref_seq, "alt_seq": alt_seq})

    def _score_sequences(self, fpr: float):
        """Scan all reference and alternate sequences and store a list of
        the output scores in self.ref_matches/self.alt_matches"""
        scanner = Scanner()
        scanner.set_motifs(self.motifs)
        scanner.set_background(fname=self.fasta)
        scanner.set_threshold(fpr=fpr)

        ref_matches = list(scanner.scan(Fasta(fdict=self.sites.ref_seq.to_dict())))
        alt_matches = list(scanner.scan(Fasta(fdict=self.sites.alt_seq.to_dict())))

        return (ref_matches, alt_matches)

    def get_best_scores(self, fpr: float = 0.05) -> pd.DataFrame:
        """
        Get the best motif score for each sequence, at a given false positive rate.
        """
        ref_matches, alt_matches = self._score_sequences(fpr=fpr)
        scores = []

        for ref_match, alt_match, (_, row) in zip(
            ref_matches, alt_matches, self.sites.iterrows()
        ):
            for ref_match_l, alt_match_l, motif in zip(
                ref_match, alt_match, self.motifs
            ):
                # I'm avoiding any conditional logic on whether the max is in the
                # ref or alt strand, this means we re-score both strands unnecesarily,
                # but it makes the flow much simpler
                combined_scores = ref_match_l + alt_match_l
                if not combined_scores:
                    continue

                (_, pos, strand) = max(combined_scores, key=lambda x: x[0])

                score = row.copy()
                score["motif"] = str(motif)
                score["motif_pos"] = pos
                score["motif_strand"] = strand
                score["ref_score"] = ScoreSet.re_score(
                    score.ref_seq, pos, strand, motif
                )
                score["alt_score"] = ScoreSet.re_score(
                    score.alt_seq, pos, strand, motif
                )
                score["score_diff"] = score.ref_score - score.alt_score

                scores.append(score)

        score_set = pd.DataFrame(scores)
        return score_set

    @staticmethod
    def re_score(
        sequence, pos: int, strand: int, motif: Motif
    ) -> Tuple[float, int, int]:
        """Calculate the motif score at a given position in the sequence."""
        length = len(motif.pfm)

        if strand < 0:
            motif = ~motif

        kmer = sequence[pos : pos + length]
        return motif.score_kmer(kmer)
