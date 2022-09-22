"""Helper class for parsing nopeak result files into gimmemotif Motifs"""

import ast
from numbers import Number
from pathlib import Path

from gimmemotifs.motif import Motif
from ttp import ttp
import numpy as np

NOPEAK_TEMPLATE = """
<group name="motif">
Significant motifs:
Motif:	{{ motif }}
K-mers count:	{{ kmer_count | to_int }}

Logo as Python array for the plot_pwm.py script:
{{ pfm }}
</group>
"""


class NoPeakMotif(Motif):
    """Wrapper class for NoPeak motif output files"""

    def __init__(self, motif, kmer_count, pfm, motif_id=None):
        self.kmer_count = kmer_count
        self.motif_str = motif
        Motif.__init__(self, pfm=pfm)
        if motif_id is not None:
            self.id = motif_id

    @staticmethod
    def from_file(filename, strip_edges=True):
        """Parse NoPeak output file and produce a list of NoPeakMotifs"""
        parser = ttp(data=filename, template=NOPEAK_TEMPLATE, log_level="ERROR")
        parser.parse()
        results = parser.result()[0][0]["motif"]

        # If there's only one element in the group, ttp returns a dict rather than a list.
        if not isinstance(results, list):
            results = [results]

        for result in results:
            result["pfm"] = NoPeakMotif.parse_pfm(
                result["pfm"], strip_edges=strip_edges
            )

        return [
            NoPeakMotif(**motif, motif_id=f"{Path(filename).stem}_{i}")
            for i, motif in enumerate(results)
        ]

    def rc(self):
        return NoPeakMotif(None, self.kmer_count, Motif.rc(self).pfm, None)

    @staticmethod
    def parse_pfm(pfm_str, strip_edges):
        """Parse the PFM embedded in the NoPeak output file"""
        pfm = ast.literal_eval(pfm_str)

        if strip_edges:
            pfm = NoPeakMotif._strip_edges(pfm)
        else:
            pfm = [[row[4] / 4.0] * 4 if set(row[:4]) == {0} else row for row in pfm]

        pfm = [row[:4] for row in pfm]

        return pfm

    @staticmethod
    def _strip_left_edge(pfm):
        nonzero = False
        stripped_pfm = []
        for row in pfm:
            if sum(row[:4]):
                nonzero = True
            if nonzero:
                stripped_pfm.append(row)

        return stripped_pfm

    @staticmethod
    def _strip_edges(pfm):
        stripped_pfm = NoPeakMotif._strip_left_edge(pfm)
        stripped_pfm = NoPeakMotif._strip_left_edge(stripped_pfm[::-1])[::-1]
        return stripped_pfm

    def __mul__(self, factor):
        if not isinstance(factor, Number):
            return NotImplemented

        new_pfm = factor * np.array(self.pfm)

        return NoPeakMotif(self.motif_str, self.kmer_count, new_pfm.tolist(), self.id)

    def __add__(self, other):
        if not isinstance(other, NoPeakMotif):
            return NotImplemented

        pfm_1 = self.pfm
        pfm_2 = other.pfm

        size_diff = len(pfm_1) - len(pfm_2)

        if size_diff > 0:
            pfm_2 = NoPeakMotif._pad_right(pfm_2, size_diff)
        elif size_diff < 0:
            pfm_1 = NoPeakMotif._pad_right(pfm_1, abs(size_diff))

        pfm_1 = np.array(pfm_1)
        pfm_2 = np.array(pfm_2)

        if pfm_1.shape != pfm_2.shape:
            raise ValueError(f"Malformed PFMs: {pfm_1.shape} != {pfm_2.shape}")

        new_pfm = pfm_1 + pfm_2
        return NoPeakMotif(None, self.kmer_count + other.kmer_count, new_pfm.tolist())

    def __rshift__(self, num):
        if not isinstance(num, int):
            return NotImplemented

        new_str = (num * "n" + self.motif_str) if self.motif_str else None

        return NoPeakMotif(
            new_str,
            self.kmer_count,
            NoPeakMotif._pad_left(self.pfm, num),
            self.id,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    @staticmethod
    def _pad_right(pfm, num):
        """pad a PFM with num sets of zeroes on the right"""
        return pfm + num * [[0, 0, 0, 0]]

    @staticmethod
    def _pad_left(pfm, num):
        """pad a PFM with num sets of zeroes on the left"""
        return num * [[0, 0, 0, 0]] + pfm

    @staticmethod
    def add_with_offset(first, second, offset):
        if offset > 0:
            second = second >> offset
        elif offset < 0:
            first = first >> offset

        return first + second
