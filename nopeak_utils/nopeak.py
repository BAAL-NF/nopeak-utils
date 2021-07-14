import ast

from gimmemotifs.motif import Motif
from ttp import ttp

nopeak_template = """
<group name="motif">
Significant motifs:
Motif:	{{ motif }}
K-mers count:	{{ kmer_count }}

Logo as Python array for the plot_pwm.py script:
{{ pwm }}
</group>
"""


class NoPeakMotif(Motif):
    def __init__(self, motif="", kmer_count=0, pwm=[]):
        self.kmer_count = kmer_count
        self.motif_str = motif
        Motif.__init__(self, pwm)
        self.id = self.motif_str

    @staticmethod
    def from_file(filename):
        parser = ttp(data=filename, template=nopeak_template, log_level="ERROR")
        parser.parse()
        results = parser.result()[0][0]["motif"]

        # If there's only one element in the group, ttp returns a dict rather than a list.
        if not isinstance(results, list):
            results = [results]

        for result in results:
            result["pwm"] = NoPeakMotif.parse_pwm(result["pwm"])

        return [NoPeakMotif(**motif) for motif in results]

    @staticmethod
    def parse_pwm(pwm_str):
        pwm = ast.literal_eval(pwm_str)
        pwm = [row[:4] for row in pwm]
        pwm = NoPeakMotif._strip_n(pwm)
        return pwm

    @staticmethod
    def _strip_left_n(pwm):
        nonzero = False
        new_pwm = []
        for row in pwm:
            if sum(row):
                nonzero = True
            if nonzero:
                new_pwm.append(row)

        return new_pwm

    @staticmethod
    def _strip_n(pwm):
        result = NoPeakMotif._strip_left_n(pwm)
        result = NoPeakMotif._strip_left_n(result[::-1])[::-1]
        return result
