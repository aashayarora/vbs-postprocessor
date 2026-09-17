"""Per-channel analysis definitions shared by run_analysis.py and closure.py.

A channel is defined by its fat-jet tagger columns, the regions.yaml key each
cut is stored under, and how the tagger cuts are combined ("and"/"or").
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Channel:
    name: str
    # (dataframe column, regions.yaml key) per tagger, in optimize_cuts order.
    taggers: tuple
    combine: str  # "and" | "or"
    # Historical per-channel file name for the background profile plot.
    profile_name: str

    @property
    def tagger_cols(self):
        return [col for col, _ in self.taggers]

    @property
    def cut_keys(self):
        return [key for _, key in self.taggers]

    def _join(self, parts):
        op = " & " if self.combine == "and" else " | "
        return op.join(parts)

    def preselection_query(self, threshold=0.1):
        """Pandas query string for the loose tagger preselection."""
        return self._join(f"{col} >= {threshold}" for col in self.tagger_cols)

    def kinematic_query(self, cut_values):
        """Pandas query string for the optimized tagger cuts."""
        return self._join(
            f"{col} >= {val}" for col, val in zip(self.tagger_cols, cut_values)
        )

    def kinematic_mask(self, df, cuts):
        """Boolean numpy mask of the tagger cuts, reading values from a
        regions.yaml-style cuts dict."""
        parts = [df[col].to_numpy() >= float(cuts[key]) for col, key in self.taggers]
        mask = parts[0].copy()
        for p in parts[1:]:
            mask = (mask & p) if self.combine == "and" else (mask | p)
        return mask


CHANNELS = {
    "1fj": Channel(
        name="1fj",
        taggers=(("boosted_h_candidate_score", "h_cut"),
                 ("boosted_v_candidate_score", "v_cut")),
        combine="or",
        profile_name="background_dnn_vs_vbs_median_profile",
    ),
    "2fj": Channel(
        name="2fj",
        taggers=(("boosted_h_candidate_score", "h_cut"),
                 ("boosted_v_candidate_score", "v_cut")),
        combine="and",
        profile_name="background_dnn_vs_vbs_median_profile",
    ),
    "3fj": Channel(
        name="3fj",
        taggers=(("boosted_h_candidate_score", "h_cut"),
                 ("boosted_v1_candidate_score", "v1_cut"),
                 ("boosted_v2_candidate_score", "v2_cut")),
        combine="and",
        profile_name="background_dnn_vs_vbs_mean_profile",
    ),
}
