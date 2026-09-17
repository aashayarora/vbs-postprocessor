"""Config loading and interpretation for the ABCD pipeline.

The YAML config is parsed once into a ``RunConfig`` that resolves every
default in a single place, so the rest of the pipeline never calls
``cfg.get(...)`` with scattered fallback values.
"""

import glob
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

import bdt as bdt_lib
from systematics import SYST_WEIGHTS


def load_yaml(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(base, paths):
    """Expand config path globs relative to a base directory."""
    if isinstance(paths, str):
        paths = [paths]
    resolved = []
    for p in paths:
        full = os.path.expanduser(os.path.join(base, p) if base else p)
        matches = sorted(glob.glob(full))
        if matches:
            resolved.extend(matches)
        else:
            resolved.append(full)
    return resolved


def parse_training_features(training_features_cfg):
    """Parse the ``[name, {name: transform}, ...]`` feature list from the config."""
    training_features = []
    feature_transforms = {}

    for feat in training_features_cfg:
        if isinstance(feat, dict):
            for name, tf in feat.items():
                training_features.append(name)
                feature_transforms[name] = tf
        else:
            training_features.append(feat)
            feature_transforms[feat] = "none"

    return training_features, feature_transforms


def _extract_expression_variables(expression):
    tokens = re.findall(r"(?<!\.)\b[A-Za-z_][A-Za-z0-9_]*\b", expression)
    excluded = {"np", "abs", "and", "or", "not", "True", "False"}
    return [tok for tok in tokens if tok not in excluded]


def normalize_derived_vars_cfg(derived_vars_cfg):
    """Accept ``derived_vars`` as a mapping or a list of single-item mappings."""
    if derived_vars_cfg is None:
        return {}
    if isinstance(derived_vars_cfg, dict):
        return dict(derived_vars_cfg)
    if isinstance(derived_vars_cfg, list):
        normalized = {}
        for item in derived_vars_cfg:
            if not isinstance(item, dict):
                raise ValueError("Each item in 'derived_vars' list must be a mapping of 'new_var: expression'.")
            for key, value in item.items():
                normalized[key] = value
        return normalized
    raise ValueError("'derived_vars' must be either a mapping or a list of single-item mappings.")


def collect_derived_input_vars(derived_vars_cfg):
    """Branches that must be loaded from the trees to evaluate the derived vars."""
    derived_names = set(derived_vars_cfg.keys())
    needed = []
    for expr in derived_vars_cfg.values():
        if not isinstance(expr, str):
            continue
        for var in _extract_expression_variables(expr):
            if var not in derived_names:
                needed.append(var)
    return list(dict.fromkeys(needed))


@dataclass
class RunConfig:
    """All settings for one training/inference run, defaults resolved."""

    raw: dict
    flavor: str

    training_features: list = field(init=False)
    feature_transforms: dict = field(init=False)
    constraint_var: str = field(init=False)
    derived_vars: dict = field(init=False)
    bdt_features: list = field(init=False)
    extra_vars: list = field(init=False)
    load_features: list = field(init=False)

    def __post_init__(self):
        cfg = self.raw
        self.training_features, self.feature_transforms = parse_training_features(cfg["training_features"])
        self.constraint_var = cfg["constraint_var"]
        self.derived_vars = normalize_derived_vars_cfg(cfg.get("derived_vars", {}))
        self.bdt_features = bdt_lib.parse_bdt_features(cfg.get("bdt_features", []))

        if self.use_bdt and cfg.get("use_bdt_as_constraint", True):
            self.constraint_var = bdt_lib.BDT_SCORE_NAME
            logging.info(
                "BDT enabled: '%s' will be used as the DNN decorrelation constraint",
                self.constraint_var,
            )

        # Double disco has no external constraint: the constraint var becomes a
        # regular input and the two heads are decorrelated against each other.
        if self.flavor == "double" and self.constraint_var not in self.training_features:
            self.training_features.append(self.constraint_var)
            self.feature_transforms[self.constraint_var] = cfg.get("constraint_as_feature_transform", "none")
            logging.info(
                "Double flavor: added constraint_var '%s' to training features as regular input",
                self.constraint_var,
            )

        if cfg.get("auto_include_derived_vars", False):
            auto_tf = cfg.get("auto_include_derived_vars_transform", "none")
            added_count = 0
            for derived_name in self.derived_vars.keys():
                if derived_name in self.training_features:
                    continue
                self.training_features.append(derived_name)
                self.feature_transforms[derived_name] = auto_tf
                added_count += 1
            logging.info(
                "Auto-include derived vars enabled: added %d derived features with transform='%s'",
                added_count,
                auto_tf,
            )

        # Branches to read from the trees: everything the model needs that is
        # not computed in memory (derived vars, BDT score).
        computed_vars = set(self.derived_vars.keys())
        if self.use_bdt:
            computed_vars.add(bdt_lib.BDT_SCORE_NAME)
        load_training = [f for f in self.training_features if f not in computed_vars]
        load_constraint = [] if self.constraint_var in computed_vars else [self.constraint_var]
        self.load_features = list(dict.fromkeys(
            load_training
            + load_constraint
            + collect_derived_input_vars(self.derived_vars)
            + (self.bdt_features if self.use_bdt else [])
        ))

        self.extra_vars = list(cfg.get("extra_vars", []))
        if "weight" not in self.extra_vars:
            self.extra_vars.append("weight")

        # Systematic weight variations, read for signal only: the datacards apply
        # them to the signal alone (the background rate is a rateParam measured
        # from the data control regions). Background rows get NaN for these when
        # the two are concatenated. Requested per-kind via extra_vars_for.
        self.syst_vars = []
        if cfg.get("store_systematics", True):
            self.syst_vars = [b for b in SYST_WEIGHTS if b not in self.extra_vars]

    def extra_vars_for(self, kind):
        """Extra branches to read for kind in {'sig', 'bkg', 'data'}."""
        if kind == "sig":
            return self.extra_vars + self.syst_vars
        return self.extra_vars

    @property
    def use_bdt(self):
        return len(self.bdt_features) > 0

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("output", "simple_abcdisco_output"))

    @property
    def batch_size(self):
        return self.raw.get("batch_size", 4096)

    @property
    def io_workers(self):
        return int(self.raw.get("io_workers", min(8, os.cpu_count() or 1)))

    @property
    def preselection(self):
        return self.raw.get("preselection")

    @property
    def plot_style(self):
        """CMS header settings for every figure this run draws (see style.py).

        Optional top-level config keys; anything left out falls back to the
        style module's default (or its ABCD_* environment override).
        """
        return {
            "label": self.raw.get("cms_label"),
            "lumi": self.raw.get("lumi"),
            "com": self.raw.get("com"),
            "extra": self.raw.get("channel_label"),
        }

    def sample_paths(self, kind, for_inference=False):
        """Resolved input files for kind in {'sig', 'bkg', 'data'}.

        In inference mode an optional '{kind}_infer_path' overrides the training
        '{kind}_path' (with its own optional '{kind}_infer_base_path', else the
        training base path). If no inference path is defined, the training path
        is used, so training and inference read the same files by default.
        """
        if for_inference and f"{kind}_infer_path" in self.raw:
            base = self.raw.get(
                f"{kind}_infer_base_path",
                self.raw.get(f"{kind}_base_path", self.raw.get("input_base_path", "")),
            )
            return resolve_paths(base, self.raw[f"{kind}_infer_path"])
        base = self.raw.get(f"{kind}_base_path", self.raw.get("input_base_path", ""))
        return resolve_paths(base, self.raw[f"{kind}_path"])

    def model_kwargs(self, input_size, for_inference=False):
        """Constructor arguments for ABCDLightningModule from the config.

        Training uses the full set; inference builds the same architecture but
        does not need the optimizer/scheduler settings.
        """
        cfg = self.raw
        kwargs = dict(
            input_size=input_size,
            hidden_layers=cfg.get("architecture", [64, 32, 16]),
            learning_rate=cfg.get("learning_rate", 1e-3),
            bce_weight=cfg.get("bce_weight", 1.0),
            disco_lambda=cfg.get("disco_lambda", 0.0),
            flavor=self.flavor,
            use_batchnorm=cfg.get("use_batchnorm", True),
            dropout=cfg.get("dropout", 0.0),
        )
        if not for_inference:
            kwargs.update(
                weight_decay=cfg.get("weight_decay", 1e-2),
                label_smoothing=cfg.get("label_smoothing", 0.0),
                use_lr_scheduler=cfg.get("use_lr_scheduler", True),
                lr_scheduler_patience=cfg.get("lr_scheduler_patience", 40),
                lr_scheduler_factor=cfg.get("lr_scheduler_factor", 0.5),
                lr_scheduler_min_lr=cfg.get("lr_scheduler_min_lr", 1e-6),
            )
        return kwargs
