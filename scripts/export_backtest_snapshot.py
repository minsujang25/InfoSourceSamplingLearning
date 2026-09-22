"""Export a compact matched-seed snapshot for Mesa migration backtesting.

The harness is deliberately version-agnostic: point --model-root at either the
legacy Mesa 2.4 checkout or the migrated Mesa 3.5 checkout. The same scenario
definitions and seeds are used in both environments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCENARIOS = [
    {
        "id": "baseline_flat_k1_n0_delta",
        "mode": "baseline",
        "initial": "flat",
        "surveil_ability": 1,
        "num_max_citizen_neighbor": 0,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "random_flat_k1_n2_delta",
        "mode": "random",
        "initial": "flat",
        "surveil_ability": 1,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "random_flat_k4_n5_delta",
        "mode": "random",
        "initial": "flat",
        "surveil_ability": 4,
        "num_max_citizen_neighbor": 5,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "group_flat_k1_n2_delta",
        "mode": "group_id_matching",
        "initial": "flat",
        "surveil_ability": 1,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "group_polarized_k4_n5_delta",
        "mode": "group_id_matching",
        "initial": "polarized",
        "surveil_ability": 4,
        "num_max_citizen_neighbor": 5,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "random_consensus_k4_n2_delta",
        "mode": "random",
        "initial": "consensus",
        "surveil_ability": 4,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "delta_comparison",
    },
    {
        "id": "random_polarized_k1_n2_z",
        "mode": "random",
        "initial": "polarized",
       "surveil_ability": 1,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "z_stat_comparison",
    },
    {
        "id": "group_consensus_k4_n2_z",
        "mode": "group_id_matching",
        "initial": "consensus",
        "surveil_ability": 4,
        "num_max_citizen_neighbor": 2,
        "comparison_rule": "z_stat_comparison",
    },
]

DEFAULT_SEEDS = [101, 202, 303]
N = 42
MAX_STEPS = 40


def load_model_module(model_root: Path):
    model_file = model_root / "model" / "InfoSourceSamplingLearning.py"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    spec = importlib.util.spec_from_file_location("issl_backtest_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initial_beliefs(kind: str, seed: int) -> list[float]:
    # Use a dedicated generator so constructing initial conditions does not
    # consume the global RNG stream used by the model itself.
    rng = np.random.default_rng(seed + 10_000)
    n_citizens = N - 2
    if kind == "flat":
        citizens = rng.uniform(-5.0, 5.0, n_citizens)
    elif kind == "polarized":
        half = n_citizens // 2
        citizens = np.concatenate(
            [rng.normal(-3.0, 1.0, half), rng.normal(3.0, 1.0, n_citizens - half)]
        )
    elif kind == "consensus":
        citizens = rng.normal(0.0, 1.0, n_citizens)
    else:
        raise ValueError(f"Unknown initial condition: {kind}")
    return [0.0, 4.0] + citizens.tolist()


def build_config(module, scenario: dict, seed: int) -> dict:
    mu_theta = initial_beliefs(scenario["initial"], seed)
    return {
        "state_of_the_world": 0.0,
        "num_nodes": N,
        "comparison_rule": scenario["comparison_rule"],
        "epsilon": 0.05,
        "credit": 20,
        "mu_delta": [[0.0] * 8 for _ in range(N)],
        "sd_delta": [[5.0] * 8 for _ in range(N)],
        "mu_theta": mu_theta,
        "sd_theta": [1.0, 1.0] + [5.0] * (N - 2),
        "initial_theta_type": scenario["initial"],
        "seq_meaningful": True,
        "type_of_agent": [module.InfoProvider, module.DisruptiveJammer]
        + [module.Citizen] * (N - 2),
        "max_steps": MAX_STEPS,
        "network_type": "fully_connected",
        "mode": scenario["mode"],
        "network_structure": None,
        "learn_method": "cautious",
        "counterpart_pick_mechanism": "equal",
        "surveil_ability": scenario["surveil_ability"],
        "num_max_citizen_neighbor": scenario["num_max_citizen_neighbor"],
    }


def agents_of(model):
    if hasattr(model, "schedule"):
        return list(model.schedule.agents)
    return list(model.agents)


def model_steps(model) -> int:
    if hasattr(model, "schedule"):
        return int(model.schedule.steps)
    return int(model.steps)


def canonical_clusters(model) -> list[list[int]]:
    agents = agents_of(model)
    jammers = [a for a in agents if a.type_of_agent == "disruptivejammer"]
    if not jammers:
        return []
    clusters = getattr(jammers[0], "citizen_per_cl", {})
    canonical = []
    for citizens in clusters.values():
        canonical.append(sorted(int(c.pos) for c in citizens))
    return sorted(canonical)


def citizen_final_by_pos(model) -> list[dict]:
    citizens = sorted(
        [a for a in agents_of(model) if a.type_of_agent == "citizen"],
        key=lambda a: int(a.pos),
    )
    return [
        {
            "pos": int(a.pos),
            "mu_theta": float(a.mu_theta_beliefs-BÙÝ]HØ]
KÙÝ]WØ[YYËP¢Ð¢f÷"â6F¦Vç0¢Ð  ¦FVb'VåööæRÖöGVÆRÂ66Væ&ó¢F7BÂ6VVC¢çBÓâF7C ¢çç&æFöÒç6VVB6VVB¢6öæfrÒ'VÆEö6öæfrÖöGVÆRÂ66Væ&òÂ6VVB ¢G' ¢ÖöFVÂÒÖöGVÆRäæfõ6×ÆTÖöFVÂÖöFVÅöGG&'WFSÖ6öæfrÂ&æs×6VVB¢W6WBGTW'&÷# ¢ÖöFVÂÒÖöGVÆRäæfõ6×ÆTÖöFVÂÖöFVÅöGG&'WFSÖ6öæfr ¢ÖöFVÂç'VåöÖöFVÂ&çE÷FÖSÔfÇ6R ¢G&¦V7F÷'Ò°¢¶fÆöBf÷"â&÷uÐ¢f÷"&÷râvWFGG"ÖöFVÂÂ&vVçEö×U÷FWFöÆ7B"ÂµÒ¢Ð¢VFvW2Ò6÷'FVB¢µ¶çBRÂçBbÒf÷"RÂbâÖöFVÂææWGv÷&²æVFvW2ÒÀ¢¶WÖÆÖ&FS¢U³ÒÂU³ÒÀ¢¢fæÂÒ6F¦VåöfæÅö'÷÷2ÖöFVÂ¢fæÅö×RÒçæ'&·&÷u²&×U÷FWF%Òf÷"&÷râfæÅÒÂGGSÖfÆöB ¢&WGW&â°¢'66Væ&õöB#¢66Væ&õ²&B%ÒÀ¢'6VVB#¢6VVBÀ¢'7FW2#¢ÖöFVÅ÷7FW2ÖöFVÂÀ¢&æWGv÷&µöVFvW2#¢VFvW2À¢&¦ÖÖW%ö6ÇW7FW'2#¢6æöæ6Åö6ÇW7FW'2ÖöFVÂÀ¢'G&¦V7F÷'#¢G&¦V7F÷'À¢&fæÅö'÷÷2#¢fæÂÀ¢'7VÖÖ'#¢°¢&ÖVåöfæÂ#¢fÆöBçæÖVâfæÅö×RÀ¢'6EöfæÂ#¢fÆöBçç7FBfæÅö×RÀ¢&ÖU÷G'WF#¢fÆöBçæÖVâçæ'2fæÅö×RÀ¢ÒÀ¢Ð  ¦FVbÖâ ¢'6W"Ò&w'6Rä&wVÖVçE'6W"¢'6W"æFEö&wVÖVçB"ÒÖÖöFVÂ×&ö÷B"ÂGSÕFÂ&WV&VCÕG'VR¢'6W"æFEö&wVÖVçB"ÒÖ÷WGWB"ÂGSÕFÂ&WV&VCÕG'VR¢'6W"æFEö&wVÖVçB"Ò×6VVG2"ÂGSÖçBÂæ&w3Ò"²"ÂFVfVÇCÔDTdTÅEõ4TTE2¢&w2Ò'6W"ç'6Uö&w2 ¢ÖöGVÆRÒÆöEöÖöFVÅöÖöGVÆR&w2æÖöFVÅ÷&ö÷Bç&W6öÇfR¢'Vç2ÒµÐ¢f÷"66Væ&òâ44Tä$õ3 ¢f÷"6VVBâ&w2ç6VVG3 ¢'Vç2æVæB'VåööæRÖöGVÆRÂ66Væ&òÂ6VVB ¢ÆöBÒ°¢'66VÖ÷fW'6öâ#¢À¢&â#¢âÀ¢&Ö÷7FW2#¢Ôõ5DU2À¢'6VVG2#¢&w2ç6VVG2À¢'66Væ&÷2#¢44Tä$õ2À¢''Vç2#¢'Vç2À¢Ð¢&w2æ÷WGWBç&VçBæÖ¶F"&VçG3ÕG'VRÂW7Eöö³ÕG'VR¢&w2æ÷WGWBçw&FU÷FWB§6öâæGV×2ÆöBÂæFVçCÓ"ÂVæ6öFæsÒ'WFbÓ"¢&çBb%w&÷FR¶ÆVâ'Vç2ÒÖF6VB&6·FW7B'Vç2Fò¶&w2æ÷WGWGÒ"  ¦bõöæÖUõòÓÒ%õöÖåõò# ¢Öâ