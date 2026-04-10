"""
Four-Agent Event Extraction Pipeline: Detector + Extractor + Revisor + Verifier

This pipeline implements a multi-agent iterative training framework for
document-level event argument extraction. Four agents collaborate:

    Detector  (BART):       Detects event triggers and classifies event types
    Extractor (BART):       Extracts event arguments via template filling
    Revisor   (LLM+LoRA):   Reviews and revises extraction results
    Verifier  (LLM+LoRA):   Validates results against knowledge & ontology

Training Loop:
    Phase 1 (Initialization):
        - Train all agents on labeled training data

    Phase 2 (Iterative Refinement):
        For each iteration:
        1. Generator produces synthetic event data
        2. All agents estimate quality scores:
           - Detector:  detection NLL
           - Extractor: extraction NLL
           - Revisor:   revision penalty + consistency check
           - Verifier:  ontology + cross-argument + knowledge penalty
        3. Revisor reviews and revises extraction results
        4. Verifier validates results against knowledge constraints
        5. Combine scores into unified reward → filter high-quality data
        6. Retrain all agents on filtered data

Reward Computation:
    reward = normalize(
        w1 * extractor_nll +
        w2 * detector_nll +
        w3 * revision_penalty +
        w4 * verification_score +
        generator_none_penalty
    )

Usage:
    python pipeline_three_agent.py \\
        --path-train dataset/rams2rams/train.jsonl \\
        --path-dev dataset/rams2rams/dev.jsonl \\
        --path-test dataset/rams2rams/test.jsonl \\
        --save-dir experiments/three_agent/output \\
        --num-iter 5
"""

import json
import random
import math
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
import re
from tqdm import tqdm
from transformers import set_seed as hf_set_seed
import pandas as pd

from Agent.detector import Detector
from Agent.extractor import Extractor
from Agent.revisor import Revisor
from Agent.verifier import Verifier
from Agent.generator import Generator

from Dataset import Dataset


# ---------------------------------------------------------------------------
# Reward & Data Filtering
# ---------------------------------------------------------------------------

def compute_four_agent_reward(data: list) -> dict:
    """
    Compute unified reward scores from all four agents' quality signals.

    Combines:
        - extractor_nll:      Extractor's confidence in the extraction
        - detector_nll:       Detector's confidence in trigger detection
        - revision_penalty:   Revisor's consistency/quality penalty
        - verification_score: Verifier's knowledge/ontology penalty
        - penalty:            Generator's none-ratio penalty

    Each signal is z-score normalized, then combined with equal weights.
    The final reward is min-max scaled to [0, 1].

    Args:
        data: List of dicts, each with the above fields.

    Returns:
        Dict mapping event_type -> sorted list of instances with 'reward' field.
    """
    groups = defaultdict(list)
    for item in data:
        event_type = item['Event type']
        groups[event_type].append(item)

    # Collect all scores for normalization
    ext_list = []
    det_list = []
    rev_list = []
    ver_list = []
    gen_list = []

    for k in groups:
        for tri in groups[k]:
            ext_list.append(tri.get('extractor_nll', 0))
            det_list.append(tri.get('detector_nll', 0))
            rev_list.append(tri.get('revision_penalty', 0))
            ver_list.append(tri.get('verification_score', 0))
            gen_list.append(tri.get('cond_generator_nll', 0))

    ext_mean, ext_std = np.mean(ext_list), np.std(ext_list)
    det_mean, det_std = np.mean(det_list), np.std(det_list)
    rev_mean, rev_std = np.mean(rev_list), np.std(rev_list)
    ver_mean, ver_std = np.mean(ver_list), np.std(ver_list)
    gen_mean, gen_std = np.mean(gen_list), np.std(gen_list)

    def z_score(x, mean, std):
        return ((x - mean) / std) if std != 0 else (x - mean)

    def nll_func(x):
        """Combined quality score (lower = better)."""
        score = (
            z_score(x.get('extractor_nll', 0), ext_mean, ext_std) +
            z_score(x.get('detector_nll', 0), det_mean, det_std) +
            z_score(x.get('revision_penalty', 0), rev_mean, rev_std) +
            z_score(x.get('verification_score', 0), ver_mean, ver_std) +
            z_score(x.get('cond_generator_nll', 0), gen_mean, gen_std) +
            x.get('penalty', 0)
        )
        return score

    # Compute reward for each instance
    max_ll, min_ll = None, None
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: nll_func(x))
        ll_list = [-nll_func(x) for x in groups[k]]
        max_ll = max(max_ll, max(ll_list)) if max_ll is not None else max(ll_list)
        min_ll = min(min_ll, min(ll_list)) if min_ll is not None else min(ll_list)

    for k in groups:
        def scaler_func(x):
            return (x - min_ll) / (max_ll - min_ll) if max_ll != min_ll else 1.0

        for tri in groups[k]:
            tri['reward'] = scaler_func(-nll_func(tri))

    return groups


def filter_data_four_agent(path_pseudo: str,
                           path_train: str,
                           path_out: str,
                           total_pseudo_per_label: int,
                           pseudo_ratio: float,
                           task_name: str,
                           unseen_meta_path: str,
                           meta_path: str,
                           sampling_cfg: dict):
    """
    Filter and merge synthetic data with training data based on
    four-agent reward scores.

    The filtering strategy:
        1. Rank all synthetic data by combined four-agent reward
        2. Select top-k instances (based on pseudo_ratio schedule)
        3. Balance non-empty and empty argument instances
        4. Ensure minimum representation per event type
        5. Merge with real training data

    Args:
        path_pseudo:            Path to scored synthetic data
        path_train:             Path to original training data
        path_out:               Path to write filtered output
        total_pseudo_per_label: Target number of pseudo instances per event type
        pseudo_ratio:           Current ratio (increases over iterations)
        task_name:              Dataset name
        unseen_meta_path:       Path to unseen event types meta
        meta_path:              Path to seen event types meta
        sampling_cfg:           Sampling configuration dict
    """
    print(dict(filter_data_four_agent=locals()))
    if Path(path_out).exists():
        return

    def count_non_empty_args(tri):
        return sum(1 for v in tri['arguments'].values() if v and v.lower() != "none")

    train_data = Dataset().load(path_train)
    pseudo_data = Dataset().load(path_pseudo)
    train_data = trans_format(task_name=task_name, meta_path=meta_path, data=train_data)

    pseudo_labels = get_meta(task_name, unseen_meta_path)
    num_pseudo = int(total_pseudo_per_label * pseudo_ratio * len(pseudo_labels))
    num_pseudo_per_label = int(num_pseudo / len(pseudo_labels))

    # Score and sort using four-agent reward
    pseudo_data = compute_four_agent_reward(pseudo_data)
    train_data = compute_four_agent_reward(train_data)

    nb_rel_train_data = [x for k, v in train_data.items() for x in v]
    nb_rel_pseudo_data = [x for k, v in pseudo_data.items() for x in v]

    origin_pseudo_data = sorted(nb_rel_pseudo_data, key=lambda x: x['reward'], reverse=True)

    pseudo_empty_threshold = sampling_cfg['pseudo_empty_threshold']
    pseudo_empty_ratio = sampling_cfg['pseudo_empty_ratio']

    non_empty_data = [x for x in origin_pseudo_data
                      if count_non_empty_args(x) >= pseudo_empty_threshold]
    empty_data = [x for x in origin_pseudo_data
                  if count_non_empty_args(x) < pseudo_empty_threshold]

    nb_rel_pseudo_data = non_empty_data[:num_pseudo]
    max_empty = min(len(empty_data), int(pseudo_empty_ratio * len(nb_rel_pseudo_data)))
    nb_rel_pseudo_data += empty_data[:max_empty]

    # Ensure minimum representation per event type
    for rel in pseudo_labels:
        rel_pseudo_data = [x for x in nb_rel_pseudo_data if x['Event type'] == rel]
        num = len(rel_pseudo_data)
        if num < num_pseudo_per_label / 5:
            rel_data = [x for x in origin_pseudo_data
                        if x['Event type'] == rel and x not in rel_pseudo_data]
            nb_rel_pseudo_data.extend(rel_data[:num_pseudo_per_label // 5 - num])

    pseudo_reward = [x['reward'] for x in nb_rel_pseudo_data]
    mean_pseudo_reward = np.mean(pseudo_reward) if pseudo_reward else 0.0

    all_data = nb_rel_train_data + nb_rel_pseudo_data

    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    with open(path_out, "w") as f:
        for d in all_data:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')

    print(f'num of filtered data: {len(all_data)}, mean pseudo reward: {mean_pseudo_reward:.4f}')


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_meta(task_name: str, meta_path: str) -> dict:
    """Load event type -> roles mapping from meta file."""
    meta = json.load(open(meta_path, 'r', encoding='utf8'))
    type2roles = {}
    for rams_m in meta:
        rams_event_type, role_types = rams_m
        role_types = [
            r[11:].lower() if 'evt' in r else r.lower()
            for r in role_types
        ]
        type2roles[rams_event_type] = role_types
    return type2roles


def trans_format(data: list, meta_path: str, task_name: str) -> list:
    """Convert raw RAMS-format data into unified format with all fields."""
    type2roles = get_meta(task_name, meta_path)

    all_data = []
    for d in tqdm(data, desc="Converting format"):
        sentences = d['sentences']
        all_tokens = []
        for sentence in sentences:
            all_tokens.extend(sentence)
        doc_text = ' '.join(all_tokens)

        event_type = d['evt_triggers'][0][-1][0][0]
        trigger_s, trigger_e = d['evt_triggers'][0][0], d['evt_triggers'][0][1]
        trigger = ' '.join(all_tokens[trigger_s:trigger_e + 1])

        gold_evt_links = d['gold_evt_links']
        all_roles = type2roles[event_type]

        role2span = {}
        for role in all_roles:
            role2span[role.lower()] = None

        for gold_evt_link in gold_evt_links:
            current_role = gold_evt_link[-1].lower()
            if 'evt' in current_role:
                current_role = current_role[11:]
            s, e = gold_evt_link[1]
            span = ' '.join(all_tokens[s:e + 1])
            role2span[current_role] = span

        new_d = {
            'Context': doc_text,
            'Trigger': trigger,
            'Event type': event_type,
            'arguments': role2span,
            'extractor_nll': 0,
            'detector_nll': 0,
            'revision_penalty': 0,
            'reward': 1.0,
            'cond_generator_nll': 0.0,
            'generator_nll': 0.0
        }
        all_data.append(new_d)
    return all_data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(path_model: str, path_test: str, mode: str, is_eval: bool,
             ontology_dict_path: str, meta_path: str, task_name: str,
             limit: int = 0):
    """Run evaluation using the Extractor model."""
    print(dict(run_eval=locals()))
    is_eval_str = f'is_eval_{is_eval}'
    if limit == 0:
        path_results = str(Path(path_model) / f"results_{mode}_{is_eval_str}.json")
    else:
        path_results = str(Path(path_model) / f"results_{mode}_{is_eval_str}_limit{limit}.json")

    if Path(path_results).exists():
        return

    data = Dataset().trans_extractor(
        data_path=path_test, meta_path=meta_path,
        ontology_dict_path=ontology_dict_path, task_name=task_name, task='eval'
    )
    model = Extractor(
        load_dir=str(Path(path_model) / "model"),
        save_dir=path_model
    )

    if limit > 0:
        random.seed(0)
        random.shuffle(data)

    if limit == 0:
        path_in = str(Path(path_model) / f"pred_in_{mode}_{is_eval_str}.jsonl")
        path_out = str(Path(path_model) / f"pred_out_{mode}_{is_eval_str}.jsonl")
    else:
        path_in = str(Path(path_model) / f"pred_in_{mode}_{is_eval_str}_limit{limit}.jsonl")
        path_out = str(Path(path_model) / f"pred_out_{mode}_{is_eval_str}_limit{limit}.jsonl")

    Path(path_in).parent.mkdir(exist_ok=True, parents=True)
    with open(path_in, "w") as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')

    model.predict(path_in, path_out)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main_four_agent(
    path_train: str,
    path_dev: str,
    path_test: str,
    task_name: str,
    meta_path: str,
    unseen_meta: str,
    ontology_dict_path: str,
    ontology_test_dict_path: str,
    generator_unseen_label_path: str,
    save_dir: str,
    num_iter: int,
    limit: int = 5000,
    g_encoder_name: str = 'generate',
    num_gen_per_label: int = 250,
    diverse: bool = False,
    gen_PLM_path: str = 'GPT2',
    ext_PLM_path: str = 'Bart-large',
    rev_PLM_path: str = 'Meta-Llama-3-8B',
    ver_PLM_path: str = 'Meta-Llama-3-8B',
    sampling_cfg: dict = None,
):
    """
    Main four-agent training pipeline.

    Architecture:
        Generator (LLM+LoRA):  Generates synthetic event data
        Detector  (BART):      Detects triggers and event types
        Extractor (BART):      Extracts arguments via template filling
        Revisor   (LLM+LoRA):  Reviews and revises extraction results
        Verifier  (LLM+LoRA):  Validates results against knowledge & ontology

    The pipeline runs in two phases:
        Phase 1: Initial training of all agents on labeled data
        Phase 2: Iterative co-evolution loop
    """
    if sampling_cfg is None:
        sampling_cfg = {}

    print("=" * 70)
    print("Four-Agent Event Extraction Pipeline")
    print("  Agents: Generator + Detector + Extractor + Revisor + Verifier")
    print("=" * 70)
    print(dict(main=locals()))

    task_train_data, task_test_data = task_name.split('2')

    # ------------------------------------------------------------------
    # Phase 1: Initialize all four agents
    # ------------------------------------------------------------------
    print("\n[Phase 1] Initializing agents on labeled data...")

    # Generator: generates synthetic event data
    generator = Generator(
        load_dir=gen_PLM_path,
        save_dir=str(Path(save_dir) / "generator/iter0"),
        num_gen_per_label=num_gen_per_label + 100 if not diverse else num_gen_per_label * 2,
    )

    # Detector: detects triggers and event types
    detector = Detector(
        load_dir=ext_PLM_path,
        save_dir=str(Path(save_dir) / "detector/iter0"),
    )

    # Extractor: extracts arguments via template filling
    extractor = Extractor(
        load_dir=ext_PLM_path,
        save_dir=str(Path(save_dir) / "extractor/iter0"),
    )

    # Revisor: reviews and revises extraction results
    revisor = Revisor(
        load_dir=rev_PLM_path,
        save_dir=str(Path(save_dir) / "revisor/iter0"),
    )

    # Verifier: validates results against knowledge constraints
    verifier = Verifier(
        load_dir=ver_PLM_path,
        save_dir=str(Path(save_dir) / "verifier/iter0"),
    )

    # Train all agents on initial data
    print("  Training Generator...")
    generator.fit(path_train, path_dev, task_train_data, meta_path, iter=0)

    print("  Training Detector...")
    detector.fit(path_train, path_dev, task_train_data, meta_path,
                 ontology_dict_path, iter=0)

    print("  Training Extractor...")
    extractor.fit(path_train, path_dev, task_train_data, meta_path,
                  ontology_dict_path, iter=0)

    print("  Training Revisor...")
    revisor.fit(path_train, path_dev, task_train_data, meta_path, iter=0)

    print("  Training Verifier...")
    verifier.fit(path_train, path_dev, task_train_data, meta_path, iter=0)

    # Evaluate initial Extractor
    print("  Evaluating initial Extractor...")
    run_eval(
        path_model=str(Path(save_dir) / "extractor/iter0"),
        path_test=path_test, mode='single', is_eval=True,
        limit=limit, meta_path=unseen_meta,
        ontology_dict_path=ontology_test_dict_path,
        task_name=task_test_data
    )

    # ------------------------------------------------------------------
    # Phase 2: Iterative co-evolution
    # ------------------------------------------------------------------
    labels_test = json.load(open(generator_unseen_label_path, 'r'))

    for i in range(num_iter):
        print(f"\n{'=' * 70}")
        print(f"[Phase 2] Iteration {i + 1}/{num_iter}")
        print(f"{'=' * 70}")

        path_synthetic = str(Path(save_dir) / "synthetic" / f"{i}.jsonl")
        path_revised = str(Path(save_dir) / "synthetic" / f"{i}_revised.jsonl")

        # Step 1: Generator generates synthetic event data
        print(f"  Step 1: Generator producing synthetic data...")
        generator.generate(labels_test, path_out=path_synthetic)

        # Step 2: Generator self-estimates quality (none_penalty)
        print(f"  Step 2: Generator estimating quality...")
        generator.estimate(path_synthetic, path_synthetic,
                           path_train, task_name=task_name, meta_path=meta_path)

        # Step 3: Extractor estimates extraction confidence (NLL)
        print(f"  Step 3: Extractor estimating extraction NLL...")
        extractor.estimate(path_synthetic, path_synthetic, ontology_test_dict_path)

        # Step 4: Detector estimates detection confidence (NLL)
        print(f"  Step 4: Detector estimating detection NLL...")
        detector.estimate(path_synthetic, path_synthetic)

        # Step 5: Revisor reviews and revises results
        print(f"  Step 5: Revisor reviewing and revising...")
        revisor.revise(path_synthetic, path_revised)

        # Step 6: Revisor estimates revision quality
        print(f"  Step 6: Revisor estimating revision quality...")
        revisor.estimate(path_revised, path_revised,
                         path_train, task_name=task_name, meta_path=meta_path)

        # Step 7: Verifier validates revised results
        path_verified = str(Path(save_dir) / "synthetic" / f"{i}_verified.jsonl")
        print(f"  Step 7: Verifier validating results...")
        verifier.verify(path_revised, path_verified)

        # Step 8: Verifier estimates verification quality (rule-based)
        print(f"  Step 8: Verifier estimating verification quality...")
        verifier.estimate(path_verified, path_verified)

        # Step 9: Filter data using combined four-agent reward
        print(f"  Step 9: Filtering data with four-agent reward...")
        path_filtered = str(Path(save_dir) / "filtered" / f"{i}.jsonl")
        filter_data_four_agent(
            path_pseudo=path_verified,
            path_train=path_train,
            path_out=path_filtered,
            total_pseudo_per_label=num_gen_per_label,
            pseudo_ratio=(i + 1.) / num_iter,
            task_name=task_train_data,
            unseen_meta_path=unseen_meta,
            meta_path=meta_path,
            sampling_cfg=sampling_cfg
        )

        # Step 10: Retrain all agents on filtered data
        print(f"  Step 10: Retraining all agents...")

        # Re-initialize agents for next iteration
        extractor = Extractor(
            load_dir=str(Path(save_dir) / "extractor" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "extractor" / f'iter{i + 1}'),
        )

        detector = Detector(
            load_dir=str(Path(save_dir) / "detector" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "detector" / f'iter{i + 1}'),
        )

        generator = Generator(
            load_dir=gen_PLM_path,
            lora_dir=str(Path(save_dir) / "generator" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "generator" / f'iter{i + 1}'),
            num_gen_per_label=num_gen_per_label + 100,
            encoder_name=g_encoder_name,
        )

        revisor = Revisor(
            load_dir=rev_PLM_path,
            lora_dir=str(Path(save_dir) / "revisor" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "revisor" / f'iter{i + 1}'),
        )

        verifier = Verifier(
            load_dir=ver_PLM_path,
            lora_dir=str(Path(save_dir) / "verifier" / f'iter{i}' / "model"),
            save_dir=str(Path(save_dir) / "verifier" / f'iter{i + 1}'),
        )

        # Train all agents
        print(f"    Training Extractor iter{i + 1}...")
        extractor.fit(path_filtered, path_dev, task_train_data, meta_path,
                      ontology_dict_path, iter=i + 1)

        print(f"    Training Detector iter{i + 1}...")
        detector.fit(path_filtered, path_dev, task_train_data, meta_path,
                     ontology_dict_path, iter=i + 1)

        print(f"    Training Generator iter{i + 1}...")
        generator.fit(path_filtered, path_dev, task_train_data, meta_path,
                      iter=i + 1)

        print(f"    Training Revisor iter{i + 1}...")
        revisor.fit(path_filtered, path_dev, task_train_data, meta_path,
                    iter=i + 1)

        print(f"    Training Verifier iter{i + 1}...")
        verifier.fit(path_filtered, path_dev, task_train_data, meta_path,
                     iter=i + 1)

        # Evaluate
        print(f"    Evaluating Extractor iter{i + 1}...")
        run_eval(
            path_model=str(Path(save_dir) / "extractor" / f'iter{i + 1}'),
            path_test=path_test, mode='single', is_eval=True,
            limit=limit, meta_path=unseen_meta,
            ontology_dict_path=ontology_test_dict_path,
            task_name=task_test_data
        )

    print(f"\n{'=' * 70}")
    print("Four-Agent Pipeline Complete!")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        print("Warning: torch.use_deterministic_algorithms not available or failed")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Four-Agent (Detector+Extractor+Revisor+Verifier) Event Extraction Pipeline"
    )

    # Data paths
    parser.add_argument("--path-train", default="dataset/rams2rams/train.jsonl",
                        help="Path to training data")
    parser.add_argument("--path-dev", default="dataset/rams2rams/dev.jsonl",
                        help="Path to development data")
    parser.add_argument("--path-test", default="dataset/rams2rams/test.jsonl",
                        help="Path to test data")

    # Meta and ontology
    parser.add_argument("--seen-meta", default="dataset/rams2rams/meta_seen.json",
                        help="Path to seen event types meta file")
    parser.add_argument("--unseen-meta", default="dataset/rams2rams/meta_unseen.json",
                        help="Path to unseen event types meta file")
    parser.add_argument("--ontology-dict-path", default="dataset/rams2rams/ontology.csv",
                        help="Path to ontology CSV for training event types")
    parser.add_argument("--ontology-test-dict-path", default="dataset/rams2rams/ontology.csv",
                        help="Path to ontology CSV for test event types")
    parser.add_argument("--generator-unseen-label-path",
                        default="data/rams2rams/generator_unseen_label_rams2rams_code.json",
                        help="Path to generator unseen label prompts")

    # Output and training
    parser.add_argument("--save-dir", default="experiments/four_agent/output",
                        help="Directory to save all models and outputs")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducibility")
    parser.add_argument("--num-iter", type=int, default=5,
                        help="Number of iterative refinement iterations")
    parser.add_argument("--task-name", default='rams2rams', type=str,
                        help="Task name (format: source2target)")

    # Model paths
    parser.add_argument("--gen-plm-path", default="Meta-Llama-3-8B",
                        help="Path to Generator base LLM")
    parser.add_argument("--ext-plm-path",
                        default="facebook/bart-large",
                        help="Path to Extractor/Detector base model (BART)")
    parser.add_argument("--rev-plm-path", default="Meta-Llama-3-8B",
                        help="Path to Revisor base LLM")
    parser.add_argument("--ver-plm-path", default="Meta-Llama-3-8B",
                        help="Path to Verifier base LLM")

    # Generation config
    parser.add_argument("--num-gen-per-label", type=int, default=100,
                        help="Number of synthetic instances to generate per event type")
    parser.add_argument("--block-size", default=512, type=int,
                        help="Max sequence length for CLM models")

    # Sampling config
    parser.add_argument("--pseudo-empty-ratio", default=0.1, type=float,
                        help="Ratio of empty-argument instances to include")
    parser.add_argument("--pseudo-empty-threshold", default=2, type=float,
                        help="Minimum non-empty arguments to be considered non-empty")

    args = parser.parse_args()

    set_seed(args.seed)
    hf_set_seed(args.seed)

    main_four_agent(
        path_train=args.path_train,
        path_dev=args.path_dev,
        path_test=args.path_test,
        task_name=args.task_name,
        meta_path=args.seen_meta,
        unseen_meta=args.unseen_meta,
        ontology_dict_path=args.ontology_dict_path,
        ontology_test_dict_path=args.ontology_test_dict_path,
        generator_unseen_label_path=args.generator_unseen_label_path,
        save_dir=args.save_dir,
        num_iter=args.num_iter,
        num_gen_per_label=args.num_gen_per_label,
        gen_PLM_path=args.gen_plm_path,
        ext_PLM_path=args.ext_plm_path,
        rev_PLM_path=args.rev_plm_path,
        ver_PLM_path=args.ver_plm_path,
        sampling_cfg={
            'pseudo_empty_ratio': args.pseudo_empty_ratio,
            'pseudo_empty_threshold': args.pseudo_empty_threshold
        }
    )
