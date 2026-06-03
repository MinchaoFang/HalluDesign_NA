from __future__ import annotations
import os
import random
import shutil
from pathlib import Path
from typing import Any

from backends.common import copy_prediction_to_cycle_output
from na_sequence_utils import NACandidate, ChainSpec, random_na_sequence
from na_symmetry import generate_identity_symmetry_from_chains, generate_segment_symmetry


def run_na_design_cycle(
    current_input: str | None,
    cycle: int,
    output_dir: str,
    file_tag: str,
    specs: list[ChainSpec],
    na_designer,
    backend,
    num_seqs: int,
    design_begin: bool,
    fixed_residues: list[str],
    redesigned_residues: list[str],
    symmetry_residues: str,
    symmetry_chains: str,
    symmetry_segments: int,
    symmetry_segment_chain: str,
    ref_time_steps: int,
    seed: int,
    random_init: bool = False,
    random_init_chain_ranges: dict[str, tuple[int, int]] | None = None,
    run_optimization: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    target_dir = os.path.join(output_dir, f"recycle_{cycle + 1}")
    os.makedirs(target_dir, exist_ok=True)
    cycle_prefix = os.path.join(target_dir, f"{file_tag}_recycle_{cycle + 1}")
    copied_file = f"{cycle_prefix}.pdb"
    no_pdb_random_init = current_input is None
    if no_pdb_random_init:
        if not (random_init and cycle == 0):
            raise ValueError("No-PDB NA design is only supported for cycle 1 random init.")
    else:
        shutil.copy(current_input, copied_file)

    design_chain_ids = [spec.chain_id for spec in specs if spec.chain_type in {"dna", "rna"}]
    if not design_chain_ids:
        raise ValueError("HalluDesign_NA requires at least one DNA or RNA chain.")

    if no_pdb_random_init and (symmetry_chains or symmetry_segments or symmetry_residues):
        print("NA symmetry constraints will start after the first predicted PDB is available.")
    elif symmetry_chains:
        symmetry_residues = generate_identity_symmetry_from_chains(copied_file, symmetry_chains)
        print(f"NA-MPNN identity symmetry: {symmetry_residues}")
    elif symmetry_segments:
        segment_chain = symmetry_segment_chain or design_chain_ids[0]
        if segment_chain not in design_chain_ids:
            raise ValueError(
                f"--symmetry_segment_chain {segment_chain} is not a designed NA chain."
            )
        symmetry_residues = generate_segment_symmetry(
            copied_file,
            segment_chain,
            int(symmetry_segments),
        )
        print(f"NA-MPNN segment symmetry: {symmetry_residues}")
    elif symmetry_residues:
        print(f"NA-MPNN explicit symmetry: {symmetry_residues}")

    if random_init and cycle == 0:
        candidates = _random_init_candidates(
            specs=specs,
            design_chain_ids=design_chain_ids,
            copied_file=None if no_pdb_random_init else copied_file,
            num_seqs=max(1, int(num_seqs)),
            chain_length_ranges=random_init_chain_ranges or {},
            seed=seed + cycle,
        )
    else:
        candidates = na_designer.design(
            pdb_path=copied_file,
            output_dir=copied_file.replace(".pdb", "_na_mpnn_eval"),
            num_seqs=max(1, int(num_seqs)),
            design_chain_ids=design_chain_ids,
            fixed_residues=fixed_residues,
            redesigned_residues=redesigned_residues,
            symmetry_residues=symmetry_residues,
            seed=seed + cycle,
        )

    metrics = []
    eval_dir = f"{cycle_prefix}_{backend.backend_name}_eval"
    os.makedirs(eval_dir, exist_ok=True)
    if design_begin:
        print(f"{backend.backend_name} NA self-consistency evaluation")
        for index, candidate in enumerate(candidates):
            name = f"{file_tag}_recycle_{cycle + 1}_{index}"
            json_path = backend.write_candidate_json(
                output_dir=eval_dir,
                name=name,
                sequence_by_chain=candidate.sequence_by_chain,
                seed=seed,
            )
            prediction = backend.predict_and_score(
                json_path=json_path,
                output_dir=eval_dir,
                name=name,
                seed=seed,
                use_refinement=False,
            )
            metric = _candidate_metric(
                file_tag=file_tag,
                cycle=cycle,
                copied_file=None if no_pdb_random_init else copied_file,
                candidate=candidate,
                prediction_metrics=prediction.metrics,
                prediction_pdb=prediction.pdb_path,
                chain_ids=[spec.chain_id for spec in specs],
            )
            metrics.append(metric)
            print(_format_candidate_log(index, len(candidates), metric))
        metrics.sort(
            key=lambda item: _ranking_score(item),
            reverse=True,
        )
        best = metrics[0]
    else:
        print(f"no {backend.backend_name} NA self-consistency evaluation")
        best_candidate = candidates[0]
        best = _unevaluated_candidate_metric(
            file_tag=file_tag,
            cycle=cycle,
            copied_file=None if no_pdb_random_init else copied_file,
            candidate=best_candidate,
        )
        metrics.append(best)
    print(
        "Selected NA candidate: "
        f"ranking={best.get('eval_ranking_score')}, "
        f"path={best.get('eval_pdb_path')}"
    )
    best_candidate = best["_candidate"]
    if not run_optimization:
        for metric in metrics:
            metric.pop("_candidate", None)
            metric["selected"] = metric is best
            metric["HalluDesign_Status"] = "not_run_last_cycle"
        return metrics, best_candidate.packed_path

    op_prediction = _run_ref_guided_optimization(
        backend=backend,
        target_dir=target_dir,
        file_tag=file_tag,
        cycle=cycle,
        best_candidate=best_candidate,
        ref_time_steps=ref_time_steps,
        seed=seed,
        random_init=random_init,
    )
    for metric in metrics:
        metric.pop("_candidate", None)
        metric["selected"] = metric is best
        metric["HalluDesign_Status"] = "success" if metric is best else "evaluated"
    _merge_op_metrics(best, op_prediction)
    return metrics, op_prediction.pdb_path


def _candidate_metric(
    file_tag: str,
    cycle: int,
    copied_file: str | None,
    candidate: NACandidate,
    prediction_metrics: dict[str, Any],
    prediction_pdb: str,
    chain_ids: list[str],
) -> dict[str, Any]:
    metric = {
        "file_name": file_tag,
        "cycle": cycle,
        "origin_path": copied_file,
        "packed_path": candidate.packed_path,
        "mpnn_model": "NA-MPNN",
        "mpnn_sequence": candidate.mpnn_sequence,
        "na_sequence_by_chain": repr(candidate.sequence_by_chain),
        "na_mpnn_score": candidate.na_mpnn_score,
        "eval_status": "success",
        "eval_pdb_path": prediction_pdb,
        "_candidate": candidate,
    }
    metric.update(prediction_metrics)
    try:
        from data.utility import calculate_average_b_factor

        metric["eval_plddt"] = calculate_average_b_factor(
            prediction_metrics["eval_path"],
            chain_ids,
        )
    except Exception:
        metric.setdefault("eval_plddt", None)
    return metric


def _unevaluated_candidate_metric(
    file_tag: str,
    cycle: int,
    copied_file: str | None,
    candidate: NACandidate,
) -> dict[str, Any]:
    return {
        "file_name": file_tag,
        "cycle": cycle,
        "origin_path": copied_file,
        "packed_path": candidate.packed_path,
        "mpnn_model": "NA-MPNN",
        "mpnn_sequence": candidate.mpnn_sequence,
        "na_sequence_by_chain": repr(candidate.sequence_by_chain),
        "na_mpnn_score": candidate.na_mpnn_score,
        "eval_status": "not_run_before_design_epoch",
        "eval_pdb_path": None,
        "_candidate": candidate,
    }


def _run_ref_guided_optimization(
    backend,
    target_dir: str,
    file_tag: str,
    cycle: int,
    best_candidate: NACandidate,
    ref_time_steps: int,
    seed: int,
    random_init: bool,
):
    name = f"{file_tag}_recycle_{cycle + 1}"
    op_dir = os.path.join(target_dir, f"{name}_{backend.backend_name}_op")
    pred_dir = os.path.join(target_dir, f"{name}_{backend.backend_name}_pred")
    os.makedirs(op_dir, exist_ok=True)
    json_path = backend.write_candidate_json(
        output_dir=target_dir,
        name=name,
        sequence_by_chain=best_candidate.sequence_by_chain,
        seed=seed,
    )
    pure_prediction = bool((random_init and cycle == 0) or ref_time_steps >= 200)
    if pure_prediction:
        print(f"{backend.backend_name} NA optimization: pure prediction")
        prediction = backend.predict_and_score(
            json_path=json_path,
            output_dir=pred_dir,
            name=name,
            seed=seed,
            use_refinement=False,
        )
    else:
        ref_input = backend.prepare_ref_input(
            pdb_path=best_candidate.packed_path,
            output_dir=op_dir,
            name=name,
        )
        print(
            f"{backend.backend_name} NA optimization: ref-guided "
            f"{ref_time_steps} steps from {best_candidate.packed_path}"
        )
        prediction = backend.predict_and_score(
            json_path=json_path,
            output_dir=pred_dir,
            name=name,
            seed=seed,
            use_refinement=True,
            ref_pdb_path=ref_input,
            ref_time_steps=ref_time_steps,
        )
    return copy_prediction_to_cycle_output(prediction, op_dir)


def _merge_op_metrics(metric: dict[str, Any], op_prediction) -> None:
    metric["op_cif_path"] = op_prediction.cif_path
    metric["op_pdb_path"] = op_prediction.pdb_path
    metric["HalluDesign_Status"] = "success"
    for key, value in op_prediction.metrics.items():
        if key == "eval_path":
            continue
        if key.startswith("eval_"):
            metric["op_" + key[len("eval_") :]] = value
        else:
            metric[f"op_{key}"] = value


def _random_init_candidates(
    specs: list[ChainSpec],
    design_chain_ids: list[str],
    copied_file: str | None,
    num_seqs: int,
    chain_length_ranges: dict[str, tuple[int, int]] | None = None,
    seed: int = 0,
) -> list[NACandidate]:
    candidates = []
    chain_type_by_id = {spec.chain_id: spec.chain_type for spec in specs}
    length_by_id = {
        spec.chain_id: len(spec.sequence)
        for spec in specs
        if spec.chain_id in set(design_chain_ids)
    }
    rng = random.Random(seed)
    chain_length_ranges = chain_length_ranges or {}
    for index in range(num_seqs):
        sequence_by_chain = {
            chain_id: random_na_sequence(
                _random_init_chain_length(
                    chain_id,
                    length_by_id,
                    chain_length_ranges,
                    rng,
                ),
                chain_type_by_id[chain_id],
                rng=rng,
            )
            for chain_id in design_chain_ids
        }
        candidates.append(
            NACandidate(
                sequence_by_chain=sequence_by_chain,
                mpnn_sequence="/".join(sequence_by_chain[chain] for chain in design_chain_ids),
                packed_path=copied_file or "",
                na_mpnn_score=None,
                metadata={"random_init_index": index},
            )
        )
    return candidates


def _random_init_chain_length(
    chain_id: str,
    template_lengths: dict[str, int],
    chain_length_ranges: dict[str, tuple[int, int]],
    rng,
) -> int:
    if chain_id in chain_length_ranges:
        start, end = chain_length_ranges[chain_id]
        return rng.randint(start, end)
    length = int(template_lengths.get(chain_id, 0))
    if length <= 0:
        raise ValueError(
            f"No random-init length is available for NA chain {chain_id}. "
            "Add it to --random_init_chain_spec, e.g. B:20."
        )
    return length


def _format_candidate_log(
    index: int,
    total: int,
    metric: dict[str, Any],
) -> str:
    return (
        f"NA candidate {index + 1}/{total}: "
        f"ranking={metric.get('eval_ranking_score')}, "
        f"iptm={metric.get('eval_iptm')}, "
        f"ptm={metric.get('eval_ptm')}, "
        f"plddt={metric.get('eval_plddt')}, "
        f"seq={metric.get('mpnn_sequence')}"
    )


def _ranking_score(metric: dict[str, Any]) -> float:
    value = metric.get("eval_ranking_score")
    try:
        return float(value)
    except Exception:
        return float("-inf")
