from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
try:
    from filelock import FileLock
except ImportError:
    class FileLock:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from na_design_cycle import run_na_design_cycle
from na_sequence_utils import (
    load_json,
    parse_random_init_chain_spec,
    parse_template_chains,
    template_format,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HalluDesign_NA with NA-MPNN and AF3/Protenix."
    )
    parser.add_argument("--optimizer", choices=["af3", "protenix"], required=True)
    parser.add_argument("--input_file", type=str, required=False)
    parser.add_argument("--pdb_list", type=str, required=False)
    parser.add_argument("--template_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_seqs", type=int, default=8)
    parser.add_argument("--num_recycles", type=int, default=10)
    parser.add_argument("--design_epoch_begin", type=int, default=0)
    parser.add_argument(
        "--ref_time_steps",
        type=int,
        default=50,
        help=(
            "Number of ref-guided denoising steps for the optimization stage. "
            "Set to 200 for pure prediction without coordinate input."
        ),
    )
    parser.add_argument("--mpnn_temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--random_init", action="store_true", default=False)
    parser.add_argument(
        "--random_init_chain_spec",
        type=str,
        default="",
        help="NA chain length spec for no-PDB random init, e.g. B:20-24,C:18.",
    )
    parser.add_argument(
        "--fix_res_index",
        type=str,
        default="",
        help="NA residues kept fixed during NA-MPNN design, e.g. 'B1 B4'.",
    )
    parser.add_argument(
        "--redesign_res_index",
        type=str,
        default="",
        help="If set, only these NA residues are redesigned, e.g. 'B1 B4'.",
    )
    parser.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="Direct NA-MPNN identity tying string, e.g. 'B1,C1|B2,C2'.",
    )
    parser.add_argument("--symmetry_chains", type=str, default="")
    parser.add_argument(
        "--symmetry_segments",
        type=int,
        default=0,
        help=(
            "Tie repeated segments within the first designed NA chain. "
            "For example 3 ties residue i, i+L/3, i+2L/3."
        ),
    )
    parser.add_argument(
        "--symmetry_segment_chain",
        type=str,
        default="",
        help="Optional chain ID for --symmetry_segments; defaults to the first designed NA chain.",
    )
    parser.add_argument(
        "--na_mpnn_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "NA-MPNN"),
    )
    parser.add_argument("--af3_num_samples", type=int, default=5)
    return parser.parse_args()


def load_pdb_files(args) -> list[str | None]:
    if args.input_file and args.pdb_list:
        raise ValueError("Cannot specify both --input_file and --pdb_list.")
    if args.random_init_chain_spec and (args.input_file or args.pdb_list):
        raise ValueError(
            "--random_init_chain_spec is for no-PDB random init; do not combine it "
            "with --input_file or --pdb_list."
        )
    if args.input_file:
        return [args.input_file]
    if args.pdb_list:
        with open(args.pdb_list, "r") as handle:
            return [line.strip() for line in handle if line.strip()]
    if args.random_init_chain_spec:
        args.random_init = True
        return [None]
    raise ValueError("Specify either --input_file or --pdb_list.")


def build_backend(args, specs):
    if args.optimizer == "af3":
        from backends.af3_backend import AF3Backend

        return AF3Backend(
            repo_root=str(REPO_ROOT),
            template_path=args.template_path,
            specs=specs,
            output_dir=args.output_dir,
            num_samples=args.af3_num_samples,
        )
    os.environ.setdefault("LAYERNORM_TYPE", "fast_layernorm")
    os.environ.setdefault("USE_DEEPSPEED_EVO_ATTENTION", "true")
    os.environ.setdefault("CUTLASS_PATH", str(REPO_ROOT / "cutlass"))
    from backends.protenix_backend import ProtenixBackend

    return ProtenixBackend(
        repo_root=str(REPO_ROOT),
        template_path=args.template_path,
        specs=specs,
    )


def append_metrics(csv_path: str, metrics: list[dict], lock: FileLock) -> None:
    with lock:
        exists = os.path.exists(csv_path)
        pd.DataFrame(metrics).to_csv(csv_path, mode="a", header=not exists, index=False)


def main():
    args = parse_args()
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    fmt = template_format(load_json(args.template_path))
    if args.optimizer == "af3" and fmt != "af3":
        raise ValueError("Use an AlphaFold 3 JSON template with --optimizer af3.")
    if args.optimizer == "protenix" and fmt != "protenix":
        raise ValueError("Use a Protenix JSON-list template with --optimizer protenix.")

    specs, counts = parse_template_chains(args.template_path)
    if counts["dna"] + counts["rna"] == 0:
        raise ValueError("Template contains no DNA/RNA chain for HalluDesign_NA.")
    random_init_chain_ranges = parse_random_init_chain_spec(
        args.random_init_chain_spec,
        specs,
    )
    active_symmetry_modes = sum(
        bool(value)
        for value in (args.symmetry_residues, args.symmetry_chains, args.symmetry_segments)
    )
    if active_symmetry_modes > 1:
        raise ValueError(
            "Use only one of --symmetry_residues, --symmetry_chains, or --symmetry_segments."
        )

    from na_mpnn_wrapper import NAMPNNDesigner

    na_designer = NAMPNNDesigner(
        na_mpnn_dir=args.na_mpnn_dir,
        temperature=args.mpnn_temperature,
    )
    backend = build_backend(args, specs)

    csv_path = os.path.join(args.output_dir, "processing_results.csv")
    lock = FileLock(f"{csv_path}.lock")

    for pdb_file in load_pdb_files(args):
        current_input = pdb_file
        file_tag = Path(pdb_file).stem.lower() if pdb_file else "random_init"
        print(f"Processing {pdb_file if pdb_file else 'no-PDB random initialization'}")
        for cycle in range(args.num_recycles):
            print(f"Starting NA design cycle {cycle + 1}")
            design_begin = cycle >= args.design_epoch_begin
            num_seqs = args.num_seqs if design_begin else 1
            no_pdb_start = current_input is None
            metrics, next_input = run_na_design_cycle(
                current_input=current_input,
                cycle=cycle,
                output_dir=args.output_dir,
                file_tag=file_tag,
                specs=specs,
                na_designer=na_designer,
                backend=backend,
                num_seqs=num_seqs,
                design_begin=design_begin,
                fixed_residues=args.fix_res_index.split() if args.fix_res_index else [],
                redesigned_residues=(
                    args.redesign_res_index.split() if args.redesign_res_index else []
                ),
                symmetry_residues=args.symmetry_residues,
                symmetry_chains=args.symmetry_chains,
                symmetry_segments=args.symmetry_segments,
                symmetry_segment_chain=args.symmetry_segment_chain,
                ref_time_steps=args.ref_time_steps,
                seed=args.seed,
                random_init=args.random_init,
                random_init_chain_ranges=random_init_chain_ranges,
                run_optimization=(cycle != args.num_recycles - 1) or no_pdb_start,
            )
            append_metrics(csv_path, metrics, lock)
            current_input = next_input

    print(f"Processing completed. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
