from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from na_metrics import chain_pair_mean, to_float
from na_sequence_utils import ChainSpec, load_json, update_template_sequences, write_json


@dataclass
class BackendPrediction:
    cif_path: str
    pdb_path: str
    metrics: dict[str, Any]
    results: Any


class StructureBackend:
    backend_name = "base"

    def __init__(self, template_path: str, specs: list[ChainSpec]):
        self.template_path = str(Path(template_path).expanduser().resolve())
        self.template = load_json(self.template_path)
        self.specs = specs

    def write_candidate_json(
        self,
        output_dir: str,
        name: str,
        sequence_by_chain: dict[str, str],
        seed: int,
    ) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        data = update_template_sequences(
            template=self.template,
            specs=self.specs,
            sequence_by_chain=sequence_by_chain,
            name=name,
            model_seed=seed,
        )
        path = os.path.join(output_dir, f"{name}.json")
        write_json(data, path)
        return path

    def predict_and_score(
        self,
        json_path: str,
        output_dir: str,
        name: str,
        seed: int,
        use_refinement: bool = False,
        ref_pdb_path: str = "",
        ref_time_steps: int = 0,
    ) -> BackendPrediction:
        raise NotImplementedError

    def prepare_ref_input(
        self,
        pdb_path: str,
        output_dir: str,
        name: str,
    ) -> str:
        return pdb_path

    def _common_metrics(
        self,
        ranking_score: Any,
        ptm: Any,
        iptm: Any,
        chain_ptm: Any,
        chain_pair_iptm: Any,
        cif_path: str,
        plddt: Any = None,
    ) -> dict[str, Any]:
        protein_indices = [
            index for index, spec in enumerate(self.specs) if spec.chain_type == "protein"
        ]
        other_indices = [
            index for index, spec in enumerate(self.specs) if spec.chain_type != "protein"
        ]
        interface_iptm = chain_pair_mean(chain_pair_iptm, protein_indices, other_indices)
        return {
            "prediction_model": self.backend_name,
            "eval_path": cif_path,
            "eval_ranking_score": to_float(ranking_score),
            "eval_plddt": to_float(plddt),
            "eval_ptm": to_float(ptm),
            "eval_iptm": to_float(iptm),
            "eval_all_iptm_to_protein": interface_iptm if interface_iptm is not None else 0.0,
            "chain_ptm": _to_list(chain_ptm),
            "chain_pair_iptm": _to_list(chain_pair_iptm),
        }


def _to_list(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def convert_cif_to_pdb(cif_path: str, pdb_path: str) -> str:
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("prediction", cif_path)
    # Use PDBIO for PDB output; import lazily to keep this helper explicit.
    from Bio.PDB import PDBIO

    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(pdb_path)
    return pdb_path


def copy_prediction_to_cycle_output(prediction: BackendPrediction, output_dir: str) -> BackendPrediction:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cif_path = os.path.join(output_dir, os.path.basename(prediction.cif_path))
    pdb_path = os.path.join(output_dir, os.path.basename(prediction.pdb_path))
    if os.path.abspath(prediction.cif_path) != os.path.abspath(cif_path):
        shutil.copy(prediction.cif_path, cif_path)
    if os.path.abspath(prediction.pdb_path) != os.path.abspath(pdb_path):
        shutil.copy(prediction.pdb_path, pdb_path)
    metrics = dict(prediction.metrics)
    metrics["eval_path"] = cif_path
    return BackendPrediction(
        cif_path=cif_path,
        pdb_path=pdb_path,
        metrics=metrics,
        results=prediction.results,
    )
