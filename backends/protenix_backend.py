from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from backends.common import BackendPrediction, StructureBackend, convert_cif_to_pdb
from na_metrics import best_summary_index


class ProtenixBackend(StructureBackend):
    backend_name = "protenix"

    def __init__(
        self,
        repo_root: str,
        template_path: str,
        specs,
        n_cycle: int = 10,
        n_sample: int = 5,
        n_step: int = 200,
        use_esm: bool = True,
        use_msa: bool = False,
    ):
        super().__init__(template_path=template_path, specs=specs)
        protenix_root = Path(repo_root) / "Protenix"
        sys.path.insert(0, str(protenix_root))
        from runner.inference import ProtenixInferrer

        self.inferrer = ProtenixInferrer(
            **{
                "model.N_cycle": int(n_cycle),
                "sample_diffusion.N_sample": int(n_sample),
                "sample_diffusion.N_step": int(n_step),
                "use_esm": bool(use_esm),
                "use_msa": bool(use_msa),
                "need_atom_confidence": True,
                "sorted_by_ranking_score": True,
            },
        )

    def prepare_ref_input(
        self,
        pdb_path: str,
        output_dir: str,
        name: str,
    ) -> str:
        import torch
        from eval.evaluation import extract_coordinates, read_pdb_to_atom_array

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pkl_path = os.path.join(output_dir, f"{name}_atom_array.pkl")
        atom_array = read_pdb_to_atom_array(pdb_path)
        coordinates = extract_coordinates(atom_array, as_tensor=True)
        torch.save(coordinates, pkl_path)
        return pkl_path

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
        kwargs: dict[str, Any] = {}
        if use_refinement and ref_pdb_path and ref_time_steps > 0:
            kwargs["input_atom_array_path"] = ref_pdb_path
            kwargs["diffusion_steps"] = int(ref_time_steps)
        results = self.inferrer.predict(
            input_json_path=json_path,
            dump_dir=output_dir,
            seed=int(seed),
            **kwargs,
        )
        best_idx = best_summary_index(results)
        summary = results["summary_confidence"][best_idx]
        cif_path = os.path.join(
            output_dir,
            name.lower(),
            f"seed_{seed}",
            "predictions",
            f"{name.lower()}_seed_{seed}_sample_0.cif",
        )
        pdb_path = cif_path.replace(".cif", ".pdb")
        convert_cif_to_pdb(cif_path, pdb_path)
        metrics = self._common_metrics(
            ranking_score=summary.get("ranking_score"),
            ptm=summary.get("ptm", summary.get("chain_ptm")),
            iptm=summary.get("iptm", summary.get("chain_iptm")),
            chain_ptm=summary.get("chain_ptm"),
            chain_pair_iptm=summary.get("chain_pair_iptm"),
            cif_path=cif_path,
            plddt=summary.get("plddt"),
        )
        return BackendPrediction(
            cif_path=cif_path,
            pdb_path=pdb_path,
            metrics=metrics,
            results=results,
        )
