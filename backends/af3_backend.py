from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from backends.common import BackendPrediction, StructureBackend, convert_cif_to_pdb


class AF3Backend(StructureBackend):
    backend_name = "af3"

    def __init__(
        self,
        repo_root: str,
        template_path: str,
        specs,
        output_dir: str,
        num_samples: int = 5,
    ):
        super().__init__(template_path=template_path, specs=specs)
        sys.path.insert(0, str(Path(repo_root)))
        from af3_model import AF3DesignerPack

        self.designer = AF3DesignerPack(
            jax_compilation_dir=os.path.join(output_dir, "jax_compilation_cache_dir")
        )
        self.num_samples = int(num_samples)

    def prepare_ref_input(
        self,
        pdb_path: str,
        output_dir: str,
        name: str,
    ) -> str:
        from local_scripts.input_pkl_preprocess import process_single_file

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        pdb_name = os.path.basename(pdb_path)
        local_pdb = os.path.join(output_dir, pdb_name)
        if os.path.abspath(pdb_path) != os.path.abspath(local_pdb):
            import shutil

            shutil.copy(pdb_path, local_pdb)
        success, _, error = process_single_file((pdb_name, output_dir, output_dir), insert=None)
        if not success:
            raise RuntimeError(f"AF3 ref coordinate preprocessing failed: {error}")
        return os.path.join(output_dir, f"{Path(pdb_name).stem}.pkl")

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
        results = self.designer.single_file_process(
            json_path=json_path,
            out_dir=output_dir,
            ref_pdb_path=ref_pdb_path if use_refinement and ref_pdb_path else None,
            ref_time_steps=int(ref_time_steps) if use_refinement else 0,
            num_samples=self.num_samples,
        )
        best_result = None
        best_score = float("-inf")
        for seed_results in results:
            for inference_result in seed_results.inference_results:
                score = float(inference_result.metadata["ranking_score"])
                if score > best_score:
                    best_result = inference_result
                    best_score = score
        if best_result is None:
            raise RuntimeError("AF3 returned no inference result.")

        cif_path = os.path.join(output_dir, name.lower(), f"{name.lower()}_model.cif")
        pdb_path = cif_path.replace(".cif", ".pdb")
        convert_cif_to_pdb(cif_path, pdb_path)
        metadata = best_result.metadata
        metrics = self._common_metrics(
            ranking_score=metadata.get("ranking_score", best_score),
            ptm=metadata.get("ptm"),
            iptm=metadata.get("iptm"),
            chain_ptm=metadata.get("iptm_ichain"),
            chain_pair_iptm=metadata.get("chain_pair_iptm"),
            cif_path=cif_path,
            plddt=metadata.get("plddt"),
        )
        try:
            metrics["eval_pae"] = float(np.mean(best_result.numerical_data["full_pae"]))
            metrics["eval_pde"] = float(np.mean(best_result.numerical_data["full_pde"]))
        except Exception:
            metrics["eval_pae"] = None
            metrics["eval_pde"] = None
        return BackendPrediction(
            cif_path=cif_path,
            pdb_path=pdb_path,
            metrics=metrics,
            results=results,
        )
