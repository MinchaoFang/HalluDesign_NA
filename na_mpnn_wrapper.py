from __future__ import annotations

import importlib.util
import random
import sys
import copy
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from na_sequence_utils import (
    NACandidate,
    map_fasta_sequence_to_chains,
)


class NAMPNNDependencyError(RuntimeError):
    """Raised when the external NA-MPNN dependency cannot be loaded."""


class NAMPNNDesigner:
    """In-process wrapper around the NA-MPNN inference implementation."""

    def __init__(
        self,
        na_mpnn_dir: str,
        checkpoint_path: str | None = None,
        device: str = "auto",
        temperature: float = 0.1,
        na_shared_tokens: bool = True,
    ):
        self.na_mpnn_dir = Path(na_mpnn_dir).expanduser().resolve()
        self.inference_dir = self.na_mpnn_dir / "inference"
        self.checkpoint_path = Path(
            checkpoint_path or self.na_mpnn_dir / "models" / "design_model" / "s_19137.pt"
        ).expanduser().resolve()
        self.device = self._resolve_device(device)
        self.temperature = float(temperature)
        self.na_shared_tokens = bool(na_shared_tokens)

        self._normalize_sys_path()
        self._validate_paths()
        self.data_utils = self._load_module("halludesign_na_mpnn_data_utils", "data_utils.py")
        self.model_utils = self._load_module("halludesign_na_mpnn_model_utils", "model_utils.py")
        self._configure_alphabet()
        self.model = self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _normalize_sys_path() -> None:
        cwd = os.getcwd()
        sys.path[:] = [cwd if item == "" else item for item in sys.path]

    def _validate_paths(self) -> None:
        if not self.inference_dir.is_dir():
            raise NAMPNNDependencyError(
                f"NA-MPNN inference directory not found: {self.inference_dir}"
            )
        if not self.checkpoint_path.exists():
            raise NAMPNNDependencyError(
                f"NA-MPNN checkpoint not found: {self.checkpoint_path}"
            )

    def _load_module(self, module_name: str, filename: str) -> ModuleType:
        module_path = self.inference_dir / filename
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise NAMPNNDependencyError(f"Cannot load NA-MPNN module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _configure_alphabet(self) -> None:
        self.atom_types = [
            "N",
            "CA",
            "C",
            "O",
            "OP1",
            "OP2",
            "P",
            "O5'",
            "C5'",
            "C4'",
            "O4'",
            "C3'",
            "O3'",
            "C2'",
            "O2'",
            "C1'",
        ]
        self.atom_dict = dict(zip(self.atom_types, range(len(self.atom_types))))

        polytypes = ["PP", "DNA", "RNA", "UNK", "MAS", "PAD"]
        self.polytype_to_int = dict(zip(polytypes, range(len(polytypes))))

        self.restypes = [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
            "UNK",
            "DA",
            "DC",
            "DG",
            "DT",
            "DX",
            "A",
            "C",
            "G",
            "U",
            "RX",
            "MAS",
            "PAD",
        ]
        self.restype_3_to_1 = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
            "UNK": "X",
            "DA": "a",
            "DC": "c",
            "DG": "g",
            "DT": "t",
            "DX": "x",
            "A": "b",
            "C": "d",
            "G": "h",
            "U": "u",
            "RX": "y",
            "MAS": "-",
            "PAD": "+",
        }
        self.restype_1_to_3 = {value: key for key, value in self.restype_3_to_1.items()}
        self.restype_to_int = dict(zip(self.restypes, range(len(self.restypes))))
        self.dna_char_to_rna_char: dict[str, str] = {}
        if self.na_shared_tokens:
            self.restype_to_int["A"] = self.restype_to_int["DA"]
            self.restype_to_int["C"] = self.restype_to_int["DC"]
            self.restype_to_int["G"] = self.restype_to_int["DG"]
            self.restype_to_int["U"] = self.restype_to_int["DT"]
            self.restype_to_int["RX"] = self.restype_to_int["DX"]
            self.dna_char_to_rna_char = {
                "a": "b",
                "c": "d",
                "g": "h",
                "t": "u",
                "x": "y",
            }
        self.restype_str_to_int = {
            self.restype_3_to_1[key]: value for key, value in self.restype_to_int.items()
        }
        self.restype_int_to_str = {}
        for key, value in self.restype_str_to_int.items():
            self.restype_int_to_str.setdefault(value, key)
        self.alphabet = [
            self.restype_3_to_1[self.restypes[index]] for index in range(len(self.restypes))
        ]
        self.num_letters = len(self.restypes)

    def _load_model(self):
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model = self.model_utils.ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=32,
            model_type="na_mpnn",
            vocab=self.num_letters,
            num_letters=self.num_letters,
            atom_dict=self.atom_dict,
            restype_to_int=self.restype_to_int,
            polytype_to_int=self.polytype_to_int,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        print(f"Loaded NA-MPNN checkpoint: {self.checkpoint_path}")
        return model

    def design(
        self,
        pdb_path: str,
        output_dir: str,
        num_seqs: int,
        design_chain_ids: list[str],
        fixed_residues: list[str] | None = None,
        redesigned_residues: list[str] | None = None,
        symmetry_residues: str = "",
        seed: int = 0,
    ) -> list[NACandidate]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        backbone_dir = output_path / "backbones"
        backbone_dir.mkdir(parents=True, exist_ok=True)

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        macromolecule_dict, backbone, other_atoms, icodes, _ = self.data_utils.parse_PDB(
            pdb_path,
            device=self.device,
            chains=[],
            model_type="na_mpnn",
            parse_na_only=False,
            na_shared_tokens=self.na_shared_tokens,
            load_residues_with_missing_atoms=0,
        )
        chain_order = list(macromolecule_dict["chain_list"])
        encoded_residues = self._encoded_residues(macromolecule_dict, icodes)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))

        fixed_residues = fixed_residues or []
        redesigned_residues = redesigned_residues or []
        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=self.device,
        )
        if redesigned_residues:
            redesigned_positions = torch.tensor(
                [int(item not in redesigned_residues) for item in encoded_residues],
                device=self.device,
            )
        else:
            redesigned_positions = torch.zeros_like(fixed_positions)

        design_chain_set = set(design_chain_ids)
        chain_mask = torch.tensor(
            np.array(
                [chain in design_chain_set for chain in macromolecule_dict["chain_letters"]],
                dtype=np.int32,
            ),
            device=self.device,
        )
        macromolecule_dict["chain_mask"] = chain_mask * fixed_positions * (1 - redesigned_positions)
        symmetry_groups, symmetry_weights = self._parse_symmetry(
            symmetry_residues,
            encoded_residue_dict,
        )

        with torch.no_grad():
            features = self.data_utils.featurize(macromolecule_dict)
            features["batch_size"] = int(num_seqs)
            _, length, _, _ = features["X"].shape
            omit_aa = self._omit_aa_mask()
            features["temperature"] = self.temperature
            features["bias"] = (-1e8 * omit_aa[None, None, :]).repeat([1, length, 1])
            features["symmetry_residues"] = symmetry_groups
            features["symmetry_weights"] = symmetry_weights
            features["randn"] = torch.randn(
                [int(num_seqs), features["mask"].shape[1]],
                device=self.device,
            )
            outputs = self.model.sample(features)
            scores, score_per_residue = self.data_utils.get_score(
                outputs["S"],
                outputs["log_probs"],
                features["mask"] * features["chain_mask"],
                self.num_letters,
            )

        candidates = []
        for index, sequence_tokens in enumerate(outputs["S"].cpu().numpy()):
            sequence = self._sequence_from_tokens(
                sequence_tokens,
                features["rna_mask_for_token_conversion"][0].cpu().numpy(),
            )
            fasta_sequence = self._chain_joined_sequence(sequence, macromolecule_dict["mask_c"])
            sequence_by_chain = map_fasta_sequence_to_chains(
                fasta_sequence,
                chain_order,
                design_chain_set,
            )
            packed_path = self._write_candidate_pdb(
                backbone=backbone,
                other_atoms=other_atoms,
                chain_letters=list(macromolecule_dict["chain_letters"]),
                residue_indices=list(macromolecule_dict["R_idx"].cpu().numpy()),
                sequence=sequence,
                per_residue_score=score_per_residue[index].cpu().numpy(),
                output_path=backbone_dir / f"candidate_{index + 1}.pdb",
            )
            candidates.append(
                NACandidate(
                    sequence_by_chain=sequence_by_chain,
                    mpnn_sequence="/".join(sequence_by_chain[chain] for chain in design_chain_ids),
                    packed_path=str(packed_path),
                    na_mpnn_score=float(torch.exp(-scores[index]).cpu().item()),
                    metadata={"fasta_sequence": fasta_sequence},
                )
            )
        return candidates

    @staticmethod
    def _encoded_residues(macromolecule_dict: dict[str, Any], icodes: list[str]) -> list[str]:
        residue_indices = list(macromolecule_dict["R_idx"].cpu().numpy())
        chain_letters = list(macromolecule_dict["chain_letters"])
        return [
            f"{chain}{residue_index}{icode}"
            for chain, residue_index, icode in zip(chain_letters, residue_indices, icodes)
        ]

    def _omit_aa_mask(self) -> torch.Tensor:
        omit_letters = {"X"}
        if self.na_shared_tokens:
            omit_letters.update("bdhuy")
        return torch.tensor(
            np.array([char in omit_letters for char in self.alphabet]).astype(np.float32),
            device=self.device,
        )

    @staticmethod
    def _parse_symmetry(
        symmetry_residues: str,
        encoded_residue_dict: dict[str, int],
    ) -> tuple[list[list[int]], list[list[float]]]:
        if not symmetry_residues:
            return [[]], [[]]
        groups = []
        for raw_group in symmetry_residues.split("|"):
            group = []
            for residue in raw_group.split(","):
                if residue not in encoded_residue_dict:
                    raise ValueError(f"Symmetry residue {residue} is not present in the parsed PDB.")
                group.append(encoded_residue_dict[residue])
            groups.append(group)
        return groups, [[1.0] * len(group) for group in groups]

    def _sequence_from_tokens(self, tokens: np.ndarray, rna_mask: np.ndarray) -> str:
        chars = []
        for index, token in enumerate(tokens):
            char = self.restype_int_to_str[int(token)]
            if int(rna_mask[index]) == 1:
                char = self.dna_char_to_rna_char.get(char, char)
            chars.append(char)
        return "".join(chars)

    @staticmethod
    def _chain_joined_sequence(sequence: str, masks: list[torch.Tensor]) -> str:
        chars = np.array(list(sequence))
        return "/".join("".join(chars[mask.cpu().numpy()]) for mask in masks)

    def _write_candidate_pdb(
        self,
        backbone: Any,
        other_atoms: Any,
        chain_letters: list[str],
        residue_indices: list[int],
        sequence: str,
        per_residue_score: np.ndarray,
        output_path: Path,
    ) -> Path:
        try:
            backbone_copy = backbone.copy()
        except AttributeError:
            backbone_copy = copy.deepcopy(backbone)
        try:
            other_atoms_copy = other_atoms.copy() if other_atoms is not None else None
        except AttributeError:
            other_atoms_copy = copy.deepcopy(other_atoms) if other_atoms is not None else None
        sequence_resnames = np.array([self.restype_1_to_3[char] for char in sequence])

        for index, (chain, residue_index) in enumerate(zip(chain_letters, residue_indices)):
            residue = backbone_copy.select(f"chain {chain} and resnum {residue_index}")
            if residue is None:
                continue
            residue.setResnames(sequence_resnames[index])
            score = float(np.exp(-per_residue_score[index])) if per_residue_score[index] > 0.01 else 0.0
            residue.setBetas(score)

        atoms = backbone_copy if other_atoms_copy is None else backbone_copy + other_atoms_copy
        from prody import writePDB

        writePDB(str(output_path), atoms)
        return output_path
