from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DNA_BASES = "ACGT"
RNA_BASES = "ACGU"

NA_MPNN_TO_STANDARD = {
    "a": "A",
    "c": "C",
    "g": "G",
    "t": "T",
    "x": "X",
    "b": "A",
    "d": "C",
    "h": "G",
    "u": "U",
    "y": "X",
}

STANDARD_TO_NA_MPNN_DNA = {
    "A": "a",
    "C": "c",
    "G": "g",
    "T": "t",
    "X": "x",
}

STANDARD_TO_NA_MPNN_RNA = {
    "A": "b",
    "C": "d",
    "G": "h",
    "U": "u",
    "X": "y",
}


@dataclass
class ChainSpec:
    chain_id: str
    chain_type: str
    template_index: int
    sequence: str = ""


@dataclass
class NACandidate:
    sequence_by_chain: dict[str, str]
    mpnn_sequence: str
    packed_path: str
    na_mpnn_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_json(path: str | Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path) -> None:
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def is_protenix_template(template: Any) -> bool:
    return isinstance(template, list)


def get_template_job(template: Any) -> dict[str, Any]:
    if isinstance(template, list):
        if not template:
            raise ValueError("Template JSON list is empty.")
        return template[0]
    return template


def template_format(template: Any) -> str:
    return "protenix" if is_protenix_template(template) else "af3"


def parse_template_chains(template_path: str | Path) -> tuple[list[ChainSpec], dict[str, int]]:
    template = load_json(template_path)
    job = get_template_job(template)
    specs: list[ChainSpec] = []
    counts = {"protein": 0, "ligand": 0, "dna": 0, "rna": 0}
    chain_ids = iter(string.ascii_uppercase)
    fmt = template_format(template)

    for index, item in enumerate(job.get("sequences", [])):
        if fmt == "protenix":
            if "proteinChain" in item:
                block = item["proteinChain"]
                _append_specs(specs, counts, chain_ids, index, "protein", block)
            elif "dnaSequence" in item:
                block = item["dnaSequence"]
                _append_specs(specs, counts, chain_ids, index, "dna", block)
            elif "rnaSequence" in item:
                block = item["rnaSequence"]
                _append_specs(specs, counts, chain_ids, index, "rna", block)
            elif "ligand" in item:
                block = item["ligand"]
                _append_specs(specs, counts, chain_ids, index, "ligand", block)
            else:
                raise ValueError(f"Unexpected Protenix sequence block at index {index}: {item}")
        else:
            if "protein" in item:
                block = item["protein"]
                _append_specs(specs, counts, chain_ids, index, "protein", block)
            elif "dna" in item:
                block = item["dna"]
                _append_specs(specs, counts, chain_ids, index, "dna", block)
            elif "rna" in item:
                block = item["rna"]
                _append_specs(specs, counts, chain_ids, index, "rna", block)
            elif "ligand" in item:
                block = item["ligand"]
                _append_specs(specs, counts, chain_ids, index, "ligand", block)
            else:
                raise ValueError(f"Unexpected AF3 sequence block at index {index}: {item}")
    return specs, counts


def _append_specs(
    specs: list[ChainSpec],
    counts: dict[str, int],
    chain_ids: Any,
    template_index: int,
    chain_type: str,
    block: dict[str, Any],
) -> None:
    ids = block.get("id")
    count = int(block.get("count", 1))
    if ids is not None:
        if isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = list(ids)
    else:
        id_list = [next(chain_ids) for _ in range(count)]
    sequence = block.get("sequence", "")
    for chain_id in id_list:
        specs.append(
            ChainSpec(
                chain_id=str(chain_id),
                chain_type=chain_type,
                template_index=template_index,
                sequence=sequence,
            )
        )
        counts[chain_type] += 1


def chain_types_from_specs(specs: list[ChainSpec]) -> list[str]:
    return [spec.chain_type for spec in specs]


def na_chain_ids(specs: list[ChainSpec]) -> list[str]:
    return [spec.chain_id for spec in specs if spec.chain_type in {"dna", "rna"}]


def random_na_sequence(
    length: int,
    chain_type: str,
    rng: random.Random | None = None,
) -> str:
    alphabet = DNA_BASES if chain_type == "dna" else RNA_BASES
    chooser = rng.choice if rng is not None else random.choice
    return "".join(chooser(alphabet) for _ in range(length))


def parse_random_init_chain_spec(
    spec: str,
    chain_specs: list[ChainSpec],
) -> dict[str, tuple[int, int]]:
    """Parse no-PDB random-init NA chain lengths.

    Format: ``B:20,C:18-24``. Chain IDs must already exist as DNA/RNA chains in
    the template; this function does not add new chains to the JSON template.
    """
    if not spec:
        return {}
    chain_type_by_id = {item.chain_id: item.chain_type for item in chain_specs}
    parsed: dict[str, tuple[int, int]] = {}
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid --random_init_chain_spec entry '{entry}'. Use B:20 or B:18-24."
            )
        chain_id, length_text = [part.strip() for part in entry.split(":", 1)]
        if chain_id not in chain_type_by_id:
            raise ValueError(
                f"--random_init_chain_spec chain {chain_id} is not present in the template."
            )
        if chain_type_by_id[chain_id] not in {"dna", "rna"}:
            raise ValueError(
                f"--random_init_chain_spec chain {chain_id} is {chain_type_by_id[chain_id]}, "
                "but only DNA/RNA chains can be specified."
            )
        if "-" in length_text:
            start_text, end_text = [part.strip() for part in length_text.split("-", 1)]
        else:
            start_text = end_text = length_text
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid length in --random_init_chain_spec entry '{entry}'."
            ) from exc
        if start <= 0 or end <= 0 or start > end:
            raise ValueError(
                f"Invalid length range in --random_init_chain_spec entry '{entry}'."
            )
        parsed[chain_id] = (start, end)
    return parsed


def convert_na_mpnn_sequence(sequence: str) -> str:
    return "".join(NA_MPNN_TO_STANDARD.get(char, char) for char in sequence)


def split_chain_sequence(sequence: str) -> list[str]:
    return [part for part in sequence.strip().split("/") if part != ""]


def parse_na_mpnn_fasta(path: str | Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    header = None
    seq_lines: list[str] = []
    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        entries.append((header, "".join(seq_lines)))
    return entries


def get_chain_order_from_pdb(pdb_path: str | Path) -> list[str]:
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", str(pdb_path))
    model = next(structure.get_models())
    return [chain.id for chain in model]


def map_fasta_sequence_to_chains(
    fasta_sequence: str,
    pdb_chain_order: list[str],
    design_chain_ids: set[str],
) -> dict[str, str]:
    parts = split_chain_sequence(convert_na_mpnn_sequence(fasta_sequence))
    if len(parts) != len(pdb_chain_order):
        raise ValueError(
            "NA-MPNN FASTA chain count does not match PDB chain count: "
            f"{len(parts)} sequence parts vs {len(pdb_chain_order)} chains."
        )
    by_chain = dict(zip(pdb_chain_order, parts))
    return {chain_id: by_chain[chain_id] for chain_id in pdb_chain_order if chain_id in design_chain_ids}


def update_template_sequences(
    template: Any,
    specs: list[ChainSpec],
    sequence_by_chain: dict[str, str],
    name: str,
    model_seed: int | None = None,
) -> Any:
    import copy

    data = copy.deepcopy(template)
    job = get_template_job(data)
    job["name"] = name
    if model_seed is not None and not is_protenix_template(data):
        job["modelSeeds"] = [int(model_seed)]

    fmt = template_format(data)
    updated_blocks: dict[tuple[int, str], str] = {}
    for spec in specs:
        if spec.chain_id not in sequence_by_chain:
            continue
        item = job["sequences"][spec.template_index]
        sequence = sequence_by_chain[spec.chain_id]
        block_key = (spec.template_index, spec.chain_type)
        if block_key in updated_blocks and updated_blocks[block_key] != sequence:
            raise ValueError(
                "Template blocks with count > 1 require identical sequences. "
                f"Chain {spec.chain_id} has a different {spec.chain_type} sequence; "
                "split this block into one JSON sequence entry per chain."
            )
        updated_blocks[block_key] = sequence
        if fmt == "protenix":
            if spec.chain_type == "dna":
                item["dnaSequence"]["sequence"] = sequence
            elif spec.chain_type == "rna":
                item["rnaSequence"]["sequence"] = sequence
            elif spec.chain_type == "protein":
                item["proteinChain"]["sequence"] = sequence
        else:
            if spec.chain_type == "dna":
                item["dna"]["sequence"] = sequence
            elif spec.chain_type == "rna":
                item["rna"]["sequence"] = sequence
            elif spec.chain_type == "protein":
                item["protein"]["sequence"] = sequence
    return data
