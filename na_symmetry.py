from __future__ import annotations

from pathlib import Path


def _standard_residues(pdb_path: str | Path, chain_id: str):
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("symmetry", str(pdb_path))
    model = next(structure.get_models())
    if chain_id not in model:
        raise ValueError(f"Chain {chain_id} is not present in {pdb_path}.")
    return [res for res in model[chain_id] if res.id[0] == " "]


def generate_identity_symmetry_from_chains(pdb_path: str | Path, chain_pair: str) -> str:
    """Generate NA-MPNN identity tying residues for comma-separated chains.

    Example: ``B,C`` becomes ``B1,C1|B2,C2`` up to the shortest chain.
    This is identity symmetry, not Watson-Crick complementarity.
    """
    chain_ids = [chain.strip() for chain in chain_pair.split(",") if chain.strip()]
    if len(chain_ids) < 2:
        raise ValueError("--symmetry_chains requires at least two comma-separated chains.")

    chain_residues = []
    for chain_id in chain_ids:
        chain_residues.append(_standard_residues(pdb_path, chain_id))

    group_count = min(len(residues) for residues in chain_residues)
    groups = []
    for index in range(group_count):
        groups.append(
            ",".join(
                f"{chain_id}{residues[index].id[1]}"
                for chain_id, residues in zip(chain_ids, chain_residues)
            )
        )
    return "|".join(groups)


def generate_segment_symmetry(
    pdb_path: str | Path,
    chain_id: str,
    segment_count: int,
) -> str:
    """Tie repeated identity segments inside one chain.

    This mirrors HalluDesign's modulo-segment symmetry for ProteinMPNN, but it is
    applied to one NA chain and passed to NA-MPNN as identity-tied residues.
    """
    if segment_count <= 1:
        return ""
    residues = _standard_residues(pdb_path, chain_id)
    total = len(residues)
    if total % segment_count != 0:
        raise ValueError(
            f"Chain {chain_id} length ({total}) is not divisible by "
            f"--symmetry_segments ({segment_count})."
        )
    stride = total // segment_count
    groups = []
    for offset in range(stride):
        group = []
        for segment_index in range(segment_count):
            residue = residues[offset + segment_index * stride]
            group.append(f"{chain_id}{residue.id[1]}")
        groups.append(",".join(group))
    return "|".join(groups)
