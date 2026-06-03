# HalluDesign_NA

`HalluDesign_NA` is an independent HalluDesign interface for nucleic-acid
sequence optimization. It uses NA-MPNN to propose DNA/RNA sequences and AF3 or
Protenix as the structure oracle for self-consistency evaluation.

This directory is intentionally separated from the original HalluDesign runner:
all NA-design code lives under `HalluDesign_NA/`, while the original
HalluDesign files are left unchanged.

## Scope

- Protein chains are kept fixed in the first implementation.
- DNA/RNA chains are redesigned by NA-MPNN.
- AF3 or Protenix evaluates NA-MPNN candidates after `--design_epoch_begin`.
- The selected NA-MPNN packed structure is then passed into AF3/Protenix as
  ref-guided coordinates for an optimization prediction.
- The optimized structure is converted to PDB and recycled into the next cycle.
- `--symmetry_chains` supports identity tying, for example `B,C` means
  `B1,C1|B2,C2|...`. This is not Watson-Crick reverse-complement tying.
- `--symmetry_residues` can pass an explicit NA-MPNN tying string.
- `--symmetry_segments N` ties repeated identity segments within one NA chain,
  following the same modulo idea as HalluDesign protein segment symmetry.

The self-consistency evaluation stage uses sequence-only structure prediction
to rank candidate NA sequences. The optimization stage is ref-guided: it uses
the selected candidate PDB coordinates as the starting structure and continues
denoising for `--ref_time_steps`, matching the original HalluDesign pattern.

## Environment

The computational environment for HalluDesign-NA follows the requirements of
the original HalluDesign framework. Please refer to the installation
instructions and environment setup provided at:

```text
https://github.com/MinchaoFang/HalluDesign
```

Although HalluDesign-NA is maintained as a separate repository, it is intended
to be cloned inside the original HalluDesign directory for execution:

```text
HalluDesign/
  HalluDesign_NA/
```

The runner reuses HalluDesign's AF3 wrapper, Protenix wrapper, coordinate
preprocessing utilities, and evaluation helpers from the parent HalluDesign
repository. Running HalluDesign-NA outside the HalluDesign directory is not the
recommended layout.

## NA-MPNN Dependency

By default the runner expects NA-MPNN at:

```text
HalluDesign_NA/NA-MPNN
```

You can also pass another clone:

```bash
--na_mpnn_dir /path/to/NA-MPNN
```

The wrapper imports NA-MPNN in-process and loads the checkpoint once.

## Design Logic

Each cycle copies the current best structure and runs NA-MPNN on the DNA/RNA
chains. Before `--design_epoch_begin`, one NA-MPNN sequence is generated and
sent directly to the coordinate-guided optimization stage. From
`--design_epoch_begin` onward, `--num_seqs` NA-MPNN candidates are first
evaluated by AF3 or Protenix, the highest-ranked candidate is selected, and then
that selected candidate's packed PDB coordinates are used as the ref-guided
optimization input.

The final cycle follows the original HalluDesign convention: it records the
NA-MPNN/self-consistency result but does not launch another expensive
optimization prediction.

For no-PDB random initialization, omit `--input_file/--pdb_list` and provide
`--random_init_chain_spec`, for example `B:20,C:18-24`. The chain IDs must
already exist as DNA/RNA chains in the template JSON; this option sets the
initial random NA sequence length but does not add new chains to the template.
Cycle 1 is run as pure AF3/Protenix prediction, and later cycles recycle the
predicted PDB into the normal NA-MPNN/ref-guided loop.

Useful controls:

```bash
--fix_res_index "B1 B2"
--redesign_res_index "B10 B11 B12"
--symmetry_residues "B1,C1|B2,C2"
--symmetry_chains B,C
--symmetry_segments 3 --symmetry_segment_chain B
--ref_time_steps 150
--random_init_chain_spec "B:20,C:18-24"
```

`--fix_res_index` and `--redesign_res_index` use PDB-style chain+residue labels
as parsed by NA-MPNN. If `--redesign_res_index` is provided, only those residues
are sampled and all other residues are fixed.

## Examples

All example commands below are intended to be run from the `HalluDesign_NA`
directory after cloning it inside the original HalluDesign repository:

```bash
cd /path/to/HalluDesign/HalluDesign_NA
```

Protenix:

```bash
python ./HalluDesign_NA_run.py \
  --optimizer protenix \
  --template_path examples/RNA_Protenix.json \
  --output_dir examples/HalluDesign_NA_random \
  --random_init_chain_spec "B:20" \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10
```

AF3:

```bash
python ./HalluDesign_NA_run.py \
  --optimizer af3 \
  --template_path examples/DNA_af3.json \
  --output_dir examples/HalluDesign_NA_DNA_af3_random \
  --random_init_chain_spec "B:20" \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10
```

## Selection

When self-consistency evaluation is enabled, candidates are selected by the
structure model's ranking score. This matches the default AF3/Protenix behavior
and avoids adding an extra selection interface for routine use.

## Citation

```bibtex
@article {Fang2025.11.08.686881,
	author = {Fang, Minchao and Wang, Chentong and Shi, Jungang and Lian, Fangbai and Jin, Qihan and Wang, Zhe and Zhang, Yanzhe and Cui, Zhanyuan and Wang, YanJun and Ke, Yitao and Han, Qingzheng and Cao, Longxing},
	title = {HalluDesign: Protein Optimization and de novo Design via Iterative Structure Hallucination and Sequence design},
	elocation-id = {2025.11.08.686881},
	year = {2025},
	doi = {10.1101/2025.11.08.686881},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/11/09/2025.11.08.686881},
	eprint = {https://www.biorxiv.org/content/early/2025/11/09/2025.11.08.686881.full.pdf},
	journal = {bioRxiv}
}

@article{Abramson2024,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J. and Bambrick, Joshua and Bodenstein, Sebastian W. and Evans, David A. and Hung, Chia-Chun and O’Neill, Michael and Reiman, David and Tunyasuvunakool, Kathryn and Wu, Zachary and Žemgulytė, Akvilė and Arvaniti, Eirini and Beattie, Charles and Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and Congreve, Miles and Cowen-Rivers, Alexander I. and Cowie, Andrew and Figurnov, Michael and Fuchs, Fabian B. and Gladman, Hannah and Jain, Rishub and Khan, Yousuf A. and Low, Caroline M. R. and Perlin, Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine and Yakneen, Sergei and Zhong, Ellen D. and Zielinski, Michal and Žídek, Augustin and Bapst, Victor and Kohli, Pushmeet and Jaderberg, Max and Hassabis, Demis and Jumper, John M.},
  journal = {Nature},
  title   = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  year    = {2024},
  volume  = {630},
  number  = {8016},
  pages   = {493–-500},
  doi     = {10.1038/s41586-024-07487-w}
}

@article{bytedance2025protenix,
  title={Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction},
  author={ByteDance AML AI4Science Team and Chen, Xinshi and Zhang, Yuxuan and Lu, Chan and Ma, Wenzhi and Guan, Jiaqi and Gong, Chengyue and Yang, Jincai and Zhang, Hanyu and Zhang, Ke and Wu, Shenghao and Zhou, Kuangqi and Yang, Yanping and Liu, Zhenyu and Wang, Lan and Shi, Bo and Shi, Shaochen and Xiao, Wenzhi},
  year={2025},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.01.08.631967},
  URL={https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967},
  elocation-id={2025.01.08.631967},
  eprint={https://www.biorxiv.org/content/early/2025/01/11/2025.01.08.631967.full.pdf},
}

@article{kubaney2025nammpnn,
  title={RNA sequence design and protein--DNA specificity prediction with NA-MPNN},
  author={Kubaney, Andrew and Favor, Andrew and McHugh, Lilian and Mitra, Raktim and Pecoraro, Robert and Dauparas, Justas and Glasscock, Cameron and Baker, David},
  year={2025},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2025.10.03.679414},
  URL={https://www.biorxiv.org/content/early/2025/10/07/2025.10.03.679414},
  elocation-id={2025.10.03.679414},
  eprint={https://www.biorxiv.org/content/early/2025/10/07/2025.10.03.679414.full.pdf},
}

```
# Licence and Disclaimer

## HalluDesign
HalluDesign all code are under [MIT license](https://github.com/MinchaoFang/HalluDesign/blob/main/LICENSE) 

## AlphaFold 3 Source Code and Model Parameters

alphafold3 is not an officially supported Google product.

Copyright 2024 DeepMind Technologies Limited.

The AlphaFold 3 source code is licensed under the Creative Commons
Attribution-Non-Commercial ShareAlike International License, Version 4.0
(CC-BY-NC-SA 4.0) (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
[https://github.com/google-deepmind/alphafold3/blob/main/LICENSE](https://github.com/google-deepmind/alphafold3/blob/main/LICENSE).

The AlphaFold 3 model parameters are made available under the
[AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
(the "Terms"); you may not use these except in compliance with the Terms. You
may obtain a copy of the Terms at
[https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

## Protenix

The Protenix project including both code and model parameters is released under the [Apache 2.0 License](https://github.com/bytedance/Protenix/blob/main/LICENSE). It is free for both academic research and commercial use.

## NA-MPNN

NA-MPNN are under [MIT license](https://github.com/baker-laboratory/NA-MPNN/blob/main/LICENSE)
