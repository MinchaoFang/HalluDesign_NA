#!/usr/bin/env bash

# Run this file from the HalluDesign_NA directory, for example:
# cd /path/to/HalluDesign/HalluDesign_NA

# AF3 NA design example.
python ./HalluDesign_NA_run.py \
  --optimizer af3 \
  --input_file ../examples/RNA_binder/input.pdb \
  --template_path examples/RNA_af3.json \
  --output_dir examples/HalluDesign_NA_af3 \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10

# Protenix NA design example.
python ./HalluDesign_NA_run.py \
  --optimizer protenix \
  --input_file ../examples/RNA_binder/input.pdb \
  --template_path examples/RNA_Protenix.json \
  --output_dir examples/HalluDesign_NA_protenix \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10

# Protenix NA design with identity-tied repeated segments in chain B.
python ./HalluDesign_NA_run.py \
  --optimizer protenix \
  --input_file ../examples/RNA_binder/input.pdb \
  --template_path examples/RNA_Protenix.json \
  --output_dir examples/HalluDesign_NA_segment_sym \
  --symmetry_segments 4 \
  --symmetry_segment_chain B \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10

# Protenix no-PDB random-init NA design.
python ./HalluDesign_NA_run.py \
  --optimizer protenix \
  --template_path examples/RNA_Protenix.json \
  --output_dir examples/HalluDesign_NA_random \
  --random_init_chain_spec "B:20" \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10

# AF3 no-PDB random-init DNA design.
python ./HalluDesign_NA_run.py \
  --optimizer af3 \
  --template_path examples/DNA_af3.json \
  --output_dir examples/HalluDesign_NA_DNA_af3_random \
  --random_init_chain_spec "B:20" \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10

# Protenix no-PDB random-init DNA design.
python ./HalluDesign_NA_run.py \
  --optimizer protenix \
  --template_path examples/DNA_Protenix.json \
  --output_dir examples/HalluDesign_NA_DNA_protenix_random \
  --random_init_chain_spec "B:20" \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10

# Protenix no-PDB random-init RNA monomer design.
python ./HalluDesign_NA_run.py \
  --optimizer protenix \
  --template_path examples/RNA_monomer_Protenix.json \
  --output_dir examples/HalluDesign_NA_RNA_monomer_random \
  --random_init_chain_spec "A:20" \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10

# Protenix no-PDB random-init DNA monomer design.
python ./HalluDesign_NA_run.py \
  --optimizer protenix \
  --template_path examples/DNA_monomer_Protenix.json \
  --output_dir examples/HalluDesign_NA_DNA_monomer_random \
  --random_init_chain_spec "A:20" \
  --ref_time_steps 150 \
  --num_seqs 8 \
  --num_recycles 10
