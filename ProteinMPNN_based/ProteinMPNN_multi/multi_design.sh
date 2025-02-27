#!/bin/bash

python ./helper_scripts/parse_multiple_chains.py --input_path=./dual_pdb --output_path=./test.jsonl

python ./helper_scripts/assign_fixed_chains.py --input_path=./test.jsonl --output_path=./assigned_pdbs.jsonl --chain_list "P"

python ./helper_scripts/make_fixed_positions_dict.py --input_path=./test.jsonl --output_path=./fixed_pdbs.jsonl --chain_list "P" --position_list "1 2 3"

python ./protein_mpnn_data_p.py \
        --jsonl_path ./test.jsonl \
        --chain_id_jsonl ./assigned_pdbs.jsonl \
        --fixed_positions_jsonl ./fixed_pdbs.jsonl \
        --out_folder ./1/ \
        --num_seq_per_target 10 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1


python main.py --checkpoint_path ./vanilla_model_weights/v_48_020.pt --T 0.1 --design_num 2 --conformation_num 2 --output_path ./test1_2025_2_17.fasta
