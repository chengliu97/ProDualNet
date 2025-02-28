proteinmpnn_single_eval.ipynb and proteinmpnn_mean_eval.ipynb are the execution scripts for the comparison methods. proteinmpnn_mean_eval.ipynb can load different ProteinMPNN weights, including the ones fine-tuned on our dual-target data using NoiseMix. produalnet_eval_unconditional.ipynb is the recovery calculation script, which does not rely on an autoregressive approach. produalnet_design_fasta_eval.ipynb is the evaluation script based on an autoregressive approach. produalnet_dual_agonist_case.ipynb is a dual-agonist design script for GLP-1/GCGR, where you can specify conserved sites. The specific weights can be downloaded as follows:



This project is based on ProteinMPNN/Pifold/esm/BERT-pytorch, under the MIT License.

Source: https://github.com/dauparas/ProteinMPNN, https://github.com/A4Bio/PiFold, https://github.com/facebookresearch/esm, 
https://github.com/codertimo/BERT-pytorch/tree/master
