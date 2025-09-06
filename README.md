# ProDualNet: Dual-Target Protein Sequence Design Method Based on Protein Language Model and Structure Model, Briefings in Bioinformatics, 2025.

Important update: We have successfully designed a dual agonist using this model and are currently progressing with efficacy experiments. Please wait for our preprint—August 1, 2025.

Install Python>=3.0, PyTorch, Numpy.

- The main folder includes the execution code and test cases for ProdualNet. You can use it to design dual-target protein sequences, such as GLP-1/GCGR dual agonists, or proteins designed to bind with different receptors, causing conformational changes, using the weights produalnet_02.pt. 

- The mutation_task folder is for a zero-shot protein function prediction task, including thermal stability and DDG. 

Similar to recent work on conformational bias in mutations.

- The baseline folder on this project contains a modified multi-state design model based on ProteinMPNN, supporting multiple target protein sequence design and multiple protein complex conformations sequence design.

<img width="1773" alt="f1d" src="https://github.com/user-attachments/assets/c74fca2a-3af3-430f-a866-24b0913beaf0" />
<img width="1773" alt="f1d" src="https://github.com/chengliu97/ProDualNet/blob/main/%E5%9B%BE%E7%89%871.png" />

- The current model only supports the design of natural amino acids.

@: sjtu6597802@sjtu.edu.cn;
--------------------------------------------------------------------------------
- You may not use the material for commercial purposes.

This project is based on ProteinMPNN/Pifold/esm/BERT-pytorch, under their License.

Source: https://github.com/dauparas/ProteinMPNN, https://github.com/A4Bio/PiFold, https://github.com/facebookresearch/esm, 
https://github.com/codertimo/BERT-pytorch/tree/master
