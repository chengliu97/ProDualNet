For evaluating thermostability mutations, we utilized two databases: ProteinGym DMS[1] and DMS_stability[2,3]. From ProteinGym DMS, we selected 65 proteins with mutations linked to thermostability, which included 67,636 single-point mutations and 65,579 higher-order mutations. For the DMS_stability dataset, we worked with their pre-processed data, encompassing 298 proteins and a total of 271,231 single-point mutations.

Furthermore, to assess performance in predicting the fitness of complex mutations, we selected data from the SKEMPI V2 database[4]. This subset focused on antibody mutations, containing over 100 entries, and included 11 proteins, 1,877 single-point mutations, and 675 higher-order mutations.

DMS_mut_produalnet_pre.ipynb is the evaluation process for the DMS_stability dataset. skempi_v2_ddg_produalnet_pre.ipynb is the DDG evaluation process for 11 antibody systems. The comparison method execution scripts are in the baseline folder.

ProteinGym DMS evaluation is based on https://github.com/OATML-Markslab/ProteinGym.

https://drive.google.com/drive/folders/1u-yKZglYbHhzckpHjW5uqc6mN0Ze0oAv?usp=drive_link

Ref:
1. Notin, P. et al. Proteingym: Large-scale benchmarks for protein fitness prediction and design. Advances in Neural Information Processing Systems 36 (2024).
2. Tsuboyama, K. et al. Mega-scale experimental analysis of protein folding stability in biology and design. Nature 620, 434-444 (2023).
3. Dieckhaus, H., Brocidiacono, M., Randolph, N. Z. & Kuhlman, B. Transfer learning to leverage larger datasets for improved prediction of protein stability changes. Proceedings of the National Academy of Sciences 121, e2314853121 (2024).
4. Jankauskaitė, J., Jiménez-García, B., Dapkūnas, J., Fernández-Recio, J. & Moal, I. H. SKEMPI 2.0: an updated benchmark of changes in protein–protein binding energy, kinetics and thermodynamics upon mutation. Bioinformatics 35, 462-469 (2019).
