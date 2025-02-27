<img width="1527" alt="f1c" src="https://github.com/user-attachments/assets/9e48e8a3-9bc4-4091-b6e7-854d9e8a0af4" />

"""
We provide the processed dual-target experimental data and test datasets from the NoiseMix stage. 
The links are as follows:
https://drive.google.com/drive/folders/1-hIm_vwzXJAwDMJqAOuQJyYNQWVvFqVO?usp=drive_link

- x_train_multi.pt: Training data
- x_test_multi.pt: Test data
- x_test_sim_rmsd.pt / x_train_sim_rmsd.pt: RMSD values for the homologous and conformational changes of two protein receptors in the dataset under dual-target conditions
- dict_x_test_30_159.pt, dict_x_test_sim_50_rmsd_2.pt, lst_diff_inter_38_data.pt: Test sets for three different conditions

Additionally, different subsets of the test set can be re-extracted based on various conditions using a function from the test set.
"""
