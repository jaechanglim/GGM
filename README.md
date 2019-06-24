# GGM 
Graph generative model for molecules

### Required libraries
- PyTorch
- RDKit
- OpenBabel (or an executable `babel`)
- NumPy
- Matplotlib (optional)

## Training command example
Target properties = MW, logP, SAS

```
python -u vaetrain.py \
    --ncpus 16 \
    --save_dir mw-logp-sas-0.1 \
    --smiles_path data_egfr/id_smiles.txt \
    --data_paths data_egfr/mw/data_normalized.txt data_egfr/logp/data_normalized.txt data_egfr/sas/data_normalized.txt \
    --beta1 0.1 1> train.out 2> train.err
```
For **unconditional training**, omit `--data_paths`.

Content of `train.out` after input information would be like this:
```
epoch   cyc     totcyc  loss    loss1   loss2   loss3   time
0       0       0       26.472  26.154  0.084   0.234   51.823

1       0       229     3.454   2.821   0.455   0.178   51.861

2       0       458     2.622   2.071   0.413   0.138   51.802

...

16      0       3664    1.308   0.878   0.307   0.123   51.433

17      0       3893    1.258   0.847   0.301   0.111   50.898
```

## Generation command example
Target properties = MW, logP, SAS

Target values = 310, 3, 4

Scaffold values = 310.1, 3.2, 2.07

Scaffold = "O=C(CSc1nnc(-c2ccccc2)[nH]1)Nc1ccccc1"
```
python sample.py \
    --ncpus 16 \
    --save_fpath "mw-logp-sas-0.1/save_10_10.pt" \
    --output_filename "mw-logp-sas-0.1/4S00001_310_3_4.txt" \
    --item_per_cycle 100 \
    --scaffold "O=C(CSc1nnc(-c2ccccc2)[nH]1)Nc1ccccc1" \
    --scaffold_properties 310.1 3.2 2.07 \
    --target_properties 310 3 4 \
    --minimum_values 200 0 0 \
    --maximum_values 550 8 5 \
    --stochastic
```
For **sampling using an unconditioned  model**, omit `--target_properties`, `--scaffold_properties`, `--minimum_values` and `--maximum_values`.


OMP_NUM_THREADS=1 \
python ./train/vaetrain.py \
--num_epochs 200 \
--ncpus 30 \
--smiles_path data/ChEMBL/id_smiles_train.txt \
--data_paths data/ChEMBL/data_train.txt \
--save_dir results/20190522T0100/ \
--beta1 0.1
--dropout 0.7

OMP_NUM_THREADS=1 \
python ./train/vaetrain.py \
--num_epochs 10 \
--ncpus 15 \
--smiles_path data_egfr/id_smiles.txt \
--data_paths data_egfr/logp/data.txt \
--save_dir results/20190515T2023/ \


python ./train/predict.py \
--smiles_path data/ChEMBL/id_smiles_test.txt \
--data_paths data/ChEMBL/data_test.txt \
--save_fpath results/20190522T0100/save_199_0.pt \
--dropout 0.7

python ./train/predict.py \
--smiles_path data/ChEMBL/id_smiles_train.txt \
--data_paths data/ChEMBL/data_normalized_train.txt \
--save_fpath results/20190519T1500/save_16_0.pt \
--dropout 0.5

util.make_graphs("Nc1ncnc2c1c(cn2C3CCC(O)C3)c4ccc(Oc5ccccc5)cc4", "c1ccc(Oc2ccc(-c3cn(C4CCCC4)c4ncncc34)cc2)cc1", extra_atom_feature=True, extra_bond_feature=True)