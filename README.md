#### You are in the branch `mutli`, which currently has a problem of duplicative new-molecule generations!
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
    --ncpus=16 \
    --save_dir=save_tpsa \
    --key_dirs mw_keys logp_keys sas_keys \
    --beta 0.1 1> train.out 2> train.err
```
The start of `train.out` would be like this:
```
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
