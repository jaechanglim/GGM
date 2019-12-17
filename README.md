### Scaffold-based molecular design with a graph generative model
by [Jaechang Lim](https://github.com/jaechanglim), [Sang-Yeon Hwang](https://github.com/sangyeon-hwang), [Seokhyun Moon](https://github.com/mseok), [Seungsu Kim](https://github.com/SeungsuKim) and [Woo Youn Kim](https://aceteamkaist.wixsite.com/home), [*Chem. Sci.* 2020, Advance Article](https://doi.org/10.1039/C9SC04503A), [arXiv:1905.13639](https://arxiv.org/abs/1905.13639).

See [this branch](https://github.com/jaechanglim/GGM/tree/branch_feature_predictor_merging) for the semi-supervised version (Section 3.6).

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
    --smiles_path data/id_smiles.txt \
    --data_paths data/mw/data_normalized.txt data/logp/data_normalized.txt data/sas/data_normalized.txt \
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
