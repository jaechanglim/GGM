# GGM 
Graph generative model for molecules

# Training command
```
python -u vaetrain.py --item_per_cycle=128 --ncpus=16 --save_dir=save_tpsa/ --key_dir=tpsa_keys/ > output_tpsa
```
part of `output_tpsa` would be like this:
```
0       0       0       26.472  26.154  0.084   0.234   51.823

1       0       229     3.454   2.821   0.455   0.178   51.861

2       0       458     2.622   2.071   0.413   0.138   51.802

...

16      0       3664    1.308   0.878   0.307   0.123   51.433

17      0       3893    1.258   0.847   0.301   0.111   50.898
```

# Example of generating new molecules

target property: TPSA

target value = 100, 130 

scaffold value = 50.95

scaffold = c1ccc(Nc2ncnc3c2oc2ccccc23)cc1

```
python sample.py \
    --save_fpath=save_tpsa/save_15_200.pt \
    --target_property=130 \
    --scaffold_property=50.95 \
    --output_filename=generated_tpsa_130.txt \
    --maximum_value=150 \
    --minimum_value=0 \
    --scaffold='c1ccc(Nc2ncnc3c2oc2ccccc23)cc1' \
    --ncpus=8 \
    --item_per_cycle=16
```

Distribution plot of the generated molecules:

![TPSA](./TPSA.jpg)
