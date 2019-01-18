# GGM 
Graph generative model for molecules

# Training command example
```
python -u vaetrain.py --item_per_cycle=128 --ncpus=16 --save_dir=save_tpsa/ --key_dir=tpsa_keys/ > output_tpsa
```
The start of `output_tpsa` would be like this:
```
0       0       0       26.472  26.154  0.084   0.234   51.823

1       0       229     3.454   2.821   0.455   0.178   51.861

2       0       458     2.622   2.071   0.413   0.138   51.802

...

16      0       3664    1.308   0.878   0.307   0.123   51.433

17      0       3893    1.258   0.847   0.301   0.111   50.898
```

# Example of generating new molecules

Target property = TPSA

Target value = 100, 130 

Scaffold value = 50.95

Scaffold = c1ccc(Nc2ncnc3c2oc2ccccc23)cc1

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

# effect of beta and stochastic sampling
* validity of generated molecules (deterministic sampling)

256 times of molecule generation

|          property | target value | beta | number of valid molecules | after remove duplicates |
| ------------- | ------------- |------------- |------------- |------------- |
| TPSA       | 80 | 0.1 |  256| 32 |
| TPSA       | 80 | 0.2 |  256| 5 |
| TPSA       | 80 | 0.3 |  255| 10 |
| TPSA       | 80 | 0.5 |  256| 7 |
| TPSA       | 80 | 1.0 |  256| 6 |
| TPSA       | 100 | 0.1 |  189| 61 |
| TPSA       | 100 | 0.2 |  256| 48 |
| TPSA       | 100 | 0.3 |  254| 8 |
| TPSA       | 100 | 0.5 |  255| 47 |
| TPSA       | 100 | 1.0 |  256| 5 |
| TPSA       | 120 | 0.1 |  195| 51 |
| TPSA       | 120 | 0.2 |  256| 39 |
| TPSA       | 120 | 0.3 |  160| 24 |
| TPSA       | 120 | 0.5 |  256| 36 |
| TPSA       | 120 | 1.0 |  256| 19 |

* validity of generated molecules (stochastic sampling)

256 times of molecule generation

|          property | target value | beta | number of valid molecules | after remove duplicates |
| ------------- | ------------- |------------- |------------- |------------- |
| TPSA       | 80 | 0.1 |  240| 201 |
| TPSA       | 80 | 0.2 |  250| 156 |
| TPSA       | 80 | 0.3 |  254| 192 |
| TPSA       | 80 | 0.5 |  248| 184 |
| TPSA       | 80 | 1.0 |  253| 162 |
| TPSA       | 100 | 0.1 |  207| 182 |
| TPSA       | 100 | 0.2 |  224| 214|
| TPSA       | 100 | 0.3 |  183| 164|
| TPSA       | 100 | 0.5 |  217| 208 |
| TPSA       | 100 | 1.0 |  204| 174|
| TPSA       | 120 | 0.1 |  201| 194 |
| TPSA       | 120 | 0.2 |  229| 216|
| TPSA       | 120 | 0.3 |  198| 189|
| TPSA       | 120 | 0.5 |  221| 214|
| TPSA       | 120 | 1.0 |  204| 185|

