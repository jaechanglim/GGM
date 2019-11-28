# GGM-SSVAE
Graph generative model for molecules with semi-supervised variational autoencoder

### Required libraries
- PyTorch
- RDKit
- OpenBabel (or an executable `babel`)
- NumPy
- TensorbaordX
- tqdm
- sklearn
- seaborn
- pandas

## Package Installation Guide
1. Make `setup.py` file to the root level directory.
`setup.py` should contain the follows:
    ```
    from setuptools import setup, find_packages
    setup(name='GGM', version='1.0', packages=find_packages())
    ```
2. Use a virtual environment.

    If you already have virtual environment, go to #3.
    - Create virtual env
    ```
    python -m venv venv
    ```
    - Activate virtual env
    ```
    source ./venv/bin/activate (Linux, macOS) or ./venv/Scripts/activate (Win)
    ```
    Once you have made and activated a virtual environment, your console should give the name of the virtual environment in parenthesis

3. pip install your project in editable state.

    Install your top level package `GGM` using `pip`.
    
    In the root directory, run
    
    `pip install -e .`(note the dot, it stands for "current directory")
    
    You can also see that it is installed by using `pip freeze`

    ```
    (GGM) -bash-4.2$ pip install -e .
    Obtaining file:///your/root/directory
    Installing collected packages: GGM
      Running setup.py develop for GGM
    Successfully installed GGM
    (GGM) -bash-4.2$ pip freeze
    # Editable install with no version control (GGM==1.0)
    ```
4. Now you can run all of these codes with `GGM.` imports!

If you need more information about installing GGM as package, please visit
https://stackoverflow.com/questions/6323860/sibling-package-imports/50193944#50193944.

## Training model
You can train model with train dataset. You can handle **sampling ratio**
between labeled data and unlabeled data via `--active_ratio` . You can train
with ratio between labeled data and unlabeled data 1:5 as follow: 
```
OMP_NUM_THREADS=1 \
python -u ./train/vaetrain.py \
    --num_epochs 200 \
    --ncpus 30 \
    --smiles_path data/ChEMBL+STOCK/id_smiles_train.txt \
    --data_paths data/ChEMBL+STOCK/data_train.txt \
    --save_dir results/ChEMBL+STOCK/20190725T1311 \
    --beta1 0.1 \
    --beta2 1.0 \
    --save_every 32 \
    --active_ratio 0.833 > log/ChEMBL+STOCK/20190725T1311/ChEMBL1STOCK5_log.txt
```
Note: You can point tensorboard to the training loss result folder to
monitor the training process as follow:
```
tensorboard --logdir ./runs
```
Note that Default name of directory composed with node name, starting time so you need to change it. 

## Generate sample molecules
### Generating molecules from scaffolds
You can generate sample molecules with desired property value from scaffolds molecule with
property value between `min_scaffold_value` and `max_scaffold_value` as follow:
```
OMP_NUM_THREADS=1 \
python ./train/sample.py \
--ncpus 20 \
--item_per_cycle 5 \
--smiles_path ./data/ChEMBL+STOCK/id_smiles_CHEMBL_test.txt \
--data_path ./data/ChEMBL+STOCK/data_CHEMBL_test.txt \
--min_scaffold_value 5 \
--max_scaffold_value 6 \
--target_property 8 \
--save_fpath ./results/ChEMBL+STOCK/20190725T1311/save_20_0.pt \
--output_filename ./samples/ChEMBL+STOCK/20190725T1311/ChEMBL_56to8_epoch20.txt \
--stochastic \
--num_scaffolds 100
```
Note: If `num_scaffolds` is larger than number of possible scaffolds that have property in range, 
`num_scaffolds` automatically set to the number of scaffolds that have property value between
`min_scaffold_value` and `max_scaffold_value`.

Note that the number of generated molecules is `item_per_cycle * ncpus * num_scaffolds`.

### Organization of generated molecule data
You can simply organize the result file of generation as follow:
```
python ./utils/sample_organization.py \
--input_filename ./samples/ChEMBL+STOCK/20190725T1311/ChEMBL_56to8_epoch20.txt \
--output_filename ./samples/ChEMBL+STOCK/20190725T1311/ChEMBL_56to8_epoch20_organized.txt \
--expected_generation 10000
```
Note: `expected_generation` is the value of number of generated molecule, same with 
`item_per_cycle * ncpus * num_scaffolds` in molecule generation part.

## Calculate each epoch's predict loss
You can calculate the trained model's loss at the single epoch as follow:
```
OMP_NUM_THREADS=1 \
python ./utils/predict_loss.py \
--smiles_path ./data/ChEMBL/id_smiles_train.txt \
--data_path ./data/ChEMBL/data_train.txt \
--save_fpath ./results/ChEMBL+STOCK/20190725T1311/save_20_0.pt \
--test_smiles_path ./data/ChEMBL/id_smiles_test.txt \
--test_data_path ./data/ChEMBL/data_test.txt \
--existing_result True
```
Note: Tensorboard MAE loss shows `MAE / sqrt(2.0)` which is tensorboard loss value of existing result.
If you want to use your own result for loss calculation rather than existing result, set `--existing_result` to `False`.

## Plotting loss change per epoch
### Calculating multiple loss values from each epoch
You can make file about loss changes as epoch number changes as follow:
```
OMP_NUM_THREADS=1 \
python ./utils/predict_multiloss.py \
--ncpus 30 \
--save_dir ./results/ChEMBL+STOCK/20190725T1311 \
--max_epoch 200 \
--epoch_interval 10 \
--output_filename ./results/loss_predict/ChEMBL1STOCK5.txt \
--existing_result True
```

### Plotting loss graph based on loss values
You can make plot file with the file contains loss changes per epoch as follow: 
```
python ./utils/plot.py \
--file_path ./results/loss_predict/ChEMBL1STOCK5.txt \
--value_type mae
```
Note: Only mae, mse, r2 values can be plotted. You can see
result png files in `file_path` directory.
