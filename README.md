# DeepResolve
This repo contains the data and the code for genomic neural network visualization tool DeepResolve.

## Code
The `Code` directory contains code for general DeepResolve Feature Importance Map and OFIV generation for Keras models, and task specific scripts for reproducing the experimental results appeared in the paper. 

### Run FIV generation on a given Keras model
A trained Keras model should contain a `.json` architecture file and a `.h5` model weight file in the same path. Run the following command to conduct gradient ascent and generate FIVs for `T` times.
```
python FIM_generation.py <modelpath> <model_file_suffix> <resultdir> <L2_coeff> <Learning_rate> <T>
```
This will produce a `importance_map-<L2coeff>-<LearnRate>` file under `<resultdir>` that contains T generated FIVs, and a `importance_score-<L2coeff>-<LearnRate>` file that contains class output score for these FIVs.

Run
```
python OFIV_generation.py <NIV_file_path> <NIV_score_path> <weight_dir> <resultdir>
```
To generate an Overall Feature Importance Map as well as the Inconsistency Level (variance) of each feature channel.

### Run 422 TF experiment
The CNN model files for each TF binding prediction task is located under `models/422tf` where each folder stands from one ChIP-seq experiment. Run following to reproduce the TOMTOM score matching results for DeepResolve.
```
bash code/script/422tf_deepresolve.sh
```
