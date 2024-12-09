# Use Perturbations when Learning from Explanations

Code accompanying submission titled "Use Perturbations when Learning from Explanations" to NeurIPS 2023. 

Algorithm implementations can be found in `src/module` and dataset implementations in `src/datasets`.   
Hyperparameter searches and best hyperoparameters are documented in `src/configs/expts.py`.

## How to run the code

### Dependencies

- [Python and Conda](https://www.anaconda.com/)
- Setup the conda environment IBPEx by running:

    ```bash
    conda env create -f setup_dependencies.yml
    ```
  If you want to make use of your GPU, you might have to install a cuda-enabled pytorch version manually. Use the appropriate command provided [here](https://pytorch.org/) to achieve this.
  Before running code, activate the environment IBPEx.

    ```bash
    source activate IBPEx
    ```

### Datasets

- We have one synthetic dataset (decoy-MNIST) and two real dataset (ISIC and Plant phenotyping).
- The default experiment settings for each dataset are in `src/configs/dataset_defaults.py`

**The dataset splits that we used in the paper can be found at this [link](https://1drv.ms/f/c/361ed53576f381ed/Ep8nnnUiFvhLjyD3PcquNy8BkjqJ2Ej1ZmnA4WPJTZrZxA?e=vzHzg7)**. 


### Algorithms

- We have 5 methods (RRR, IBP-Ex, PGD-Ex, CDEP, CoRM) in `src/module`, whose parent module is `base_trainer.py`.
- The default experiment settings for each method are in `src/configs/alg_defaults.py`

### Experiments

- Run the code with the best hyperparameter setting we found for each dataset in `src/configs/expts.py`.
- For example, to train the model using our IBP-Ex method on the Plant phenotyping dataset,

    ``` bash
    python main.py --name 'ibp_plant_expt3'
    ```

- View all possible command-line options by running

    ``` bash
    python main.py --help
    ```    

# Contact
Please reach out to Vihari Piratla (viharipiratla [at] gmail) if you have any questions.

# Cite
If you found our code/dataset/paper useful, please cite using the following bibtex entry.

```
@inproceedings{
    heo2023use,
    title={Use perturbations when learning from explanations},
    author={Juyeon Heo and Vihari Piratla and Matthew Robert Wicker and Adrian Weller},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=guyhQMSp2F}
}
```
