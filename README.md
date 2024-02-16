
#  readme file for MpLessionSegNet


# Description of directory structure

-algorithm: Store different algorithm models and related training and testing scripts (separate folder for each algorithm model)
- Algorithm1
-network: This is where the model definition code goes
-train: This holds training code
-test: This holds inference/test code
-dataset: This is where the relevant implementations of the Dataset are stored
-loss: This holds some custom Loss implementations
- optimizer: This holds some custom optimizers and lr scheduler implementations
-scripts: This holds common scripts such as dataset transformations, unified test scripts, and so on
-utils: This will be used for common utilities like metrics calculation, post-processing, etc
- assets: This will hold some illustrations used in the documentation

# Code requirements

- import paths always use the project root
- Irrelevant files are specified in.gitignore
- If the algorithm has a specific dependency, add a requirements.txt or README.md to the corresponding algorithm directory explaining how the dependency should be configured

# Install dependencies

First install torch 1.13.1, refer to the torch website

` ` `
pip install -r requirements.txt
pip install 'monai[all]'
pip install monai==1.2.0
```