# CarcGC
=====        
### Carcinogenicity prediction model based on interpretable graph convolutional neural network       
<br>     

CarcGC is a graph convolutional network for chemical carcinogenicity prediction. It takes chemical structure as inputs and predicts the chemical carcinogenicity (binary classification).

<!-- TOC START min:1 max:3 link:true asterisk:false update:true -->
- [Requirements](#requirements)
- [User Guide](#user-guide)
  - [Step 1](#step-1)
  - [Step 2](#step-2)
  - [Step 3.](#step-3)
- [Contact Us](#contact-us)
<!-- TOC END -->

# Requirements
**We generated the relevant packages that the project relies on as Requirements.txt files**
These packages can be installed directly in batches using CONDA:
    `conda install --yes --file requirements.txt`

    deepchem=2.6.0.dev20210509222234
    hickle=4.0.4
    Keras=2.4.3
    matplotlib=3.4.0
    networkx=2.5
    numpy=1.19.5
    pandas=1.2.3
    rdkit=2017.09.1
    scikit-learn=0.24.1
    scipy=1.6.2
    seaborn=0.11.1
    sklearn=0.0
    tensorboard=2.4.1
    tensorboard-plugin-wit=1.8.0
    tensorflow-addons=0.13.0
    tensorflow-determinism=0.3.0
    tensorflow-estimator=2.4.0
    tensorflow-gpu=2.4.1
    tensorflow-gpu-estimator=2.3.0

  - It is recommended to execute this project in a linux environment, such as Anaconda3

# User Guide
We provide detailed step-by-step instructions for running CarcGC model including data preprocessing, model training, and model test.
## Step 1
**Drug feature representation**

In this project, each chemical in our study will be represented as a graph containing nodes and edges. We collected 1003 chemicals and 40 chemicals from CPDB database and ISSCAN database that have SMILES. Then we put the SMILES and ID of chemicals into smiles (for example: ./Dataset/chemicals.smiles) and run `process_drug.py` script to extract three types of features by [deepchem](https://github.com/deepchem/deepchem) library. The node feature (48 dimension) corresponds to an atom in within a chemical, with includes atom type, degree and hybridization, etc. The adjacent features denote the all the neighboring atoms, and the degree features denote the number of neighboring atoms. The above feature list will be further compressed as ID.hkl using hickle library, and placed in drug_graph_feat (for example: ./CarcGC/drug_graph_feat) folder. 

Please note that we provided the exacted features of 1043 chemicals from CPDB and ISSCAN database, just unzip the drug_graph_feat.zip file in CarcGC/drug_graph_feat folder. 


## Step 2
**CarcGC model selection, training and testing**

This project provides a model hyperparameter screening module `CarcGC_gridsearch.py` and a predictor training module `CarcGC_predict.py`. One can run python `CarcGC_gridsearch.py` to implement the model hyperparameter screening. we use grid search to select different learning rates, batch sizes, dropout and L2 regularization term, train models on various hyperparameter combinations, and compare the predictive performance of each model on the validation set. The evaluation metrics of different models on the validation set are put into csv (for example: ./CarcGC/gridsearch_result.csv) file. 

And for model training and test case, after setting the optimal hyperparameters, one can run python `CarcGC_predict.py` to implement the CarcGC classification model. The trained model will be saved in h5 (for example: ./CarcGC/bestmodel/BestCarcGC_Cartoxicity_highestAUCROC _256_256_256_bn_relu_GMP.h5) file. The overall auROC and auPRC of validation set and external test set will be calculated. And the predicted result of each chemical in external test set will be placed in csv (for example: ./CarcGC/predict_result/ lr5e-05_batch32_dropout0.1_0.001l2_extra_res.csv) file.


## Step 3
**Predicted the structural alerts**

One can run python `CarcGC_predict_fragment.py` to calculate the chemical carcinogenicity probability prediction value of each atom of each carcinogen. And run python GetFragmentWeight.py to map atomic carcinogenicity contribution and identify structural alerts.

# Contact Us

Email: hzzhang@rcees.ac.cn

# License
This project is licensed under the MIT License - see the LICENSE.md file for details
