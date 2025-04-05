# MGCNA
 Submitted to journal IEEE Journal of Biomedical and Health Informatics 
## 1. Overview
The code for paper "Multi-view fusion based on graph convolutional network with attention mechanism for predicting miRNA related to drugss". The repository is organized as follows:

+ `data/` contains the data in the paper:
  * `miRNA-drug-matrix.xlsx` contains known miRNA-drug interaction;
  * `drug-gene-matrix.xlsx` contains known drug-gene interaction;
  * `miRNA-gene-matrix.xlsx` contains known miRNA-gene interaction;
  * `drug-smiles.xlsx` contains drug ID, name, smiles ;
  * `miRNA-sequences.xlsx` contains miRNA ID, name, sequences ;
  * `gene-name.xlsx` contains Gene Symbol.
    
+ `code/`
  * `multi view construction.py` is used to calculate miRNA and drug views;
  * `parms_setting.py`contains hyperparmeters;
  * `data_preprocess.py` contains the preprocess of data;
  * `layer.py` contains MGCNA's model layer;
  * `train.py` contains training and testing code;
  * main.py` runs MGCNA;
  
## 2. Dependencies
* numpy == 1.21.1
* torch == 2.0.0+cu118
* sklearn == 0.24.1
* torch-geometric == 2.3.0

## 3. Quick Start
Here we provide a example:

1. Download and upzip our data and code files
2. Run "multi view construction.py" to obtain miRNA and drug similairty graph 
3. Run "main.py"

## 4. Reminder
It is recommended that you save the training and test sets for each fold and then calculate the MDA-based miRNA and drug view. Then continue with subsequent calculations, which will speed up the calculation.

## 5. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
