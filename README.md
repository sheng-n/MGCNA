# Multi-view fusion based on graph convolutional network with attention mechanism for predicting miRNA related to drugs
 Submitted to journal IEEE Journal of Biomedical and Health Informatics 
## 1. Overview
The code for paper "Multi-view fusion based on graph convolutional network with attention mechanism for predicting miRNA related to drugss". The repository is organized as follows:

+ `data/` contains the dataset in the paper, with dataset as an example;
  * `miRNA-drug-matrix.xlsx` contains known miRNA-drug interaction;
  * `drug-gene-matrix.xlsx` contains known drug-gene interaction;
  * `miRNA-gene-matrix.xlsx` contains known miRNA-gene interaction;
  * `drug-smiles.xlsx` contains drug ID, name, smiles ;
  * `miRNA-sequences.xlsx` contains miRNA ID, name, sequences ;
  * `gene-name.xlsx` contains Gene Symbol.
+ `code/`
  * `data_preparation.py` is used to calculate lncRNA/miRNA k-mer features and construct knn graph (attribute graph) of lncRNA/miRNA/disease.
  * `calculating_similarity.py` is used to calclulate lncRNA/miRNA/disease GIPK similarities and obtain the intra-edges in the topology graph;
  * `parms_setting.py`contains hyperparmeters;
  * `utils.py` contains preprocessing function of the data;
  * `data_preprocess.py` contains the preprocess of data;
  * `layer.py` contains SSCLMD's model layer;
  * `train.py` contains training and testing code;
  * `main.py` runs SSCLMD;

## 2. Dependencies
* numpy == 1.21.1
* torch == 2.0.0+cu118
* sklearn == 0.24.1
* torch-geometric == 2.3.0

## 3. Quick Start
Here we provide a example to predict the lncRNA-disease association scores on dataset 1:

1. Download and upzip our data and code files
2. Run data_preparation.py and calculating_similarity.py to obtain lncRNA/miRNA/disease attribute graph and intra_edge of topology graph 
3. Run main.py (in file-- dataset1/LDA.edgelist, neg_sample-- dataset1/non_LDA.edgelist, task_type--LDAl)

## 4. Reminder
It is recommended that you save the training and test sets for each fold and then calculate the lncRNA/miRNA/disease functional similarity. Then continue with subsequent calculations, which will speed up the calculation.

## 5. Contacts
If you have any questions, please email Nan Sheng (shengnan@jlu.edu.cn)
