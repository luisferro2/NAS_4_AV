# NAS_4_AV
Project to use Neural Architecture Search (NAS) coupled with an Evolutionary Algorithm and a deep neural network search space to solve the Authorship Verification (AV) task.

To explore the proposed deep neural network search space when solving AV, follow the steps outlined (CUDA enabled machine required):

## 1.- Create environment
You may prefer to use a virtual environment manager like venv or conda

Example with conda:
```
conda create --name my_env
```

## 2.- Activate the environment
Example with conda:
```
conda activate my_env
```

## 3.- Clone the repository

```
git clone https://github.com/luisferro2/NAS_4_AV.git
```

## 4.- Install requirements

Navigate to the folder of the repository.
```
cd NAS_4_AV
```
Install Python 3.12 (example with conda):
```
conda install python==3.12
```
Install list of requirements.
```
pip install -r requirements.txt
```

## 5.- Install cuda and cuda-enabled PyTorch

The steps to install cuda on Windows are in the documentation:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

The command to install cuda-enabled PyTorch can be obtained with your particular system and cuda version in the following link:
https://pytorch.org/get-started/locally/

## 4.- Generate dataset in format

This project is based around the problem configuration of one known document with one unknown document. To use your own dataset, transform your training and test data into the following table format:

![alt text](<Screenshot 2025-05-06 at 3.10.54 p.m..png>)

The column _'text1'_ represents the known document, column _'text2'_ represents the unknown document, and the _'label'_ (0 or 1) represents the answer to the authorship verification problem. Then, save these tables as a csv file with the custom separator backslash '\\' in the folder Dataset, with the names _'training.csv'_ and _'test.csv'_.

We use the open access datasets from CLEF-PAN, which are of the years 2013, 2014, 2015 and 2020(21) in a standard format. To download the original datasets, visit: https://pan.webis.de/shared-tasks.html

## 5.- Execute the notebook

Open notebook _NAS_AV.py_ and execute step by step to see the process of applying NAS to solve AV.

## Other files/folders

- The _surrogates_ folder is meant to serve as a place where some results will be saved after running the _NAS_AV.ipynb_ notebook

- The _preprocessing.ipynb_ is provided to understand the code that was used to generate the datasets in format. It was executed in Google Colab with a Google Drive mount.

- The _experimentationBaselineApproach.ipynb_ is provided to understand the code that was used to execute the baseline approach of NAS to solve AV. It was executed in Google Colab with a Google Drive mount.

- The OurResults folder contains the raw results obtained by us after performing the experimentation from the baseline approach, and also after performing the expermentation described in notebook _NAS_AV.ipynb_, It also contains the code that was used to generate visualizations and statistical tests from the results. 
