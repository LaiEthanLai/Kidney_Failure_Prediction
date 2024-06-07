# Kidney Failure Prediction Using a DNN
> [!IMPORTANT]
> We thank patients willing to participate in this project and generously share their data. We genuinely respect their privacy, so we will not release our dataset or the weights of our DNN. 


This project involves Professor [Feng-Tsun Chien](https://sites.google.com/nycu.edu.tw/ftchien/home?authuser=0) from the Institute of Electronics and the Institute of Artificial Intelligence Innovation, National Yang Ming Chiao Tung University (NYCU), his two affiliated research assistants [Ethan Lai](https://github.com/LaiEthanLai) and [Sam Wang](https://github.com/SamWang0807), and the Department of Internal Medicine -- Nephrology at National Taiwan University Hospital. 

Our goal is to build a powerful artificial intelligence-assisted diagnosis system. Specifically, we train deep neural networks (DNNs) to predict whether a patient will experience kidney failure and, if so, when it will occur.

## What is in this repository?

In `utils`, we implement three imputation methods to mitigate the missing data problem. The missing data arises because patients require various medical tests. We may not be able to collect all the above-mentioned metrics from every patient.

The DNN we adopt is in `model/net.py`.

## Methods
### Data Preprocessing
We use various biometrics as inputs for prediction: 

B_CRE, B_K, B_NA, B_UN, Hemoglobin, MCHC, PLT, WBC, Albumin, B_P, B_UA, Calcium, Triglyceride, DL, UPCR. 

These are common metrics a doctor will refer to when diagonizing a patient.
While experiencing a severe data missing problem on those metrics, we applied 3 different kinds of imputation strategies to our dataset and compare their results. We also implemented average moving filter to attain data smoothing and reduce noise. 



### Model

10-layer MLP with input dimension 16 and output dimension X. X refers to the class number that we want to divide our result in. For example, we categorize patients based on whether they will develop kidney disease within two years. Those who will are placed in one category, while those who won't are placed in another. As such, we let X=2.



## Installation
To get started with our kidney project, clone this repository and install the required dependencies:
```shell
$git clone git@github.com:LaiEthanLai/Project_Kidney.git
$cd ./Project_Kidney
$pip install -r requirements.txt

```


## Training
```shell
$python train.py
```