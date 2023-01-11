## 0. Indications
This repo is based on https://github.com/hihihihiwsf/AST repo with correction of some errors.

## 1. Download data
electricity: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams 20112014  
traffic: https://archive.ics.uci.edu/ml/datasets/PEMS-SF  
solar data： https://www.nrel.gov/grid/solar-power-data.html  
wind: https://www.kaggle.com/sohier/30-years-of-european-wind-generation   
## 2. Preprocess data
To process the data, you must first type this 
` python preprocess_data.py `  
But you can skip this step and just use the reduced data I placed in the `/data/elect` directory

## 3. Train and evaluate the model
To run the training and see the behavior of the whole code before making any changes you should type :  
` python train_gan.py --dataset elect --model test `  
--dataset The name of the preprocessed data  
--model the position of the specific params.json file under the folder of "experiments".  

## 4. Illustration (Training-output)
Output should look like this...  
<video width="320" height="240" controls>
  <source src="others/GithubVID1.mp4" type="video/mp4">
</video>