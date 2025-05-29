
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from torch.nn import Linear, Module, Dropout
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.stats
import shap
import os

class CSVDataset(Dataset):
    ## Store the inputs and outputs in the tensor format
    def __init__(self, data):
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
    ## Get the row numbers of the dataset    
    def __len__(self):
        return len(self.X)
    ## Get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

 
class MLP(Module):
    ## Define the structure of MLP
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs,50)
        self.dropout = Dropout(0)
        self.hidden2 = Linear(50,1)
 
    ## Forward propagation
    def forward(self, X):
        X = self.dropout(X)
        X = self.hidden1(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.hidden2(X)
        return X

## Read data 
## Assume that each data is structured as below:
## Rows: n RILs(n rows in total) 
## columns: the first column for id, m columns for markers and the last column for phenotype(m+2 columns in total)
data_train = pd.read_csv('../data/example_train.csv')
data_test = pd.read_csv('../data/example_test.csv')

train_id, test_id = data_train.iloc[:,0], data_test.iloc[:,0]
data_train, data_test = data_train.iloc[:,1:], data_test.iloc[:,1:]

## Convert the data into the specified format
train = CSVDataset(data_train)
test = CSVDataset(data_test)

## Create batches                
train_loader = DataLoader(train, batch_size=8, shuffle=True)
test_loader = DataLoader(test, batch_size=8, shuffle=False)

## Develop MLP
model = MLP(data_train.shape[1]-2)

## Define the optimiser
optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=0.005, weight_decay=5e-4) #0.0005

## Train the model
for epoch in range(1000):
     loss_train_sum = 0
     for inputs, targets in train_loader:
         optimizer.zero_grad()
         yhat = model(inputs)
         loss = F.mse_loss(yhat, targets)
         loss_train_sum += loss
         loss.backward()
         optimizer.step()
     print(f'Epoch {epoch:>3} | Train Loss: {loss_train_sum/len(train_loader):.5f}')

## Extract predicted and observed values for the test set
predicted_test = []
actual_test = []
for inputs, targets in test_loader:
     yhat = model(inputs)
     yhat = yhat.detach().tolist()
     actual = targets.detach().tolist()
     predicted_test.append([item for sublist in yhat for item in sublist])
     actual_test.append([item for sublist in actual for item in sublist])

predicted = [item for sublist in predicted_test for item in sublist]
actuals = [item for sublist in actual_test for item in sublist]
 
## Calculate the metrics
mse = mean_squared_error(actuals, predicted)
r = scipy.stats.pearsonr(actuals, predicted)[0]

## Store the metrics
record = pd.DataFrame([r,mse]).T
record.columns = ['Pearson_r','MSE']

## Store prediction result for the test data
result_prediction_test = pd.concat([pd.DataFrame(test_id), pd.DataFrame(predicted),pd.DataFrame(actual_test)],axis=1)
result_prediction_test.columns = ['id','predicted','actual']  

## Extract predicted and observed values for the train set
train_loader = DataLoader(train, batch_size=8, shuffle=False)
predicted_train = []
actual_train = []
for inputs, targets in train_loader:
     yhat = model(inputs)
     yhat = yhat.detach().tolist()
     actual = targets.detach().tolist()
     predicted_train.append([item for sublist in yhat for item in sublist])
     actual_train.append([item for sublist in actual for item in sublist])

predicted = [item for sublist in predicted_train for item in sublist]
actual_train = [item for sublist in actual_train for item in sublist]       
         
## Store prediction result for the train data
result_prediction_train = pd.concat([pd.DataFrame(train_id), pd.DataFrame(predicted_train),pd.DataFrame(actual_train)],axis=1)
result_prediction_train.columns = ['id','predicted','actual']   

## Extract marker effect
d_train = torch.tensor(data_train.iloc[:,:-1].values, dtype=torch.float32)
d_test = torch.tensor(data_test.iloc[:,:-1].values, dtype=torch.float32)

explainer = shap.DeepExplainer(model,shap.sample(d_train, 50))
effect = abs(explainer.shap_values(shap.sample(d_test, 50),check_additivity=False)).sum(axis=0)
effect = pd.DataFrame(effect).T
effect.columns = list(data_train.columns)[1:-1]

## Save all results
record.to_csv('../output/Metric_MLP.csv')
effect.to_csv('../output/Marker_effect_MLP.csv')
result_prediction_train.to_csv('../output/Prediction_result_train_MLP.csv')
result_prediction_test.to_csv('../output/Prediction_result_test_MLP.csv')