import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics
import torch
import torchvision
from torch.autograd import Variable

torch.manual_seed(1)

# functions/classes taken from: https://gist.github.com/t-vi/9f6118ff84867e89f3348707c7a1271f
# to help create validation set
class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds)>=offset+length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        return self.parent_ds[i+self.offset]

def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).
    
       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds
       
       """
    val_offset = int(len(dataset)*(1-val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset)-val_offset)


# load in MNIST
train_validation = torchvision.datasets.MNIST(root = os.getcwd(), train=True, transform=torchvision.transforms.ToTensor(), download = True)
train, validation = validation_split(train_validation, 1/6) # get train/validation split
test = torchvision.datasets.MNIST(root = os.getcwd(), train=False, transform=torchvision.transforms.ToTensor(), download = True)

# create train, validation, and test batches
batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size, shuffle = False)


# Deep NN architecture
# 12 layers, ReLU activations functions
class Deep_FFNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Deep_FFNN, self).__init__()
        
        self.linear1 = torch.nn.Linear(in_features = input_size, out_features = hidden_size, bias = True).cuda()
        self.relu1 = torch.nn.ReLU().cuda()
        self.linear2 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu2 = torch.nn.ReLU().cuda()
        self.linear3 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu3 = torch.nn.ReLU().cuda()
        self.linear4 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu4 = torch.nn.ReLU().cuda()
        self.linear5 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu5 = torch.nn.ReLU().cuda()
        self.linear6 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu6 = torch.nn.ReLU().cuda()
        self.linear7 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu7 = torch.nn.ReLU().cuda()
        self.linear8 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu8 = torch.nn.ReLU().cuda()
        self.linear9 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu9 = torch.nn.ReLU().cuda()
        self.linear10 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu10 = torch.nn.ReLU().cuda()
        self.linear11 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu11 = torch.nn.ReLU().cuda()
        self.linear12 = torch.nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True).cuda()
        self.relu12 = torch.nn.ReLU().cuda()
        self.linearOut = torch.nn.Linear(in_features = hidden_size, out_features = 10, bias = True).cuda()
        
    def forward(self, input):
        hidden = self.linear1(input)
        hidden = self.relu1(hidden)
        hidden = self.linear2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.linear3(hidden)
        hidden = self.relu3(hidden)
        hidden = self.linear4(hidden)
        hidden = self.relu4(hidden)
        hidden = self.linear5(hidden)
        hidden = self.relu5(hidden)
        hidden = self.linear6(hidden)
        hidden = self.relu6(hidden)
        hidden = self.linear7(hidden)
        hidden = self.relu7(hidden)
        hidden = self.linear8(hidden)
        hidden = self.relu8(hidden)
        hidden = self.linear9(hidden)
        hidden = self.relu9(hidden)
        hidden = self.linear10(hidden)
        hidden = self.relu10(hidden)
        hidden = self.linear11(hidden)
        hidden = self.relu11(hidden)
        hidden = self.linear12(hidden)
        hidden = self.relu12(hidden)
        output = self.linearOut(hidden)

        return output
    
    
# function for computing accuracy
def compute_accuracy(data_loader, model, input_size):
    model_pred = []
    targets = []
    for batch in data_loader:
        model_output = model(Variable(torch.squeeze(batch[0], 1).view(len(batch[0]), input_size)).cuda()) # output from NN
        model_probs = softmax(model_output).cpu().data.numpy() # digit probabilities
        model_pred += np.argmax(model_probs, axis = 1).tolist() # digit predictions
        targets += batch[1].numpy().tolist() # true digit values
        
    return sklearn.metrics.accuracy_score(targets, model_pred)


# hyperparameters to try
lam = 0.000001
lrs = [0.1, 0.01]
etas = [0.00001, 0.0001, 0.001, 0.01]
gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
num_epochs = 70

input_size = 28*28 # mnist image sizes
hidden_size = 50

softmax = torch.nn.Softmax() # softmax to compute output probabilities 
loss_function = torch.nn.CrossEntropyLoss() # cross entropy loss function


# loop through hyperparams
best_val_acc = 0
for lr in lrs:
    for gamma in gammas:
        for eta in etas:

            # instantiate model
            deep_FFNN = Deep_FFNN(input_size, hidden_size)

            # SGD optimizer
            optimizer = torch.optim.SGD(deep_FFNN.parameters(), lr, weight_decay = lam)

            # for lessening noise over time
            t = 0
            for epoch in range(num_epochs):   
                for batch in train_loader:
                    model_output = deep_FFNN(Variable(torch.squeeze(batch[0], 1).view(len(batch[0]), input_size)).cuda()) # model predictions
                    targets = Variable(batch[1]).cuda() # true digit values

                    optimizer.zero_grad() # zero gradient
                    loss_batch = loss_function(model_output, targets) # compute loss
                    loss_batch.backward() # take the gradient wrt parameters
                        
                    sigma_t = np.sqrt(eta/((1 + t)**gamma)) # sigma for noise
                    noise = torch.normal(means = torch.zeros(1), std = torch.ones(1) * sigma_t).numpy()[0].astype(np.float64) # get noise value

                    for param in list(deep_FFNN.parameters()):
                        param.grad += noise # add noise to gradient with respect to parameter
                            
                    optimizer.step() # update parameters
                    t += 1
                                    
            # see validation results
            val_acc = compute_accuracy(validation_loader, deep_FFNN, input_size)
            print('Validation accuracy for gamma=' + str(gamma) + ', learning rate=' + str(lr) 
                   + ', eta=' + str(eta) + ': ' + str(val_acc))
                
            # save best model and best model hyperparams
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_gamma = gamma
                best_eta = eta
                best_model = deep_FFNN

# see test results
test_acc = compute_accuracy(test_loader, best_model, input_size)
print('Test accuracy for gamma=' + str(best_gamma) + ', learning rate=' + str(best_lr) 
      + ', eta=' + str(best_eta) + ': ' + str(test_acc))