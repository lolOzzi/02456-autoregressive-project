import torch
import torch.nn as nn
from mlp import MLP 
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader

device = torch.device("cuda")
num_epochs = 20
model = MLP().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_data = ProteinDataset('data/train.csv')
val_data = ProteinDataset("data/val.csv")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True) 
val_loader = DataLoader(val_data, batch_size=32, shuffle=True) 

WINDOW_SIZE = 3

def make_windows(x, x_shifted):
    x_small_sequences = x.unfold(1, WINDOW_SIZE, 1).float()
    tgt = x_shifted[:, WINDOW_SIZE - 1:]

    inputs  = x_small_sequences.reshape(-1, WINDOW_SIZE).float() # (N, 3)
    targets = tgt.reshape(-1).long()               # (N,)
    return inputs, targets


for epoch in range(num_epochs):
    #For each epoch
    train_correct, train_num = 0, 0
    for index, (x, x_shifted, y, length) in enumerate(train_loader):
        data, target = x.to(device), x_shifted.to(device)
        #Zero the gradients computed for enumeratech weight

        x_small_sequences, tgt = make_windows(data, target)
        
        optimizer.zero_grad()
        #Forward pass your image through the network
        output = model(x_small_sequences)
        #Compute the loss
        loss = criterion(output, tgt)
        #Backward pass through the network
        loss.backward()
        #Update the weights
        optimizer.step()
        
        #Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct += (tgt==predicted).sum().cpu().item()
        
        valid = tgt != -1
        train_num += valid.sum().item()

    #Comput the test accuracy
    test_correct, test_num = 0, 0
    for index, (x, x_shifted, y, length) in enumerate(val_loader):
        data = x.to(device)
        target = x_shifted.to(device)

        x_small_sequences, tgt = make_windows(data, target)
        with torch.no_grad():
            output = model(x_small_sequences)
            predicted = output.argmax(1)
            test_correct += (tgt==predicted).sum().item()
            valid = tgt != -1
            test_num += valid.sum().item()

    train_acc = train_correct/train_num
    test_acc = test_correct/test_num
    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))