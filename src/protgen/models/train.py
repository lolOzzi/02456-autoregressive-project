import torch
import torch.nn as nn
from mlp import MLP 
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader

device = torch.device("cpu")
num_epochs = 5
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_data = ProteinDataset("../../../data/train.csv")
print(train_data[0])

print(train_data[1])
print(train_data[2])
val_data = ProteinDataset("../../../data/val.csv")

train_loader = DataLoader(train_data, batch_size=1, shuffle=True) 
val_loader = DataLoader(val_data, batch_size=1, shuffle=True) 


for epoch in range(num_epochs):
    #For each epoch
    train_correct = 0
    for index, (x, x_shifted, y, length) in enumerate(train_loader):
        data, target = x.to(device), x_shifted.to(device)
        #Zero the gradients computed for enumeratech weight

        for i in range(2, length[0]):
            part_x = data[0][i-2:i+1].float()
            part_t = target[0][i]
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(part_x)
            #Compute the loss
            loss = criterion(output, part_t)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()
        
        #Compute how many were correctly classified
            predicted = output.argmax()
            train_correct += (target==predicted).sum().cpu().item()
    #Comput the test accuracy
    test_correct = 0
    for index, (x, x_shifted, y, length) in test_loader:
        data = data.to(device)
        for i in range(2, length[0]):
            part_x = data[0][i-2:i+1]
            part_t = target[0][i]
            with torch.no_grad():
                output = model(part_x)
            predicted = output.argmax(1).cpu()
            test_correct += (part_t==predicted).sum().item()
    train_acc = train_correct/len(trainset)
    test_acc = test_correct/len(testset)
    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100*test_acc, train=100*train_acc))
