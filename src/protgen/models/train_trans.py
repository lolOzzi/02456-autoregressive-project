import torch
import torch.nn as nn
from protgen.models.transformer import Transformer 
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader

device = torch.device("cuda")
num_epochs = 100

model = Transformer(src_vocab_size=23, tgt_vocab_size=23, d_model=256, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=256, dropout=0.1).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_data = ProteinDataset('data/train.csv')
val_data = ProteinDataset("data/val.csv")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True) 
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    #For each epoch
    model.train()
    train_correct, train_num = 0, 0
    for index, (x, x_shifted, y, length) in enumerate(train_loader):
        data, target = x.to(device), x_shifted.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output.reshape(-1, 23), target.reshape(-1))

        loss.backward()
        #Update the weights
        optimizer.step()
        
        train_correct += loss.item()
        train_num += 1

    model.eval()
    val_correct, val_total, val_loss_sum, val_batches = 0, 0, 0.0, 0
    with torch.no_grad():
        for x, x_shifted, y, length in val_loader:
            data, target = x.to(device), x_shifted.to(device)
            output = model(data)                                  # (batch, T, 23)

            val_loss_sum += criterion(output.reshape(-1, 23), target.reshape(-1)).item()
            val_batches += 1

            pred = output.argmax(dim=-1)                          # (batch, T)
            valid = target != -1                                  # ignore padding
            val_correct += (pred[valid] == target[valid]).sum().item()
            val_total += valid.sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"epoch {epoch} | train loss {train_correct / train_num:.4f} " f"| val loss {val_loss_sum / val_batches:.4f} | val acc {val_acc:.2f}%")