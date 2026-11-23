import torch
import torch.nn as nn
from src.protgen.models.transformer import Transformer

device = torch.device("cuda")
model = Transformer(src_vocab_size=23, tgt_vocab_size=23, d_model=256, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=256, dropout=0.1).to(device)

trained = torch.load('data/trans_model.pt', weights_only=False, map_location=torch.device('cuda'))
model.load_state_dict(trained)
model.eval()




def generateProtein(max_len = 128):
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        protein = torch.tensor([[1]]).to(device)
        for i in range(max_len):
            output = model(protein)

            logits = output[:, -1, :]
            probabilities = softmax(logits)

            greedy = probabilities.argmax(dim=-1)
            sampled = torch.multinomial(probabilities, 1)

            #print(probabilities, probabilities.sum())
           # print(greedy, sampled)


            protein = torch.cat([protein, sampled], dim=1)
            if sampled.item() == 2:
                break
   # print(protein)
    return protein

def convertToSequence(protein):
    mapstring = "ACDEFGHIKLMNPQRSTVWY"
    letters = [mapstring[i - 3] for i in protein.flatten().tolist() if i > 2]
    sequence = "".join(letters)
    return sequence

numberOfProteins = 20
for i in range(numberOfProteins):
    protein = generateProtein()
    sequence = convertToSequence(protein)
    print(sequence)



#print(model)

