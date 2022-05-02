import os
import math
import torch
import numpy
import librosa
import soundfile
from torch import nn
from torch import optim
from torch._C import device
from torch.nn import functional
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import modules
import dataset

BATCH_SIZE = 32
SEQ_LEN = 512
RESAMPLE = 2205

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("using device : " + str(device))


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = 3
        self.lstm_nodes = 512

        self.feature = nn.Conv1d(1, 256, 1)
        self.pooling = nn.Conv1d(self.lstm_nodes, 1, 1)

        self.lstm = nn.LSTM(256, self.lstm_nodes, num_layers=self.layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(SEQ_LEN, 256)

    #complete forward pass
    def forward(self, x, old_state):
        flag = (False, 0)
        if x.size(0) < BATCH_SIZE:
            flag = (True, x.size(0))
            x = functional.pad(x, (0, 0, 0, BATCH_SIZE - x.size(0)))

        #extract 256 features
        x = self.feature(x.unsqueeze(1))

        x = torch.transpose(x, 1, 2)
        x, new_state = self.lstm(x, old_state)

        #x = torch.mean(x, dim=2)
        x = self.pooling(x).flatten(start_dim=1)

        x = self.fc(x)

        if flag[0]:
            x = x[:flag[1]]

        return x, new_state

    def init_state(self, batch_len):
        return (
            torch.zeros(self.layers, batch_len, self.lstm_nodes),
            torch.zeros(self.layers, batch_len, self.lstm_nodes)
        )


def train(epochs):
    net.train()
    print("LR : " + str(optimizer.param_groups[0]["lr"]))

    loss_list = []
    for epoch in range(epochs):
        total_loss = 0.
        h,c = net.init_state(BATCH_SIZE)
        
        for idx, (source, expected) in enumerate(train_loader):
            net.zero_grad() #reset gradient between batches

            output, (hn,cn) = net(source.to(device), (h.to(device),c.to(device)))
            loss = functional.cross_entropy(output, expected.to(device))

            h = hn.detach()
            c = cn.detach()
            
            loss.backward() #backpropagate loss func
            optimizer.step()

            total_loss += float(loss.item())
            print(idx, end="\r")

            if idx % 5 == 0:
                loss_list.append(float(loss.item()))
            if idx == 5000:
                break
        
        print("EPOCH "+str(epoch)+" : total loss "+str(total_loss))

    torch.save({"state": net.state_dict(), "optim": optimizer.state_dict()}, r".\model.pt")
    plt.plot(loss_list)
    plt.show()


def test(wav, pred_len):
    saved = torch.load(r".\model.pt")
    net.load_state_dict(saved["state"])
    optimizer.load_state_dict(saved["optim"])

    net.eval()
    print("="*25)

    audio,_ = librosa.load(wav, sr=RESAMPLE)
    source = torch.tensor(audio[-SEQ_LEN:]).to(device)
    h,c = net.init_state(32)

    with torch.no_grad():
        for pred in range(math.ceil(pred_len * RESAMPLE)):
            output, (hn,cn) = net(source.unsqueeze(0), (h.to(device),c.to(device)))
            output = torch.softmax(output.flatten(start_dim=0), dim=0).cpu().numpy()

            chosen = modules.a_law_decode(numpy.random.choice(numpy.arange(256), size=1, p=output))

            h = hn.detach()
            c = cn.detach()

            audio = numpy.append(audio, chosen.item())

            source = torch.cat((source[1:], chosen.to(device)), dim=0)
            print(str(round(pred/pred_len, 2)) + "%", end="\r")

    soundfile.write(r".\output.wav", audio, RESAMPLE, subtype="PCM_24")
    print("Audio file produced : output.wav")


audio_dataset = dataset.AudioData(SEQ_LEN, RESAMPLE, BATCH_SIZE)
train_loader = DataLoader(audio_dataset, shuffle=True, batch_size=BATCH_SIZE)

net = Net()

torch.cuda.empty_cache()
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.00001) #0.001, 0.00001

#saved = torch.load(r".\model.pt")
#net.load_state_dict(saved["state"])
#optimizer.load_state_dict(saved["optim"])


# SET TASK
#train(1)
#test(os.sep.join(["archive", "genres", "blues", "blues.00000.wav"]), 1)
