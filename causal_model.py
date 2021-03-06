import os
import math
import torch
import numpy
import msvcrt
import librosa
import cProfile
import soundfile
from torch import nn
from torch import optim
from torch._C import device
from torch.nn import functional
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import modules
import dataset


BATCH_SIZE = 64
SEQ_LEN = 512*4
RESAMPLE = 220

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("using device : " + str(device))
torch.cuda.empty_cache()


class Net(nn.Module):
    def __init__(self):         #receptive range = blocks * (2 ^ dilations)
        super().__init__()
        self.num_blocks = 4
        self.dilations = 8
        self.channels = 8

        self.norm = nn.BatchNorm1d(2048);

        self.blocks = nn.ModuleList()

        self.conv = nn.Conv1d(1, self.channels, 1)

        for x in range(self.num_blocks):
            for y in range(self.dilations):
                block = nn.ModuleList()
                block.append(modules.CausalConv1d(self.channels, self.channels, 2, dilation=2**y))  #dilated conv
                block.append(nn.Conv1d(self.channels, 256, 1))                                      #skip
                block.append(nn.Conv1d(self.channels, self.channels, 1))                            #filter
                block.append(nn.Conv1d(self.channels, self.channels, 1))                            #gate
                block.append(nn.Conv1d(self.channels, self.channels, 1))                            #residual

                self.blocks.append(block)

        self.out_convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1)
        )

        self.fc1 = nn.Linear(SEQ_LEN, SEQ_LEN)
        self.fc2 = nn.Linear(SEQ_LEN, 256)

    def forward(self, x):
        x = self.norm(x)

        x = x.unsqueeze(1)

        x = self.conv(x)
        
        skip = 0
        for blk in self.blocks:
            res = x
            x = functional.relu(blk[0](x))  #dilated conv
            skip += blk[1](x)               #skip
            f = torch.tanh(blk[2](x))       #filter
            g = torch.sigmoid(blk[3](x))    #gate
            x = blk[4](f * g) + res         #residual

        x = skip[:, :, -1:]   #take last value
        x = self.out_convs(x)
        
        return x.flatten(1)


def train(epochs:int, warmup:int, load:bool):
    net.train()
    
    warmup_sched = StepLR(optimizer, 1, 1.5)
    scheduler = StepLR(optimizer, 1, 0.95)

    if load:
        saved = torch.load(r".\model-causal.pt")
        net.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optim"])

    loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0.

        for idx, (source, expected) in enumerate(train_loader):
            net.zero_grad() #reset gradient between batches

            if msvcrt.kbhit():
                try:
                    print(msvcrt.getch().decode(), "pressed, stopping training...")
                except UnicodeDecodeError:
                    print("unknown key pressed, stopping training...")
                break

            output = net(source.to(device))

            loss = functional.cross_entropy(output, expected.to(device))
            loss.backward() #backpropagate loss func
            optimizer.step()

            epoch_loss += float(loss.item())
            print("BATCH: "+str(idx)+", LR: "+str(optimizer.param_groups[0]["lr"])+", loss: "+str(loss.item()), end="\t\t\r")
            #loss_list.append(loss.item())
        
        print("EPOCH: "+str(epoch)+", LR: "+str(optimizer.param_groups[0]["lr"])+", total loss: "+str(epoch_loss))
        loss_list.append(float(epoch_loss))

        if (epoch < warmup):
            warmup_sched.step()
        else:
            scheduler.step()

    torch.save({"state": net.state_dict(), "optim": optimizer.state_dict()}, r".\model-causal.pt")

    plt.plot(loss_list)
    plt.show()

def test(wav, pred_len):
    saved = torch.load(r".\model-causal.pt")
    net.load_state_dict(saved["state"])
    optimizer.load_state_dict(saved["optim"])

    net.eval()
    print("="*25)

    audio,sr = librosa.load(wav, sr=RESAMPLE, duration=20)
    source = torch.tensor(audio[-SEQ_LEN:]).to(device)

    with torch.no_grad():
        for pred in range(math.ceil(pred_len * RESAMPLE)):
            output = net(source.unsqueeze(0)).flatten(start_dim=0)
            output = torch.softmax(output, dim=0).cpu().numpy()

            chosen = modules.a_law_decode(numpy.random.choice(numpy.arange(256), size=1, p=output))
            #print(chosen.item(), end=",")

            audio = numpy.append(audio, chosen.item())

            source = torch.cat((source[1:], chosen.to(device)), dim=0)
            print(str(round(100*pred/pred_len, 2)) + "%", end="\r")
            
    soundfile.write(r".\output.wav", audio, RESAMPLE, subtype="PCM_24")
    print("Audio file produced : output.wav", audio.shape)


audio_dataset = dataset.AudioData(SEQ_LEN, RESAMPLE, BATCH_SIZE)
train_loader = DataLoader(audio_dataset, shuffle=True, batch_size=BATCH_SIZE)

net = Net()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.05, weight_decay=0.0001)

#train(10, 0, False)
#test(os.sep.join(["archive", "genres", "blues", "blues.00000.wav"]), RESAMPLE*5) #seconds
