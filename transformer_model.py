import os
import math
import torch
import numpy
import msvcrt
import librosa
import soundfile
from torch import nn
from torch import optim
from torch._C import device
from torch.nn import functional
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import modules
import dataset


BATCH_SIZE = 16
SEQ_LEN = 128
RESAMPLE = 220

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("using device : " + str(device))
torch.cuda.empty_cache()


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        features = 512

        self.norm = nn.BatchNorm1d(SEQ_LEN)
        
        self.src_embed = nn.Embedding(65536, 512)
        self.tgt_embed = nn.Embedding(65538, 512)

        self.src_feature = nn.Conv1d(1, features, 1)
        self.tgt_feature = nn.Conv1d(1, features, 1)

        self.src_pos_embed = modules.PositionalEmbedding(features)
        self.tgt_pos_embed = modules.PositionalEmbedding(features)

        self.transformer = nn.Transformer(d_model=features, nhead=4, num_encoder_layers=4, num_decoder_layers=4, batch_first=True)
        self.linear = nn.Linear(features, 256)

    def forward(self, x, y):
        #x = self.norm(x)

        x = self.src_feature(x.unsqueeze(1))
        y = self.tgt_feature(y.unsqueeze(1))

        x = x.transpose(1, 2)
        y = y.transpose(1, 2)

        x = self.src_pos_embed(x.float())
        y = self.tgt_pos_embed(y.float())

        out = self.transformer(x, y)
        out = torch.mean(out, dim=1)

        return self.linear(out)


def train(epochs:int, warmup:int):
    net.train()

    loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0.

        for idx, (source, _) in enumerate(train_loader):
            if msvcrt.kbhit():
                try:
                    print(msvcrt.getch().decode(), "pressed, stopping training...")
                except UnicodeDecodeError:
                    print("unknown key pressed, stopping training...")
                break

            batch_loss = 0.
            net.zero_grad() #reset gradient between batches

            #teacher forcing
            for i in range(SEQ_LEN - 2):
                target = source[:, :i+1].to(device)
                #expected = source[:, 1:i+2].to(device)
                expected = modules.a_law_encode(source[:, i+1]).to(device)

                output = net(source.to(device), target)

                loss = functional.cross_entropy(output, expected)
                loss.backward() #backpropagate loss func
                optimizer.step()

                batch_loss += float(loss.item())

            #output = net(source, target)
            #loss = functional.mse_loss(output, expected)
            #loss.backward() #backpropagate loss func
            #optimizer.step()

            print("BATCH: "+str(idx)+", LR: "+str(optimizer.param_groups[0]["lr"])+", total loss: "+str(batch_loss))

            if (idx < warmup):
                warmup_sched.step()
            elif (idx > 100):
                break
            else:
                scheduler.step()

            epoch_loss += batch_loss
            loss_list.append(float(batch_loss))
        
        print("EPOCH "+str(epoch)+" : total loss "+str(epoch_loss))
        #loss_list.append(float(epoch_loss))

    torch.save({"state": net.state_dict(), "optim": optimizer.state_dict()}, r".\model-tfrm.pt")

    plt.plot(loss_list)
    plt.show()

def test(wav, pred_len):
    saved = torch.load(r".\model-tfrm.pt")
    net.load_state_dict(saved["state"])
    optimizer.load_state_dict(saved["optim"])
    
    net.eval()
    print("="*25)

    audio,_ = librosa.load(wav, sr=RESAMPLE)
    source = torch.tensor(audio[-SEQ_LEN:], device=device).unsqueeze(0)

    with torch.no_grad():
        for pred in range(math.ceil(pred_len * RESAMPLE)):
            output = net(source.unsqueeze(0), (h.to(device),c.to(device)))
            output = torch.softmax(output.flatten(start_dim=0), dim=0).cpu().numpy()

            chosen = modules.a_law_decode(numpy.random.choice(numpy.arange(256), size=1, p=output))

            audio = numpy.append(audio, chosen.item())
            
            source = torch.cat((source[1:], chosen.to(device)), dim=0)
            print(str(round(100*pred/pred_len, 2)) + "%", end="\r")

    soundfile.write(r".\output.wav", audio, RESAMPLE, subtype="PCM_24")
    print("Audio file produced : output.wav", audio.shape)


audio_dataset = dataset.AudioData(SEQ_LEN, RESAMPLE, BATCH_SIZE)
train_loader = DataLoader(audio_dataset, shuffle=True, batch_size=BATCH_SIZE)

net = Net()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.00001)

warmup_sched = optim.lr_scheduler.StepLR(optimizer, 1, 1.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

#train(1, warmup=10)
#test(os.sep.join(["archive", "genres", "blues", "blues.00000.wav"]), RESAMPLE*1) #seconds
