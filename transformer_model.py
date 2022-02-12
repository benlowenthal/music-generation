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
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

import modules

BATCH_SIZE = 16
SEQ_LEN = 128
RESAMPLE = 2205

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("using device : " + str(device))
torch.cuda.empty_cache()


class AudioData(Dataset):
    def __init__(self):
        self.data = []
        self.seq_len = SEQ_LEN
        self.min_signal_len = 10**10

        #for dir in tqdm.tqdm(os.listdir(os.sep.join([r".\archive", "genres"]))):
        #    for wav in os.listdir(os.sep.join([r".\archive", "genres", dir])):
        #        #audio loading w/o resample
        #        signal,_ = librosa.load(os.sep.join([r".\archive", "genres", dir, wav]), sr=RESAMPLE)
        #        self.data.append(signal)
        #
        #        if len(signal) < self.min_signal_len:
        #            self.min_signal_len = len(signal)

        # only blues
        #for wav in os.listdir(os.sep.join([r".\archive", "genres", "blues"])):
        #    signal,_ = librosa.load(os.sep.join([r".\archive", "genres", "blues", wav]), sr=RESAMPLE)
        #    self.data.append(signal)

        #    if len(signal) < self.min_signal_len:
        #        self.min_signal_len = len(signal)

        self.data.append(librosa.load(os.sep.join([r".\archive", "genres", "blues", "blues.00000.wav"]), sr=RESAMPLE)[0])
        self.min_signal_len = len(self.data[0])

        print("data points:", len(self.data)*(self.min_signal_len-self.seq_len))
        print("possible batches:", len(self.data)*(self.min_signal_len-self.seq_len) / BATCH_SIZE)

    def __len__(self):
        return len(self.data)*(self.min_signal_len-self.seq_len)

    def __getitem__(self, idx):
        track = idx // (self.min_signal_len-self.seq_len)
        pos = idx % (self.min_signal_len-self.seq_len)
        return torch.tensor(self.data[track][pos:pos+self.seq_len], device=device)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm = nn.BatchNorm2d(1)
        
        self.src_embed = nn.Embedding(65536, 512)
        self.tgt_embed = nn.Embedding(65538, 512)

        self.src_pos_embed = modules.PositionalEmbedding(512)
        self.tgt_pos_embed = modules.PositionalEmbedding(512)

        self.transformer = nn.Transformer(d_model=512, nhead=4, num_encoder_layers=4, num_decoder_layers=4, batch_first=True)
        self.linear = nn.Linear(512, 1)

    def forward(self, x, y):
        #x = self.norm(x)

        #scale from -1,1 to 0,65535
        x = self.src_embed((x * 32767.5 + 32767.5).int())
        y = self.tgt_embed((y * 32767.5 + 32767.5).int())

        x = self.src_pos_embed(x.float())
        y = self.tgt_pos_embed(y.float())

        out = self.transformer(x, y)

        return torch.tanh(self.linear(out).flatten(start_dim=1)) #values between -1 and 1


def train(epochs:int, warmup:int, load:bool):
    net.train()
    
    if load:
        saved = torch.load(r".\model-tfrm.pt")
        net.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optim"])

    loss_list = []
    for epoch in range(epochs):
        epoch_loss = 0.

        for idx, batch in enumerate(train_loader):
            if msvcrt.kbhit():
                try:
                    print(msvcrt.getch().decode(), "pressed, stopping training...")
                except UnicodeDecodeError:
                    print("unknown key pressed, stopping training...")
                break

            batch_loss = 0.
            net.zero_grad() #reset gradient between batches

            source = batch
            batch = torch.cat((torch.full((BATCH_SIZE, 1), (2**15+2) / (2**15), device=device), batch), 1)
            batch = torch.cat((batch, torch.full((BATCH_SIZE, 1), (2**15+1) / (2**15), device=device)), 1)

            #teacher forcing
            for i in range(batch.shape[1] - 2):
                target = batch[:, :i+1]
                expected = batch[:, 1:i+2]

                output = net(source, target)

                loss = functional.mse_loss(output, expected)
                loss.backward() #backpropagate loss func
                optimizer.step()

                batch_loss += float(loss.item())

            print("LR: "+str(optimizer.param_groups[0]["lr"])+", BATCH: "+str(idx)+", total loss: "+str(batch_loss))

            if (batch < warmup):
                warmup_sched.step()
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
    net.eval()
    print("="*25)

    audio,_ = librosa.load(wav, sr=RESAMPLE)
    source = torch.tensor(audio[-SEQ_LEN:], device=device).unsqueeze(0)
    start_token = torch.tensor((2**15+2) / (2**15), device=device).unsqueeze(0).unsqueeze(0) #target starts as start token
    target = start_token

    with torch.no_grad():
        for pred in range(pred_len):
            output = net(source, target)
            
            if pred < SEQ_LEN-1:
                target = torch.cat((start_token, output), 1)
            else:
                audio = numpy.append(audio, output[0][0].cpu())
                target = output

    soundfile.write(r".\output.wav", audio, RESAMPLE, subtype="PCM_24")
    print("Audio file produced : output.wav", audio.shape)


audio_dataset = AudioData()
train_loader = DataLoader(audio_dataset, shuffle=True, batch_size=BATCH_SIZE)

net = Net()
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.00001)

warmup_sched = optim.lr_scheduler.StepLR(optimizer, 1, 1.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

train(1, warmup=10, load=False)
test(os.sep.join(["archive", "genres", "blues", "blues.00000.wav"]), RESAMPLE*1) #seconds
