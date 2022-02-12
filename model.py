import os
import torch
import numpy
import librosa
import soundfile
from torch import nn
from torch import optim
from torch._C import device
from torch.nn import functional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

BATCH_SIZE = 32

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print("using device : " + str(device))

class AudioData(Dataset):
    def __init__(self, path):
        self.data = []
        self.sr = 22050
        self.seq_len = 100000 #samples
        self.output_len = 50
        self.min_signal_len = 10**10

        #for dir in tqdm.tqdm(os.listdir(os.sep.join([path, "genres"]))):
        #    for wav in os.listdir(os.sep.join([path, "genres", dir])):
        #        #audio loading w/o resample
        #        signal,_ = librosa.load(os.sep.join([path, "genres", dir, wav]), sr=None)
        #        self.data.append(signal)

        #        if len(signal) < self.min_signal_len:
        #            self.min_signal_len = len(signal)

        #only blues
        for wav in os.listdir(os.sep.join([path, "genres", "blues"])):
            #audio loading w/o resample
            signal,_ = librosa.load(os.sep.join([path, "genres", "blues", wav]), sr=None)
            self.data.append(signal)

            if len(signal) < self.min_signal_len:
                self.min_signal_len = len(signal)

    def __len__(self):
        return len(self.data)*(self.min_signal_len-(self.seq_len+self.output_len*512))

    def __getitem__(self, idx):
        track = idx // (self.min_signal_len-(self.seq_len+self.output_len*512))
        pos = idx % (self.min_signal_len-(self.seq_len+self.output_len*512))
        return (
            torch.tensor(librosa.feature.mfcc(self.data[track][pos:pos+self.seq_len], n_mfcc=10), device=device),
            torch.tensor(librosa.feature.mfcc(self.data[track][pos+self.seq_len:pos+self.seq_len+(self.output_len*512)], n_mfcc=10), device=device)
        )

audio_dataset = AudioData(r".\archive")
train_loader = DataLoader(audio_dataset, shuffle=True, batch_size=BATCH_SIZE)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = 3
        self.lstm_nodes = 512

        self.lstm = nn.LSTM(196, self.lstm_nodes, num_layers=self.layers, dropout=0.2, batch_first=True)
        self.fc1 = nn.Linear(self.lstm_nodes, audio_dataset.output_len+1) #produces 25600 encoded samples

    #complete forward pass
    def forward(self, x, old_state):
        x, new_state = self.lstm(x, old_state)

        x = self.fc1(x)

        return x, new_state

    def init_state(self, batch_len):
        return (
            torch.zeros(self.layers, batch_len, self.lstm_nodes),
            torch.zeros(self.layers, batch_len, self.lstm_nodes)
        )

net = Net()

torch.cuda.empty_cache()
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.00001) #0.001, 0.00001

#saved = torch.load(r".\model.pt")
#net.load_state_dict(saved["state"])
#optimizer.load_state_dict(saved["optim"])
#for g in optimizer.param_groups:
#     g["lr"] = 0.0001

def train(epochs):
    net.train()
    print("LR : " + str(optimizer.param_groups[0]["lr"]))

    loss_list = []
    for epoch in range(epochs):
        total_loss = 0.
        h,c = net.init_state(BATCH_SIZE)
        
        for idx, (inputx, expected) in enumerate(train_loader):
            try:
                net.zero_grad() #reset gradient between batches
                
                output, (h,c) = net(inputx.to(device), (h.to(device),c.to(device)))
                loss = functional.mse_loss(output, expected.to(device))

                h = h.detach()
                c = c.detach()
            
                loss.backward() #backpropagate loss func
                optimizer.step()

                total_loss += float(loss.item())
                print("BATCH "+str(idx)+" : loss "+str(loss.item()))

                if idx % 5 == 0:
                    loss_list.append(float(loss.item()))
                if idx == 5000:
                    break
    
            except ValueError:
                print(str(idx) + " : " + inputx +"\n\n"+ expected)
        
        print("EPOCH "+str(epoch)+" : total loss "+str(total_loss))

    plt.plot(loss_list)
    plt.show()

    torch.save({"state": net.state_dict(), "optim": optimizer.state_dict()}, r".\model.pt")

def test(wav, pred_len):
    net.eval()
    print("="*25)

    audio,sr = librosa.load(wav, sr=None)
    h,c = net.init_state(1)
    with torch.no_grad():
        for pred in range(pred_len):
            inp = torch.tensor(librosa.feature.mfcc(audio[len(audio)-audio_dataset.seq_len:], n_mfcc=10), device=device)
            output, (h,c) = net(inp.unsqueeze(0), (h.to(device),c.to(device)))

            x = output[0].cpu().numpy()

            x = librosa.feature.inverse.mfcc_to_mel(x)

            x = librosa.feature.inverse.mel_to_audio(x)

            audio = numpy.concatenate((audio, x))

    soundfile.write(r".\output.wav", audio, sr, subtype="PCM_24")
    print("Audio file produced : output.wav")

# SET TASK
#train(1)
#test(os.sep.join(["archive", "genres", "blues", "blues.00000.wav"]), 1)
