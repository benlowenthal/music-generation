import model
import torch
import tkinter

def open_file():
    file_text.set(tkinter.filedialog.askopenfilename(
        title="Open WAV",
        filetypes=[("wav files", ".wav")]
    ))

def process():
    model.test(file_text.get(), int(text.get()) * sr)

window = tkinter.Tk()
window.title("MusicGAN")
window.geometry("300x200")

text = tkinter.StringVar()
text.set("1")
file_text = tkinter.StringVar()
file_text.set("")

openButton = tkinter.Button(window, text="Open WAV...", command=open_file)
openButton.place(x=10, y=10)

fileEditor = tkinter.Entry(window, textvariable=file_text)
fileEditor.place(x=100, y=12.5)

samplesText = tkinter.Label(window, text="Seconds to predict:")
samplesText.place(x=10, y=60)

lengthEditor = tkinter.Entry(window, textvariable=text)
lengthEditor.place(x=10, y=90)

goButton = tkinter.Button(window, text="Go!", command=process, width=30)
goButton.place(x=10, y=150)

window.mainloop()
