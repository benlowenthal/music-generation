import torch
import tkinter
import lstm_model
import transformer_model
import causal_model

def open_file():
    file_text.set(tkinter.filedialog.askopenfilename(
        title="Open WAV",
        filetypes=[("wav files", ".wav")]
    ))

def process():
    try:
        if model_text.get() == "lstm":
            lstm_model.test(file_text.get(), float(text.get()))
        elif model_text.get() == "transformer":
            transformer_model.test(file_text.get(), float(text.get()))
        else:
            causal_model.test(file_text.get(), float(text.get()))
    except Exception as e:
        print(e.args)
        print("Something went wrong, try again")

window = tkinter.Tk()
window.title("MusicAR")
window.geometry("300x200")

text = tkinter.StringVar()
text.set("1")
file_text = tkinter.StringVar()
file_text.set("")
model_text = tkinter.StringVar()
model_text.set("lstm")

openButton = tkinter.Button(window, text="Open WAV...", command=open_file)
openButton.place(x=10, y=10)

fileEditor = tkinter.Entry(window, textvariable=file_text)
fileEditor.place(x=100, y=12.5)

samplesText = tkinter.Label(window, text="Seconds to predict:")
samplesText.place(x=10, y=60)

lengthEditor = tkinter.Entry(window, textvariable=text)
lengthEditor.place(x=10, y=90)

option = tkinter.OptionMenu(window, model_text, "lstm", "transformer", "causal")
option.place(x=10, y=120)

goButton = tkinter.Button(window, text="Go!", command=process, width=30)
goButton.place(x=10, y=170)

window.mainloop()
