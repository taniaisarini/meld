import torch
import lstm_model
import torchaudio
import pandas as pd
from transformers import AutoConfig, Wav2Vec2Processor, TrainingArguments, Wav2Vec2ForSequenceClassification
from datasets import load_dataset, load_metric
import audio_classifier
import numpy as np
from transformers import EvalPrediction
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import pickle

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

import utils

# Data
dev_file_csv = "data/dev_sent_emo.csv"
test_file_csv = "data/test_sent_emo.csv"
train_file_csv = "data/train_sent_emo.csv"

datafiles = {
    "train": "data/train.csv",
    "validation": "data/test.csv"
}

df_dev = pd.read_csv(dev_file_csv)
print("Dev dataset length: {}".format(df_dev.shape[0]))
df_train = pd.read_csv(train_file_csv)
print("Train dataset length: {}".format(df_train.shape[0]))
df_test = pd.read_csv(test_file_csv)
print("Test dataset length: {}".format(df_test.shape[0]))

input_column = 'path'
output_column = 'emotion'


# Parameters
input_size = 100
output_size = 7
num_labels = 7
target_sampling_rate = 0
model_name_or_path = 'jonatasgrosman/wav2vec2-large-xlsr-53-english'

# Hyperparameters
hidden_size = 5
num_layers = 2
lr = 0.001
wd = 1e-5
pooling_mode = "mean"


# # Uncomment when running first time
# utils.create_audio_dataset(df_dev, "data/dev.csv")
# utils.create_audio_dataset(df_train, "data/train.csv")
# utils.create_audio_dataset(df_test, "data/test.csv")

dataset = load_dataset("csv", data_files=datafiles, delimiter="\t", )
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

label_list = train_dataset.unique(output_column)
num_labels = len(label_list)

model_name = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
# feature_extractor.freeze

loss = torch.nn.CrossEntropyLoss()
lstm = lstm_model.LSTM(input_size=512, output_size=7, hidden_size=100, num_layers=2, bidirectional=False, num_epochs=1)
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)


def eval(input, label):
    lstm.eval()

    op = lstm.forward(input)
    result = op[:, -1:, :].to(torch.float).flatten()
    pred = torch.argmax(result)
    target = torch.zeros(result.shape)
    target[label] = 1

    loss_calc = loss(target[None, :], result[None, :])

    correct = 1 if pred.item() == label else 0

    return loss_calc, correct


def train(input, label):
    lstm.train()

    optimizer.zero_grad()
    op = lstm.forward(input)
    result = op[:, -1:, :].to(torch.float).flatten()
    target = torch.zeros(result.shape)
    target[label] = 1
    loss_calc = loss(result[None, :], target[None, :])

    loss_calc.backward()
    optimizer.step()

    return loss_calc


def preprocess(item):
    result = {}
    path = item['path']
    input_audio, sample_rate = librosa.load(path, sr=16000)
    i = feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
    with torch.no_grad():
        o = model(i.input_values)
    result["audio"] = [o.extract_features.squeeze(0)]
    return result

# train_dataset = train_dataset.map(preprocess)
eval_dataset = eval_dataset.map(preprocess)
# with open("train_dataset.pickle", "wb") as f:
#     pickle.dump(train_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("test_dataset.pickle", "wb") as f:
    pickle.dump(eval_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('train_dataset.pickle', 'rb') as data:
#     train_dataset = pickle.load(data)
# with open('eval_dataset.pickle', 'rb') as data:
#     eval_dataset = pickle.load(data)


#
# epochs = 5
# stop = 0.001
# for epoch in range(epochs):
#     total_train_loss = 0
#     for item in train_dataset:
#         audio = torch.Tensor(item['audio'])
#
#         # TRain
#         loss_calc = train(audio, item['emotion'])
#         total_train_loss += loss_calc.item()
#
#     av_loss = total_train_loss/len(train_dataset)
#
#     print("Epoch: {}, av train loss: {}".format(epoch, av_loss))
#
# total_correct = 0
# for item in eval_dataset:
#     audio = torch.Tensor(item['audio'])
#
#     # eval
#
#     loss_calc, correct = eval(audio, item['emotion'])
#     total_correct = total_correct + correct
# print("test accuracy: {}".format(total_correct/len(eval_dataset)))






