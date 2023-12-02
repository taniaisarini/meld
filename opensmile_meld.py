import torch
import lstm_model
import utils
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
import csv
import opensmile
import tensorflow as tf


if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


lr = 0.01
# feature_extractor.freeze
train_file_csv = "data/train_sent_emo.csv"
df_train = pd.read_csv(train_file_csv)
w = utils.get_class_weights(df_train)
loss = torch.nn.CrossEntropyLoss(w)
lstm = lstm_model.LSTM(input_size=512, output_size=7, hidden_size=100, num_layers=2, bidirectional=False,
                       num_epochs=100)
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)

train_df = pd.read_csv('data/dev.csv', sep='\t')
eval_df = pd.read_csv('data/dev.csv')

# target_train = train_df.pop('emotion')

# train_dataset = tf.data.Dataset.from_tensor_slices(dict(train_df))
# eval_dataset = tf.data.Dataset.from_tensor_slices(dict(eval_df))

def eval(input, label):

    op = lstm.forward(input)
    result = op[:, -1:, :].to(torch.float).flatten()
    pred = torch.argmax(result)
    target = torch.zeros(result.shape)
    target[label] = 1

    loss_calc = loss(target[None, :], result[None, :])

    correct = 1 if pred.item() == label else 0

    return loss_calc, correct


def test(input, label):
    lstm.eval()

    op = lstm.forward(input)
    result = op[:, -1:, :].to(torch.float).flatten()
    pred = torch.argmax(result)
    target = torch.zeros(result.shape)
    target[label] = 1

    correct = 1 if pred.item() == label else 0

    return pred.item(), correct


def train(input, label):

    optimizer.zero_grad()
    op = lstm.forward(input)
    result = op[:, -1:, :].to(torch.float).flatten()
    pred = torch.argmax(result)
    target = torch.zeros(result.shape)
    target[label] = 1
    loss_calc = loss(result[None, :], target[None, :])

    correct = 1 if pred.item() == label else 0

    loss_calc.backward()
    optimizer.step()

    return correct, loss_calc


def preprocess(path):
    input_audio, sample_rate = librosa.load(path, sr=16000)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    features = smile.process_signal(
        input_audio,
        sample_rate
    )
    return features

# features = []
# for i in range(train_df.shape[0]):
#     features.append(preprocess(train_df.iloc[i]['path']))
# train_df['smile_features'] = features

train_df.to_csv('train_opensmile.csv')

train_df_feat = pd.read_csv('train_opensmile.csv')

# for item in train_dataset:
#     print(item)

# eval_dataset = train_dataset.map(preprocess)



# stop = 0.0001
# eval_pre = 100
# for epoch in range(lstm.num_epochs):
#     # Train
#     total_train_loss = 0
#     train_correct = 0
#     lstm.train()
#     for item in train_dataset:
#         audio = torch.Tensor(item['audio'])
#
#         # TRain
#         correct, loss_calc = train(audio, item['emotion'])
#         total_train_loss += loss_calc.item()
#         train_correct = train_correct + correct
#
#     av_loss = total_train_loss/len(train_dataset)
#     av_train_accuracy = train_correct / len(train_dataset)
#
#     print("\nEpoch: {}".format(epoch))
#     print("av train loss: {}".format(av_loss))
#     print("av train accuracy: {}".format(av_train_accuracy))
#
#     # Eval
#     lstm.eval()
#     total_eval_loss = 0
#     total_eval_correct = 0
#     for item in eval_dataset:
#         audio = torch.Tensor(item['audio'])
#
#         # eval
#         loss_calc, correct = eval(audio, item['emotion'])
#         total_eval_correct = total_eval_correct + correct
#         total_eval_loss = total_eval_loss + loss_calc
#     diff = eval_pre - total_eval_loss / len(eval_dataset)
#     if(diff < stop):
#         print("Stopping early")
#         break
#     eval_pre = total_eval_loss / len(eval_dataset)
#     print("eval accuracy: {}".format(total_eval_correct/len(eval_dataset)))
#     print("avg eval loss: {}".format(total_eval_loss / len(eval_dataset)))
#
#
#
#
# # Test
# with open('ouputs/wav2vec2_lstm/early_stop.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     total_correct = 0
#     for item in test_dataset:
#         audio = torch.Tensor(item['audio'])
#
#         # eval
#         pred, correct = test(audio, item['emotion'])
#         total_correct = total_correct + correct
#
#         prediction = pred
#         true = item['emotion']
#
#
#         writer.writerow((prediction, true))
