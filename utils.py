import nltk
from transformers import BertTokenizer, AdamW, BertForSequenceClassification
from torch.utils.data import TensorDataset, random_split
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torchaudio
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
dev_max_len = 0
emotions = {'sadness': 0, 'surprise': 1, 'joy': 2, 'anger': 3, 'fear': 4, 'disgust' : 5, 'neutral': 6}
num_to_emotions = {0: 'sadness', 1: 'surprise', 2: 'joy', 3: 'anger', 4: 'fear', 5: 'disgust', 6: 'neutral'}

speaker_to_label = {'Chandler': 0, 'Phoebe': 1, 'Monica': 2, 'Joey': 3, 'Rachel': 4, 'Ross' : 5, 'other': 6}
label_to_speaker = {0: 'Chandler', 1: 'Phoebe', 2: 'Monica', 3: 'Joey', 4: 'Rachel', 5: 'Ross', 6: 'other'}

def get_class_weights(df):
    emotion_count = {}
    for i in range(df.shape[0]):
        emotion = df.iloc[i]['Emotion']
        if emotion in emotion_count.keys():
            emotion_count[emotion] = emotion_count[emotion] + 1
        else:
            emotion_count[emotion] = 1

    weights = torch.zeros(len(num_to_emotions))
    for i in range(len(num_to_emotions)):
        weights[i] = (df.shape[0]/emotion_count[num_to_emotions[i]])
    return weights


def map_emotion_labels(emotion):
    return emotions[emotion]

def map_speaker_labels(speaker):
    if speaker in speaker_to_label.keys():
        return speaker_to_label[speaker]
    else:
        return 6


def label_to_emotion(label):
    return num_to_emotions[label]


def collect_utt(df):
    utterances = []
    for i in range(df.shape[0]):
        utterances.append(df.iloc[i]['Utterance'])
    return utterances


def create_dataset(df):
    labels_emotion = df.Emotion.values
    labels = np.fromiter(map(lambda x: map_emotion_labels(x), labels_emotion.tolist()), dtype=np.int)


    utterances = collect_utt(df)
    # Make input nice and snazzy like BERT wants
    input_ids = []
    attention_masks = []
    for utt in utterances:
        encoded_dict = tokenizer.encode_plus(
            utt,
            add_special_tokens=True,
            max_length=dev_max_len + 10,  # Just in case
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    print("monkey")
    print(labels.shape)
    print(type(labels))
    labels = torch.tensor(labels)
    # TODO (remove this)
    # labels = torch.zeros(labels.shape)

    print(type(labels.shape))
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset


def create_speaker_dataset(df):
    labels_speaker = df.Speaker.values
    labels = np.fromiter(map(lambda x: map_speaker_labels(x), labels_speaker.tolist()), dtype=np.int)


    utterances = collect_utt(df)
    # Make input nice and snazzy like BERT wants
    input_ids = []
    attention_masks = []
    for utt in utterances:
        encoded_dict = tokenizer.encode_plus(
            utt,
            add_special_tokens=True,
            max_length=dev_max_len + 10,  # Just in case
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

def tokenize_utterances():
    utterances = collect_utt()
    for utt in utterances:
        input_ids = tokenizer.encode(utt, add_special_tokens=True)
        dev_max_len = max(dev_max_len, len(input_ids))
    print('Max sentence length: ', dev_max_len)


# sanity checks
def sanity(df):
    # Data check
    print("number of records: {}".format(df.shape[0]))
    # Print utterances and speaker
    for i in range(df.shape[0]):
        print("{} - {}".format(df.iloc[i]['Utterance'], df.iloc[i]['Speaker']))

    # Try tokenizer
    print(df.iloc[8]['Utterance'])
    print(tokenizer.tokenize(df.iloc[8]['Utterance']))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(df.iloc[8]['Utterance'])))


import subprocess
import os

def convert_video_to_audio_ffmpeg(video_file, output_ext="mp3"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)


def convert_to_mp3(filepath):
    for f in os.listdir(filepath):
        vf = filepath + '/' + f
        print(vf)
        convert_video_to_audio_ffmpeg(vf)


def split_audio_by_emotion(filepath):
    pass



def create_audio_dataset(df, savepath, glob_path='data/dev/dev_splits_complete/'):
    data = []
    for i in range(df.shape[0]):
        audio_path = glob_path + 'dia' + str(df.iloc[i]['Dialogue_ID']) + '_utt' + str(
            df.iloc[i]['Utterance_ID']) + '.mp3'
        try:
            s = torchaudio.load(audio_path)
            data.append({
                "path": audio_path,
                "emotion": map_emotion_labels(df.iloc[i]['Emotion']),
                # "audio": torchaudio.load(audio_path)
            })
        except:
            print("failed for {}".format(audio_path))

    dataframe = pd.DataFrame(data)
    dataframe = dataframe.reset_index(drop=True)
    dataframe.to_csv(savepath, sep="\t", encoding="utf-8", index=False)

def merged_strategy(hidden_states, mode="mean"):
    if mode == "mean":
        outputs = torch.mean(hidden_states, dim=1)
    elif mode == "sum":
        outputs = torch.sum(hidden_states, dim=1)
    elif mode == "max":
        outputs = torch.max(hidden_states, dim=1)[0]
    return outputs

# convert_to_mp3('data/dev/dev_splits_complete')
# convert_to_mp3('data/train/train_splits')
# convert_to_mp3('data/test/output_repeated_splits_test')
