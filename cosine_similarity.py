# Get audio for various pairs of utterances.
# Merge (mean, max or sum) to get to be same length
# Compute cosine similarity
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

import utils

dev_file_csv = "data/dev_sent_emo.csv"
df_dev = pd.read_csv(dev_file_csv)
df = df_dev

with open('eval_dataset.pickle', 'rb') as data:
    eval_dataset = pickle.load(data)

# emotions = {'sadness': 0, 'surprise': 1, 'joy': 2, 'anger': 3, 'fear': 4, 'disgust': 5, 'neutral': 6}
emotions = {0: 'sadness', 1: 'surprise', 2: 'joy', 3: 'anger', 4: 'fear', 5: 'disgust', 6: 'neutral'}

confusion_matrix = {}
for emotion in emotions.keys():
    confusion_matrix[emotion] = {}
    inner_em = confusion_matrix[emotion]
    for emotion in emotions.keys():
        inner_em[emotion] = 0


def text_similarity(text1, text2):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
    tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    similarity = 0
    try:
        # Create the TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vector1 = vectorizer.fit_transform(tokens1)
        vector2 = vectorizer.transform(tokens2)

        # Calculate the cosine similarity
        similarity = cosine_similarity(vector1, vector2)
    except:
        pass

    return similarity


def utt_similarity():
    for i in range(df.shape[0]):
        utt = df.iloc[i]['Utterance']
        print("************")
        print(utt)
        for j in range(i + 1, df.shape[0]):
            if j == i:
                continue
            utt2 = df.iloc[j]['Utterance']
        s = text_similarity(utt, utt2)
        s = torch.sum(torch.tensor(s), dim=0)
        s = torch.sum(s, dim=0)
        if (s.item() != 0):
            print(utt2)
            print(s.item())


def merged_audios(audio):
    audio_mean = utils.merged_strategy(audio, 'mean')
    # audio_max = utils.merged_strategy(audio, 'max')
    # audio_sum = utils.merged_strategy(audio, 'sum')
    return audio_mean

# utt_similarity()


unique_emotion = set()
for i in range(len(eval_dataset)):
    emotion_1 = eval_dataset[i]['emotion']
    if emotion_1 not in unique_emotion and len(unique_emotion) < 7:
        counter = {}

        curr_em_dict = confusion_matrix[emotion_1]
        unique_emotion.add(emotion_1)
        audio_1 = torch.tensor(eval_dataset[i]['audio'])
        a1_mean = merged_audios(audio_1)

        for j in range(len(eval_dataset)):
            if len(counter) == 7:
                break
            if j == i:
                continue
            emotion_2 = eval_dataset[j]['emotion']
            if emotion_2 in counter.keys():
                continue
                counter[emotion_2] = counter[emotion_2] + 1
            else:
                counter[emotion_2] = 0

            audio_2 = torch.tensor(eval_dataset[j]['audio'])
            a2_mean = merged_audios(audio_2)

            # Compute cosine similarity for all 3 pairs
            sim_mean = torch.cosine_similarity(a1_mean, a2_mean)

            curr_em_dict[emotion_2] = curr_em_dict[emotion_2] + sim_mean
for l in confusion_matrix.keys():
    print(emotions[l])
    print(confusion_matrix[l])


