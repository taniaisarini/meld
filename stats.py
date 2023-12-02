import pandas as pd

import utils

emotions = {'sadness': 0, 'surprise': 1, 'joy': 2, 'anger': 3, 'fear': 4, 'disgust': 5, 'neutral': 6}
df = pd.read_csv('ouputs/wav2vec2_lstm/early_stop.csv', header=None, names=['pred', 'true'])
confusion_matrix = {}


for emotion in emotions.keys():
    confusion_matrix[emotion] = {}
    inner_em = confusion_matrix[emotion]
    for emotion in emotions.keys():
        inner_em[emotion] = 0

def map_emotion_labels(emotion):
    return emotions[emotion]




for i in range(df.shape[0]):
    true = utils.label_to_emotion(df.iloc[i]['true'])
    pred = utils.label_to_emotion(df.iloc[i]['pred'])
    confusion_matrix[true][pred] = confusion_matrix[true][pred] + 1
#
for emotion in emotions.keys():
    print("------ emotion: {} -------".format(emotion))
    print(confusion_matrix[emotion])

