import pandas as pd

dev_file_csv = "data/dev_sent_emo.csv"
train_file_csv = "data/train_sent_emo.csv"
test_file_csv = "data/test_sent_emo.csv"

# df_dev = pd.read_csv(dev_file_csv)
df_train = pd.read_csv(train_file_csv)


# def utt_count_by_speaker():
#     speaker_count = dict()
#     for i in range(df_dev.shape[0]):
#         speaker = df_dev.iloc[i]['Speaker']
#         if speaker in speaker_count.keys():
#             speaker_count[speaker] += 1
#         else:
#             speaker_count[speaker] = 1
#     print(speaker_count)



def emotion_by_speaker(df):
    emotion_by_speaker = dict()
    for i in range(df.shape[0]):
        speaker = df.iloc[i]['Speaker']
        emotion = df.iloc[i]['Emotion']
        if speaker in emotion_by_speaker.keys():
            if emotion in emotion_by_speaker[speaker].keys():
                emotion_by_speaker[speaker][emotion] += 1
            else:
                emotion_by_speaker[speaker][emotion] = 1
        else:
            emotion_by_speaker[speaker] = dict()
            emotion_by_speaker[speaker][emotion] = 1
    for speaker in emotion_by_speaker.keys():
        print("*********{}*********".format(speaker))
        for emotion in emotion_by_speaker[speaker].keys():
            print(emotion, emotion_by_speaker[speaker][emotion])


# utt_count_by_speaker()
emotion_by_speaker(df_train)