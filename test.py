import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


input_audio, sample_rate = librosa.load("data/dev/dev_splits_complete/dia0_utt1.mp3",  sr=16000)

model_name = "facebook/wav2vec2-large-xlsr-53"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
with torch.no_grad():
  o= model(i.input_values)
print(o.keys())
print(o.last_hidden_state.shape)
print(o.extract_features.shape)