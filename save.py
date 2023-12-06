import os
import h5py

# Import SadTalker modules
from src.models import SadTalkerModel
from src.config import Config


config = Config()
model = SadTalkerModel(config)

h5_file = h5py.File('sadtalker_model.h5', 'w')

model.save_weights(h5_file, save_format='h5')


for attribute, value in config.__dict__.items():
    if attribute in ['optimizer', 'scheduler']:
        continue
    if isinstance(value, list):
        value = [str(v) for v in value]
    h5_file.attrs[attribute] = str(value)
