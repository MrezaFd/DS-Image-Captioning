import numpy as np
from collections import Counter 
import pandas as pd
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from keras.callbacks import ReduceLROnPlateau

from tqdm import tqdm
import os
from pickle import load, dump
import string
from datetime import datetime as dt
import random
from PIL import Image
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate, RepeatVector, TimeDistributed, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences