import random
from pickle import load
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from config import config, rnnConfig
from load_data import *
from unidirectional_lstm import RNNModel
from bidirectional_lstm import BidirectionalRNNModel
from model import *
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from train_data import *

# Setting random seed for reproducibility of results
random.seed(config['random_seed'])

assert type(config['num_of_epochs']) is int, 'Please provide an integer value for `num_of_epochs` parameter in config.py file'
assert type(config['max_length']) is int, 'Please provide an integer value for `max_length` parameter in config.py file'
assert type(config['batch_size']) is int, 'Please provide an integer value for `batch_size` parameter in config.py file'
assert type(config['beam_search_k']) is int, 'Please provide an integer value for `beam_search_k` parameter in config.py file'
assert type(config['random_seed']) is int, 'Please provide an integer value for `random_seed` parameter in config.py file'
assert type(rnnConfig['embedding_size']) is int, 'Please provide an integer value for `embedding_size` parameter in config.py file'
assert type(rnnConfig['LSTM_units']) is int, 'Please provide an integer value for `LSTM_units` parameter in config.py file'
assert type(rnnConfig['dense_units']) is int, 'Please provide an integer value for `dense_units` parameter in config.py file'
assert type(rnnConfig['dropout']) is float, 'Please provide a float value for `dropout` parameter in config.py file'

X1train, X2train, max_length = loadTrainData(config)
X1val, X2val = loadValData(config)

tokenizer = load(open(config['tokenizer_path'], 'rb'))
vocab_size = len(tokenizer.word_index) + 1

model = RNNModel(vocab_size, max_length, rnnConfig, config['model_type'])
# model = BidirectionalRNNModel(vocab_size, max_length, rnnConfig, config['model_type'])
print('RNN Model (Decoder) Summary : ')
print(model.summary())

num_of_epochs = config['num_of_epochs']
batch_size = config['batch_size']

steps_train = len(X2train)//batch_size
if len(X2train)%batch_size!=0:
    steps_train = steps_train+1
    
steps_val = len(X2val)//batch_size
if len(X2val)%batch_size!=0:
    steps_val = steps_val+1

model_save_path = config['model_data_path']+"Best_Model_"+str(config['model_type'])+"last.hdf5"
checkpoint = ModelCheckpoint(model_save_path
                             , monitor='val_loss'
                             , verbose=1
                             , save_best_only=True
                             , mode='min' )

reduce_lr = ReduceLROnPlateau(monitor='val_loss'
                              , patience = 4
                              , verbose = 1
                              , factor = 0.01
                              , min_lr = 1e-50
                              , decay = 0.001/32 )

def scheduler(epoch, lr):
    if epoch < 20:
        return lr
    elif epoch > 70 and epoch < 100:
        return lr
    elif epoch > 120 and epoch < 150:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),checkpoint,reduce_lr]

print('steps_train: {}, steps_val: {}'.format(steps_train,steps_val))
print('Batch Size: {}'.format(batch_size))
print('Total Number of Epochs = {}'.format(num_of_epochs))

# Shuffle train data
ids_train = list(X2train.keys())
random.shuffle(ids_train)
X2train_shuffled = {_id: X2train[_id] for _id in ids_train}
X2train = X2train_shuffled

# Create the train data generator
# returns [[img_features, text_features], out_word]
generator_train = data_generator(X1train, X2train, tokenizer, max_length, batch_size, config['random_seed'])

# Create the validation data generator
# returns [[img_features, text_features], out_word]
generator_val = data_generator(X1val, X2val, tokenizer, max_length, batch_size, config['random_seed'])

# Fit for one epoch
history = model.fit(generator_train
                    , epochs=num_of_epochs
                    , steps_per_epoch=50
                    , validation_data=generator_val
                    , validation_steps=steps_val
                    , callbacks=callbacks
                    , verbose=1 )

def evaluate_model(model, images, captions, tokenizer, max_length):
	actual, predicted = list(), list()
	for image_id, caption_list in tqdm(captions.items()):
		yhat = generate_caption(model, tokenizer, images[image_id], max_length)
		ground_truth = [caption.split() for caption in caption_list]
		actual.append(ground_truth)
		predicted.append(yhat.split())
	print('BLEU Scores with Argmax Predictions:')
	print('A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.')
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def evaluate_model_beam_search(model, images, captions, tokenizer, max_length, beam_index=3):
	actual, predicted = list(), list()
	for image_id, caption_list in tqdm(captions.items()):
		yhat = generate_caption_beam_search(model, tokenizer, images[image_id], max_length, beam_index=beam_index)
		ground_truth = [caption.split() for caption in caption_list]
		actual.append(ground_truth)
		predicted.append(yhat.split())
	print('BLEU Scores with Beam Search Predictions:')
	print('A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.')
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

print('Running model on validation set for calculating BLEU score using argmax predictions')
evaluate_model(model, X1val, X2val, tokenizer, max_length)

print('Running model on validation set for calculating BLEU score using BEAM search with k={}'.format(config['beam_search_k']))
evaluate_model_beam_search(model, X1val, X2val, tokenizer, max_length, beam_index=config['beam_search_k'])
