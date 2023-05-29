from config import config
from keras.preprocessing.image import load_img, img_to_array
from pickle import load
import tensorflow as tf
from gtts import gTTS
from IPython import display
from model import CNNModel
from keras.applications.vgg16 import preprocess_input
from model import *
import os
from PIL import Image
import matplotlib.pyplot as plt

assert type(config['max_length']) is int, 'Please provide an integer value for `max_length` parameter in config.py file'
assert type(config['beam_search_k']) is int, 'Please provide an integer value for `beam_search_k` parameter in config.py file'

# Extract features from each image in the directory
def extract_features(filename, model, model_type):
	model_type == 'vgg16'
	target_size = (224, 224)
	# Loading and resizing image
	image = load_img(filename, target_size=target_size)
	# Convert the image pixels to a numpy array
	image = img_to_array(image)
	# Reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# Prepare the image for the CNN Model model
	image = preprocess_input(image)
	# Pass image into model to get encoded features
	features = model.predict(image, verbose=0)
	return features

# Load the tokenizer
tokenizer_path = config['tokenizer_path']
tokenizer = load(open(tokenizer_path, 'rb'))

# Max sequence length (from training)
max_length = config['max_length']

# Load the model
caption_model = tf.keras.models.load_model('model_dataset/Best_Model_vgg16last.hdf5')
image_model = CNNModel(config['model_type'])

tf.keras.utils.plot_model(caption_model, to_file='modelRnn.png', show_shapes=True)

tf.keras.utils.plot_model(image_model, to_file='modelCnn.png', show_shapes=True)

# Load and prepare the image
try:
	for image_file in os.listdir(config['test_data_path']):
		if(image_file.split('--')[0]=='output'):
			continue
		if(image_file.split('.')[1]=='jpg' or image_file.split('.')[1]=='jpeg' or image_file.split('.')[1]=='png'):
			print('Generating caption for {}'.format(image_file))
			# Encode image using CNN Model
			image = extract_features(config['test_data_path']+image_file, image_model, config['model_type'])
			# Generate caption using Decoder RNN Model + argmax
			generated_caption = generate_caption(caption_model, tokenizer, image, max_length)
			# Remove startseq and endseq
			caption = 'Caption: ' + generated_caption.split()[1].capitalize()
			for x in generated_caption.split()[2:len(generated_caption.split())-1]:
				caption = caption + ' ' + x
			caption += '.'
			# Show image and its caption
			pil_im = Image.open(config['test_data_path']+image_file, 'r')
			fig, ax = plt.subplots(figsize=(8, 8))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			_ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
			speech = gTTS('Predicted ' + caption, lang = 'en', slow = False)
			speech.save('voice.mp3')
			audio_file = 'voice.mp3'
			display.display(display.Audio(audio_file, rate = None, autoplay = False))
			_ = ax.set_title("Test Result for Argmax Predictions \n{}".format(caption), fontdict={'fontsize': '20','fontweight' : '40'})
			
			plt.savefig(config['output_test_data_path']+'output--'+image_file)
except IndexError:
	print("All Images :")
	print(" ")
	pass

# Load and prepare the image
try:
	for image_file in os.listdir(config['test_data_path']):
		if(image_file.split('--')[0]=='output'):
			continue
		if(image_file.split('.')[1]=='jpg' or image_file.split('.')[1]=='jpeg' or image_file.split('.')[1]=='png'):
			print('Generating caption for {}'.format(image_file))
			# Encode image using CNN Model
			image = extract_features(config['test_data_path']+image_file, image_model, config['model_type'])
			# Generate caption using Decoder RNN Model + BEAM search
			generated_caption = generate_caption_beam_search(caption_model, tokenizer, image, max_length, beam_index=config['beam_search_k'])
			# Remove startseq and endseq
			caption = 'Caption: ' + generated_caption.split()[1].capitalize()
			for x in generated_caption.split()[2:len(generated_caption.split())-1]:
					caption = caption + ' ' + x
			caption += '.'
			# Show image and its caption
			pil_im = Image.open(config['test_data_path']+image_file, 'r')
			fig, ax = plt.subplots(figsize=(8, 8))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			_ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
			speech = gTTS('Predicted ' + caption, lang = 'en', slow = False)
			speech.save('voice.mp3')
			audio_file = 'voice.mp3'
			display.display(display.Audio(audio_file, rate = None, autoplay = False))
			_ = ax.set_title("Test Result for BEAM Search Predictions with k={}\n{}".format(config['beam_search_k'],caption),fontdict={'fontsize': '20','fontweight' : '40'})
			plt.savefig(config['output_test_data_path']+'output-Beam--'+image_file)
except IndexError:
	print("All Images :")
	print(" ")
	pass