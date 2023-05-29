from keras.layers import Input, Dense, LSTM, Embedding, concatenate, RepeatVector, TimeDistributed, Bidirectional
from keras.models import Model

def BidirectionalRNNModel(vocab_size, max_len, rnnConfig):
	embedding_size = rnnConfig['embedding_size']
	# vgg16 outputs a 4096 dimensional vector for each image, which we'll feed to RNN Model
	
	image_input = Input(shape=(4096,))
	image_model_1 = Dense(embedding_size, activation='relu')(image_input)
	image_model = RepeatVector(max_len)(image_model_1)

	caption_input = Input(shape=(max_len,))
	# mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs. E.g. it is an efficiency.
	caption_model_1 = Embedding(vocab_size, embedding_size, mask_zero=True)(caption_input)
	# Since we are going to predict the next word using the previous words
	# (length of previous words changes with every iteration over the caption), we have to set return_sequences = True.
	caption_model_2 = LSTM(rnnConfig['LSTM_units'], return_sequences=True)(caption_model_1)
	caption_model = TimeDistributed(Dense(embedding_size))(caption_model_2)

	# Merging the models and creating a softmax classifier
	final_model_1 = concatenate([image_model, caption_model])
	final_model_2 = Bidirectional(LSTM(rnnConfig['LSTM_units'], return_sequences=False))(final_model_1)
	final_model = Dense(vocab_size, activation='softmax')(final_model_2)

	model = Model(inputs=[image_input, caption_input], outputs=final_model)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	
	return model
