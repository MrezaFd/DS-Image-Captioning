from keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate, RepeatVector, TimeDistributed, Bidirectional, add
from keras.models import Model

def RNNModel(vocab_size, max_len, rnnConfig):
	embedding_size = rnnConfig['embedding_size']
	# vgg16 outputs a 4096 dimensional vector for each image, which we'll feed to RNN Model
	
	image_input = Input(shape=(4096,))
	image_model_1 = Dropout(rnnConfig['dropout'])(image_input)
	image_model = Dense(embedding_size, activation='relu')(image_model_1)

	caption_input = Input(shape=(max_len,))
	# mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs. E.g. it is an efficiency.
	caption_model_1 = Embedding(vocab_size, embedding_size, mask_zero=True)(caption_input)
	caption_model_2 = Dropout(rnnConfig['dropout'])(caption_model_1)
	caption_model = LSTM(rnnConfig['LSTM_units'])(caption_model_2)

	# Merging the models and creating a softmax classifier
	final_model_1 = add([image_model, caption_model])
	final_model_2 = Dropout(rnnConfig['dropout'])(final_model_1)
	final_model_3 = Dense(rnnConfig['dense_units'], activation='relu')(final_model_2)
	final_model = Dense(vocab_size, activation='softmax')(final_model_3)

	model = Model(inputs=[image_input, caption_input], outputs=final_model)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

	return model