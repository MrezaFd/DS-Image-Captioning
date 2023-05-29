config = {
	'images_path': 'dataset/Flickr8k_Dataset/', 
	'train_data_path': 'dataset/Flickr8k_text/Flickr_8k.trainImages.txt',
	'val_data_path': 'dataset/Flickr8k_text/Flickr_8k.devImages.txt',
	'captions_path': 'dataset/Flickr8k_text/Flickr8k.token.txt',
	'tokenizer_path': 'model_dataset/tokenizer.pkl',
	'model_data_path': 'model_dataset/', 
	'num_of_epochs': 100,
	'max_length': 40,
	'batch_size': 64,
	'beam_search_k':3,
	'test_data_path': 'test_dataset/', 
	'output_test_data_path': 'test_dataset/',
	'model_type': 'vgg16',
	'random_seed': 1035
}

rnnConfig = {
	'embedding_size': 256,
	'LSTM_units': 256,
	'dense_units': 256,
	'dropout': 0.5
}






