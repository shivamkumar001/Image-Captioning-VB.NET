from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical,plot_model
from keras.models import Model
from keras.layers import Dropout 
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from prepare_text_data_try import load_doc

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

def load_clean_descriptions(filename, dataset):
	doc = load_doc(filename)
	descriptions = dict()
	for img_file in doc.split('\n'):
		div = img_file.split()
		image_id, image_desc = div[0], div[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id] = list()
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			descriptions[image_id].append(desc)
	return descriptions


def load_photo_features(filename, dataset):
	a = load(open(filename, 'rb'))
	features = {k: a[k] for k in dataset}
	return features

# previously descriptions was dictionary . so convert it into  list
def line_to_line(t_disc):
	list1 = list()
	for i in t_disc.keys():
		for j in t_disc[i]:
			list1.append(j)
	return list1

# Tokenizer (class) -> preparing text
# fit -> text documents or integer (encoded) -> text documents.
def tokenizer_fxn(t_disc):
	lines = line_to_line(t_disc)
	# from keras.preprocessing.text import Tokenizer
	token = Tokenizer()
	# it will convert line into set of array
	token.fit_on_texts(lines)
	return token

# calculate the length of the description with the most words
def max_length(t_disc):
	para = line_to_line(t_disc)
	return max(len(i.split()) for i in para)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)

# vocab_soze= size of vocabulary   max_length= length of sentence
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# [image, sequesce] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
	# summarize model
	model.summary()
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]

# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# make photo feature 
train_features = load_photo_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))
#print("train_features ",train_features)

tokenizer = tokenizer_fxn(train_descriptions)
print("tokenizer ",tokenizer)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 10

'''
steps = len(train_descriptions)
for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	# save model
	model.save('model_' + str(i) + '.h5')
model.save('model-ep002-loss3.245-val_loss3_612.h5')
model.save_weights('model-ep002-loss3.245-val_loss3_612_weights.h5')
'''
print("model train sussesfully :")