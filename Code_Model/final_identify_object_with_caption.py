from keras.preprocessing.text import Tokenizer
from pickle import dump
from prepate_text_data import load_doc
from model import load_set
from model import load_clean_descriptions
from model import tokenizer_fxn
from model import line_to_line
import h5py
import os
import os.path
import cv2

#////////////////////////////////////////////////////////////////////////////////////////////////////////
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model,load_model

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
# New Caption Generate
# training dataset size 60000
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
tokenizer = tokenizer_fxn(train_descriptions)
print("Generate new caption sussessfully : well done")
# save the tokenizer
#dump(tokenizer, open('tokenizer.pkl', 'wb'))

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Identify the object with caption
# extract features from each photo in the directory
def extract_features(filename):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_desc(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for _ in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict which is the next  word
		prediction = model.predict([photo,sequence], verbose=1)
		# convert probability to integer
		prediction = argmax(prediction)
		# map integer to word
		word = word_for_id(prediction, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 34
model = load_model('model_project_final.h5')
model.load_weights("model_project_final_weights.h5")
# load and prepare the photograph
def input_image_file():
	input_image=input("please give image name with full file path :")
	PATH=input_image
	if os.path.isfile(PATH) and os.access(PATH, os.R_OK): 
		print("File exists and is readable")
	else:
		print("Either the file is missing or not readable : please give image which already exist ")
		brk=input("wants to break ( y / n ) : ")
		if brk is 'y':
			input_image=1
		else:
			input_image=input_image_file()
		
	return input_image
while 1:
	
	input_image=input_image_file()
	if input_image==1:
		break	
	print("input_image :- ",input_image)
	photo = extract_features(input_image)
	# generate description
	description = generate_desc(model, tokenizer, photo, max_length)
	print("Generated caption : \n",description[9:-7])
	#import matplotlib.pyplot as plt
	img=cv2.imread(input_image)
	cv2.imshow("image",img)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img,description,(30,50), font, 2,(0,0,0), 3, 0)
	#plt.grid(False)
	#plt.axis('off')
	#plt.title(description[9:-7],color='c',size=25)
	#plt.show()
	cv2.waitKey(0)

	n1=input("Wants to break : press -> y  / otherwise -> n ")
	if n1 is 'y':
		print("summary of full model :")
		print(model.summary())
		break
