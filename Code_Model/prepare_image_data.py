from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import os
from tqdm import tqdm
import time

# extract features from each photo in the directory
def extract_features(directory):
	t1=time.time()
	model = VGG16()
	# re-structure model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	print("Time to load vgg16 model: ",time.time()-t1)
	print(model.summary())
	# extract features from each photo
	features = dict()
	for img in tqdm(listdir(directory)):
		# load image from file
		filename=os.path.join(directory,img)
		image = load_img(filename, target_size=(224, 224))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# make image for vgg
		image = preprocess_input(image)
		feature = model.predict(image, verbose=1)
		image_id = img.split('.')[0]
		features[image_id] = feature
		print('>%s' % img)
	return features

# extract features from all images
img_path = 'Flicker8k_Dataset'
extracted_features = extract_features(img_path)
print('Extracted Features: %d' % len(extracted_features))
# save  file please uncomments if you wants to update
#dump(features, open('features.pkl', 'wb'))