import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import get_model_deep_speckle
from keras import optimizers
from metrics import *
##from keras.models import load_model

add = '../1_digit/'
""" Load data """
# data = np.load('Test_Cifar.npz')
data = np.load(add + 'Test_Digit.npz')
speckle = data['speckle']
label = data['gt']
speckle = np.expand_dims(speckle, axis=3)
ground_truth = np.expand_dims(label, axis=3)
for m in range(speckle.shape[0]):
	speckle[m,:] = (speckle[m,:]-speckle[m,:].min())/(speckle[m,:].max()-speckle[m,:].min())
	ground_truth[m,:] = ground_truth[m,:] / ground_truth[m,:].max()

 
model = get_model_deep_speckle()
path = 'weight.hdf5'
model.load_weights(add+path)

""" Predict """
""" Show examples """
plt.figure(dpi=130,figsize=(10,6))


for i in range(6):
	ran = np.random.randint(speckle.shape[0])
	speckle_test = speckle[ran,:]
	pred_speckle_test = model.predict(speckle_test.reshape(1,64,64,1))

	plt.subplot(3, 6, i+1)
	plt.imshow(label[ran,:].squeeze(), cmap='gray')
	plt.axis('off')

	plt.subplot(3, 6, i+1+6)
	plt.imshow(speckle[ran,:,:,0].squeeze(), cmap='gray')
	plt.axis('off')

	plt.subplot(3, 6, i+1+12)
	plt.imshow(pred_speckle_test[0,:,:,0].squeeze(), cmap='gray')
	plt.axis('off')

plt.show()



""" Evaluate """
# adam = optimizers.Adam(lr=1e-4)
# model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=[SSIM, PSNR, JI])
# preds = model.evaluate(speckle, ground_truth)
# print ( path)
# print ("Test SSIM = " + str(preds[1]))
# print ("Test PSNR = " + str(preds[2]))
# print ("Test JI = " + str(preds[3]))
