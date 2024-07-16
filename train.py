import matplotlib.pyplot as plt
import numpy as np
import time 
import os
import tensorflow as tf

from metrics import *
from model import get_model_deep_speckle
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback

print(K.tensorflow_backend._get_available_gpus())

""" Adjust learning_rate """
def adjust_lr(epoch, lr, mode="step"):
	if mode == "step":
		if (epoch+1)%60==0:
			lr = K.get_value(model.optimizer.lr)
			lr_new = lr/10.0
			K.set_value(model.optimizer.lr, lr_new)
			print("learning_rate changed to {}".format(lr_new))
		else:
			print("learning_rate changed to {}".format(lr))
	else:
		lr = K.get_value(model.optimizer.lr)
		print("learning_rate is still {}".format(lr))
	return K.get_value(model.optimizer.lr)

""" Call Back"""
##class WeightSaver(Callback):
##        def __init__(self,N):
##                self.N = N
##                self.epoch = 0
##        def on_batch_end(self,epoch,logs = {}):
##                if self.epoch % self.N==0:
##                        name = 'weights%08d.h5' % self.epoch
##                        self.model.save_weights(name)
##                self.epoch += 1
##                
##

""" Load data """
tic = time.time()

train_data = np.load('../Train_Digit.npz')
train_speckle = train_data['speckle']
train_ground_truth = train_data['gt']
train_speckle = np.expand_dims(train_speckle, axis=3)
train_ground_truth = np.expand_dims(train_ground_truth, axis=3)
for m in range(train_speckle.shape[0]):
	train_speckle[m,:] = ((train_speckle[m,:]-train_speckle[m,:].min())/(train_speckle[m,:].max()-train_speckle[m,:].min()))
	train_ground_truth[m,:] = train_ground_truth[m,:] / train_ground_truth[m,:].max()

test_data = np.load('../Test_Digit.npz')
test_speckle = test_data['speckle']
test_ground_truth = test_data['gt']
test_speckle = np.expand_dims(test_speckle, axis=3)
test_ground_truth = np.expand_dims(test_ground_truth, axis=3)
for m in range(test_speckle.shape[0]):
	test_speckle[m,:] = ((test_speckle[m,:]-test_speckle[m,:].min())/(test_speckle[m,:].max()-test_speckle[m,:].min()))
	test_ground_truth[m,:] = test_ground_truth[m,:] / test_ground_truth[m,:].max()


""" Model """
model = get_model_deep_speckle()
if os.path.exists('weight.hdf5'):
	model.load_weights('weight.hdf5')
adam = optimizers.Adam(lr=1e-4)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = [SSIM, PSNR, JI])
# model.compile(optimizer = adam, loss = 'mean_absolute_error', metrics = [SSIM, PSNR, JI])

""" Checkpoint """
save_dir = os.path.join(os.getcwd(), 'save_models')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
filepath = "model_{epoch:02d}_{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='loss', verbose=1, save_best_only=True, period=1)
""" Learning rate """
lr_decay = LearningRateScheduler(adjust_lr)


""" Train """
history = model.fit(train_speckle,train_ground_truth,batch_size=32,
	validation_data=(test_speckle, test_ground_truth), 
	epochs=100, shuffle=True, callbacks=[checkpoint, lr_decay])


""" Save """ 
model.save_weights("weight.hdf5")
print("Saved model to disk")
""" Metrics """
_metrics = ['loss', 'SSIM', 'PSNR']
for metric in _metrics:
	plt.figure()
	plt.plot(history.history[metric], label='training '+metric)
	plt.plot(history.history['val_'+metric], label='val '+metric)
	plt.legend()
	plt.savefig(metric+'.eps', bbox_inches='tight', format='eps', dpi=1000)
toc = time.time()
print('Total time:'+str((toc-tic)/60)+'min')

