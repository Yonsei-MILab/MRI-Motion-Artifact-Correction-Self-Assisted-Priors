
import numpy as np
from tensorflow.keras import backend as K  
import scipy.io
import tensorflow as tf
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import math

from read_data import read_data
from Correction_Multi_input import Correction_Multi_input

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'#'0,1,2,3,4,5,6,7'

# -------------------------------------------------------
Train = 1 # True False    
Test  = 1 # True False
# -------------------------------------------------------
nb_epoch      = 50    
learningRate  = 0.001 # 0.001
optimizer     = Adam(learning_rate=learningRate)
batch_size    = 10 
Height        = 256     # input image dimensions
Width         = 256 

# PATHES:
train_data_path  = '..path../Training/image/'  
train_GT_path    = '..path../Training/label/'
valid_data_path  = '..path../validation/image/' 
valid_GT_path    = '..path../validation/label/' 
test_data_path   = '..path../Testing/image/'
test_GT_path     = '..path../Testing/label/'

Prediction_path  = '..path../Predictions/'
Weights_path     = '..path../Weights/' 


def save_model(path_weight, model,md = 'lstm'):
	model_json = model.to_json()
	with open(path_weight+"model_"+md+".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(path_weight+"model_"+md+".h5")
	print("The model is successfully saved")    

def load_model(path_weight, md = 'lstm'):
	json_file = open(path_weight+"model_"+md+".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(path_weight+"model_"+md+".h5")
	print("Loaded model from disk")
	return loaded_model

def ssim_score(y_true, y_pred):
	score = K.mean(tf.image.ssim(y_true, y_pred, 255.0))
	return score

def ssim_loss(y_true, y_pred):
	#loss_ssim = 1.0 - K.mean((tf.image.ssim(y_true, y_pred, 255.0)+1.0)/2.0)## SSIM range is between -1~1 so --> +1/2 is added
	#loss_ssim = 1.0 - K.mean(tf.image.ssim(y_true, y_pred, 255.0))
	loss_ssim = 1.0 - K.mean(tf.image.ssim(y_true, y_pred, 255.0))    
	return loss_ssim

	
def scheduler(epoch):
	ep = 10
	if epoch < ep:
		return learningRate
	else:
		return learningRate * math.exp(0.1 * (ep - epoch)) # lr decreases exponentially by a factor of 10
# -------------------------------------------------------
def main():
	print('Reading Data ... ')
	train_data, train_label, valid_data, valid_label, test_data, test_label, fold2_train_before, fold3_valid_before, fold1_test_before, fold2_train_after, fold3_valid_after, fold1_test_after = read_data(train_data_path,train_GT_path,valid_data_path,valid_GT_path,test_data_path,test_GT_path)
    
	print('---------------------------------')
	print('Trainingdata=',train_data.shape)     
	print('Traininglabel=',train_label.shape) 
	print('valid_data=',valid_data.shape)           
	print('valid_label=',valid_label.shape)    
	print('test_data=',test_data.shape)           
	print('test_label=',test_label.shape)        
	print('---------------------------------')
	
	if Train:
		print('---------------------------------')    
		print('Model Training ...')
		print('---------------------------------')
	           
		model = Correction_Multi_input(Height, Width)    
		print(model.summary())        
		csv_logger = CSVLogger(Weights_path+'Loss_Acc.csv', append=True, separator=' ')
		reduce_lr = LearningRateScheduler(scheduler)        
		model.compile(loss=ssim_loss, optimizer=optimizer, metrics=[ssim_score,'mse'])  
		hist = model.fit(x = [fold2_train_before, train_data, fold2_train_after],  # train_CE
						y = train_label, 
						batch_size = batch_size,
						shuffle = True,#False,
						epochs = nb_epoch, #100,
						verbose = 1,          # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch    
						validation_data=([fold3_valid_before, valid_data, fold3_valid_after], valid_label),   # test_CE                         
						callbacks=[csv_logger, reduce_lr]) 
		print('Saving Model...')            
		save_model(Weights_path, model,'CorrectionUNet_') # to save the weight - 'CNN_iter_'+str(i)                   
		
	if Test:
		# Load the model
		print('========================================Load Model-s Weights=====================================')            
		model = load_model(Weights_path, 'CorrectionUNet_') # to load the weight
		print('---------------------------------')        
		print('Evaluate Model on Testing Set ...')
		print('---------------------------------')   
		#pred = model.predict(test_data)   
		pred = model.predict([fold1_test_before, test_data, fold1_test_after])  # test_CE  
		print('==================================')        
		print('Predictions=',pred.shape)    
		print('==================================')    
		
		# To save reconstructed data:         
		inps = sorted(glob.glob(os.path.join(test_data_path, "*.png")))
		assert type(inps) is list
		for i, inp in enumerate(inps):
			out_fname = os.path.join(Prediction_path, os.path.basename(inp))       
			out_img = pred[i,:,:,:]
			cv2.imwrite(out_fname, out_img)            
    
if __name__ == "__main__":
	main()  






