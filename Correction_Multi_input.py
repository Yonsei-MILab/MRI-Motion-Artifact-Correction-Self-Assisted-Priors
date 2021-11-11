import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np

from Correction_UNet import cbam_block   


IMAGE_ORDERING_CHANNELS_LAST = "channels_last"
IMAGE_ORDERING_CHANNELS_FIRST = "channels_first"

# Default IMAGE_ORDERING = channels_last
IMAGE_ORDERING = IMAGE_ORDERING_CHANNELS_LAST


if IMAGE_ORDERING == 'channels_first':
	MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
	MERGE_AXIS = -1
    
def UNet(img_input):
	k1 = 32
	k2 = 64
	k3 = 128    
	k4 = 256    
	# Block 1 in Contracting Path
	conv1 = Conv2D(k1, (3, 3), data_format=IMAGE_ORDERING,padding='same', dilation_rate=1)(img_input) 
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)  
	#conv1 = Dropout(0.2)(conv1)    
	conv1 = Conv2D(k1, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(conv1) 
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1) 
	
	conv1 = cbam_block(conv1)    # Convolutional Block Attention Module(CBAM) block        
	
	o = AveragePooling2D((2, 2), strides=(2, 2))(conv1)
	
	# Block 2 in Contracting Path
	conv2 = Conv2D(k2, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(o)  
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)       
	conv2 = Dropout(0.2)(conv2)
	conv2 = Conv2D(k2, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(conv2) 
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)    
	    
	conv2 = cbam_block(conv2)    # Convolutional Block Attention Module(CBAM) block  
	    
	o = AveragePooling2D((2, 2), strides=(2, 2))(conv2)    
	
	# Block 3 in Contracting Path
	conv3 = Conv2D(k3, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(o) 
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)  
	#conv3 = Dropout(0.2)(conv3)
	conv3 = Conv2D(k3, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(conv3) 
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)   
	    
	conv3 = cbam_block(conv3)    # Convolutional Block Attention Module(CBAM) block  
	    
	o = AveragePooling2D((2, 2), strides=(2, 2))(conv3)    
	
	 # Transition layer between contracting and expansive paths:
	conv4 = Conv2D(k4, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(o) 
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)     
	#conv4 = Dropout(0.2)(conv4)        
	conv4 = Conv2D(k4, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(conv4) 
	conv4 = BatchNormalization()(conv4)
	conv4 =Activation('relu')(conv4)     
	    
	conv4 = cbam_block(conv4)    # Convolutional Block Attention Module(CBAM) block       
	    
		
	# Block 1 in Expansive Path
	up1 = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(conv4)
	up1 = concatenate([up1, conv3], axis=MERGE_AXIS)
	deconv1 =  Conv2D(k3, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(up1) 
	deconv1 = BatchNormalization()(deconv1)
	deconv1 = Activation('relu')(deconv1)      
	#deconv1 = Dropout(0.2)(deconv1)   
	deconv1 =  Conv2D(k3, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(deconv1) 
	deconv1 = BatchNormalization()(deconv1)
	deconv1 = Activation('relu')(deconv1)     
	    
	deconv1 = cbam_block(deconv1)    # Convolutional Block Attention Module(CBAM) block    
	
	# Block 2 in Expansive Path
	up2 = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(deconv1)    
	up2 = concatenate([up2, conv2], axis=MERGE_AXIS)
	deconv2 = Conv2D(k2, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(up2)   
	deconv2 = BatchNormalization()(deconv2)
	deconv2 = Activation('relu')(deconv2)  
	#deconv2 = Dropout(0.2)(deconv2)       
	deconv2 = Conv2D(k2, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(deconv2)   
	deconv2 = BatchNormalization()(deconv2)
	deconv2 = Activation('relu')(deconv2)       
	    
	deconv2 = cbam_block(deconv2)    # Convolutional Block Attention Module(CBAM) block                                
	
	# Block 3 in Expansive Path
	up3 = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(deconv2)    
	up3 = concatenate([up3, conv1], axis=MERGE_AXIS)
	deconv3 = Conv2D(k1, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(up3)   
	deconv3 = BatchNormalization()(deconv3)
	deconv3 = Activation('relu')(deconv3)   
	#deconv3 = Dropout(0.2)(deconv3)       
	deconv3 = Conv2D(k1, (3, 3), data_format=IMAGE_ORDERING, padding='same', dilation_rate=1)(deconv3)  
	deconv3 = BatchNormalization()(deconv3)
	deconv3 = Activation('relu')(deconv3)    
	    
	deconv3 = cbam_block(deconv3)    # Convolutional Block Attention Module(CBAM) block                                    
	  
	output = Conv2D(1, (3, 3), data_format=IMAGE_ORDERING, padding='same')(deconv3)
	#o = Activation('softmax')(o)
	return output
	
def Correction_Multi_input(input_height, input_width):
	assert input_height % 32 == 0
	assert input_width % 32 == 0

#   UNET
	img_input_1 = Input(shape=(input_height, input_width, 1))	
	img_input_2 = Input(shape=(input_height, input_width, 1))	
	img_input_3 = Input(shape=(input_height, input_width, 1))	
	kk = 32
	conv1 = Conv2D(kk, (3, 3), data_format=IMAGE_ORDERING,padding='same', dilation_rate=1)(img_input_1) # dilation_rate=6
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	conv2 = Conv2D(kk, (3, 3), data_format=IMAGE_ORDERING,padding='same', dilation_rate=1)(img_input_2) # dilation_rate=6
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	conv3 = Conv2D(kk, (3, 3), data_format=IMAGE_ORDERING,padding='same', dilation_rate=1)(img_input_3) # dilation_rate=6
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)     
	
	input_concat = concatenate([conv1, conv2, conv3], axis=MERGE_AXIS)  #conv4    
	
	## Two Stacked Nets:
	pred_1  = UNet(input_concat)
	input_2 = concatenate([input_concat, pred_1], axis=MERGE_AXIS) 
	pred_2  = UNet(input_2) # 
	
	model = Model(inputs=[img_input_1,img_input_2,img_input_3], outputs=pred_2)    
	

	return model                       


if __name__ == '__main__':
	m = Correction_Multi_input()