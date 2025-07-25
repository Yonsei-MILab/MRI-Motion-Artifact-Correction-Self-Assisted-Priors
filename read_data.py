import glob
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical

Height = 256
Width  = 256
#shape  = [3, Height, Width]

def read_data(train_img_path, train_label_path, valid_img_path, valid_label_path, test_img_path, test_label_path):
	# Exact slices i
	fold2_train  = list()  # Training data 
	tr_lbl       = list()  # Training Labels   
	fold1_test   = list()  # Testing data 
	ts_lbl       = list()  # Testing Labels     
	fold3_valid  = list()  # Validation data 
	val_lbl      = list()  # Validation Labels   
    
	# Adjacent slices i-1
	fold2_train_before  = list()  # Training data 
	fold1_test_before   = list()  # Testing data 
	fold3_valid_before  = list()  # Validation data 
    
	# Adjacent slices i+1
	fold2_train_after  = list()  # Training data 
	fold1_test_after   = list()  # Testing data 
	fold3_valid_after  = list()  # Testing data  		

	train_images  = sorted(glob.glob(train_img_path+"*.png"))
	train_label   = sorted(glob.glob(train_label_path+"*.png"))        
	test_images   = sorted(glob.glob(test_img_path+"*.png"))
	test_labels   = sorted(glob.glob(test_label_path+"*.png")) 
	valid_images  = sorted(glob.glob(valid_img_path+"*.png"))
	valid_labels  = sorted(glob.glob(valid_label_path+"*.png"))             

	#Training database:-------------------------------------------------------------
	cnt = -1;
	print("Training Data...")    
	for image_ind in train_images:
		cnt = cnt +1		
		# Exact slices	
		img1 = Image.open(image_ind)       
		img1 = img1.resize((Width, Height), Image.ANTIALIAS)
		img1 = np.stack([img1]*1, axis=-1)  # To generate 3 channel image                 
		fold2_train.append(np.asarray(img1, dtype = np.float32) / 1.)
		# before slices i-1
		if int (image_ind[-7:-4])==1:    # start: S001
			fold2_train_before.append(fold2_train[cnt])
		if int (image_ind[-7:-4])!=1:
			fold2_train_before.append(fold2_train[cnt-1])
		# after slices i+1
		if cnt < len(train_images)-1: # not the last image in train_images
			index_current_slice = int (image_ind[-7:-4])
			index_next_slice    = train_images[cnt+1]  #index_current_slice + 1
			index_next_slice_no = int (index_next_slice[-7:-4])
			img1_next = Image.open(index_next_slice)       
			img1_next = img1_next.resize((Width, Height), Image.ANTIALIAS)
			img1_next = np.stack([img1_next]*1, axis=-1)  # To generate 3 channel image   
			if (index_next_slice_no-index_current_slice)==1:               
				fold2_train_after.append(np.asarray(img1_next, dtype = np.float32) / 1.)	
			else:    # last slice in this subject
				fold2_train_after.append(np.asarray(img1, dtype = np.float32) / 1.)	   # img1   not the img1_next
		else:
			fold2_train_after.append(np.asarray(img1, dtype = np.float32) / 1.)	   # img1   not the img1_next
            

	cnt2 = -1   
	print("Training Label...")      
	for image_ind in train_label: 
		cnt2 = cnt2 +1
		# Exact slices				     
		img1 = Image.open(image_ind)   
		img1 = img1.resize((Width, Height), Image.ANTIALIAS)
		img1 = np.stack([img1]*1, axis=-1)  # To generate 3 channel image         
		tr_lbl.append(np.asarray(img1, dtype = np.float32) / 1.)        
         
	#Testing database:-------------------------------------------------------------
	cnt = -1
	print("Testing Data...")      
	for image_ind in test_images:    
		cnt = cnt +1        
		# Exact slices		      
		img1 = Image.open(image_ind)    
		img1 = img1.resize((Width, Height), Image.ANTIALIAS)
		img1 = np.stack([img1]*1, axis=-1)  # To generate 3 channel image         
		fold1_test.append(np.asarray(img1, dtype = np.float32) / 1.)
		# before slices i-1
		if int (image_ind[-7:-4])==1:    # start: S001
			fold1_test_before.append(fold1_test[cnt])
		if int (image_ind[-7:-4])!=1:
			fold1_test_before.append(fold1_test[cnt-1])
		# after slices i+1
		if cnt < len(test_images)-1: # not the last image in train_images
			index_current_slice = int (image_ind[-7:-4])
			index_next_slice    = test_images[cnt+1]  #index_current_slice + 1
			index_next_slice_no    = int (index_next_slice[-7:-4])       
			img1_next = Image.open(index_next_slice)       
			img1_next = img1_next.resize((Width, Height), Image.ANTIALIAS)
			img1_next = np.stack([img1_next]*1, axis=-1)  # To generate 3 channel image   
			if (index_next_slice_no-index_current_slice)==1:               
				fold1_test_after.append(np.asarray(img1_next, dtype = np.float32) / 1.)	
			else:    # last slice in this subject
				fold1_test_after.append(np.asarray(img1, dtype = np.float32) / 1.)	   # img1   not the img1_next
		else:
			fold1_test_after.append(np.asarray(img1, dtype = np.float32) / 1.)	   # img1   not the img1_next 
        
	print("Testing Label ...")     
	cnt2=-1     
	for image_ind in test_labels:  
		cnt2 = cnt2 +1                
		# Exact slices				     	         
		img1 = Image.open(image_ind)  
		img1 = img1.resize((Width, Height), Image.ANTIALIAS)
		img1 = np.stack([img1]*1, axis=-1)  # To generate 3 channel image         
		ts_lbl.append(np.asarray(img1, dtype = np.float32) / 1.) 

	#Validation database:-------------------------------------------------------------
	cnt = -1
	print("Validation Data...")      
	for image_ind in valid_images:    
		cnt = cnt +1        
		# Exact slices		      
		img1 = Image.open(image_ind)    
		img1 = img1.resize((Width, Height), Image.ANTIALIAS)
		img1 = np.stack([img1]*1, axis=-1)  # To generate 3 channel image         
		fold3_valid.append(np.asarray(img1, dtype = np.float32) / 1.)
		# before slices i-1
		if int (image_ind[-7:-4])==1:    # start: S001
			fold3_valid_before.append(fold3_valid[cnt])
		if int (image_ind[-7:-4])!=1:
			fold3_valid_before.append(fold3_valid[cnt-1])
		# after slices i+1
		if cnt < len(valid_images)-1: # not the last image in train_images
			index_current_slice = int (image_ind[-7:-4])
			index_next_slice    = valid_images[cnt+1]  #index_current_slice + 1
			index_next_slice_no    = int (index_next_slice[-7:-4])            
			img1_next = Image.open(index_next_slice)       
			img1_next = img1_next.resize((Width, Height), Image.ANTIALIAS)
			img1_next = np.stack([img1_next]*1, axis=-1)  # To generate 3 channel image   
			if (index_next_slice_no-index_current_slice)==1:               
				fold3_valid_after.append(np.asarray(img1_next, dtype = np.float32) / 1.)	
			else:    # last slice in this subject
				fold3_valid_after.append(np.asarray(img1, dtype = np.float32) / 1.)	   # img1   not the img1_next
		else:
			fold3_valid_after.append(np.asarray(img1, dtype = np.float32) / 1.)	   # img1   not the img1_next 
        
	print("Validation Label ...")     
	cnt2=-1     
	for image_ind in valid_labels:  
		cnt2 = cnt2 +1                
		# Exact slices				     	         
		img1 = Image.open(image_ind)  
		img1 = img1.resize((Width, Height), Image.ANTIALIAS)
		img1 = np.stack([img1]*1, axis=-1)  # To generate 3 channel image         
		val_lbl.append(np.asarray(img1, dtype = np.float32) / 1.) 		
        

	fold2_train = np.asarray(fold2_train)
	tr_lbl      = np.asarray(tr_lbl)
	fold1_test  = np.asarray(fold1_test)
	ts_lbl      = np.asarray(ts_lbl)  
	fold3_valid  = np.asarray(fold3_valid)
	val_lbl      = np.asarray(val_lbl)       

	fold2_train_before = np.asarray(fold2_train_before)
	fold1_test_before  = np.asarray(fold1_test_before)
	fold3_valid_before  = np.asarray(fold3_valid_before)

	fold2_train_after = np.asarray(fold2_train_after)
	fold1_test_after  = np.asarray(fold1_test_after)
	fold3_valid_after  = np.asarray(fold3_valid_after)

    
	return fold2_train, tr_lbl, fold3_valid, val_lbl, fold1_test, ts_lbl, fold2_train_before, fold3_valid_before, fold1_test_before, fold2_train_after, fold3_valid_after, fold1_test_after
	
