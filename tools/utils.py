from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random


def extractor(args):
    def _extract_features1(example):
      features = {
          "image_raw": tf.FixedLenFeature([],tf.string),
          "mask_raw": tf.FixedLenFeature([],tf.string)
      }
      parsed_example = tf.parse_single_example(example, features)                      
      images = tf.image.decode_jpeg(parsed_example['image_raw'])
      images = tf.cast(images,tf.float32)
      masks = tf.cast(tf.image.decode_png(parsed_example['mask_raw']),tf.float32)
      #height = tf.cast(parsed_example['height'],tf.int32)
      #width = tf.cast(parsed_example['width'],tf.int32)
      height = 590
      width = 1640
      
      images_shape = tf.stack([height,width,3])
      images = tf.reshape(images,images_shape)
    
      masks_shape = tf.stack([height,width,3])
      masks = tf.reshape(masks,masks_shape)
      
      x = tf.random.uniform(shape = [1], minval = 0, maxval = width - args.crop_width, dtype = tf.int32)
      x= tf.squeeze(x)
      y = tf.random.uniform(shape = [1], minval = 0, maxval = height - args.crop_height, dtype = tf.int32)
      y = tf.squeeze(y)
      
      images = images[y:y+args.crop_height, x:x+args.crop_width, :]
      masks = masks[y:y+args.crop_height, x:x+args.crop_width, :]
      
      
      #a = tf.random.uniform(shape = [1], minval = 0, maxval = 1, dtype = tf.float32)
      #a = tf.squeeze(a)
      #angle = tf.random.uniform(shape = [1], minval = -3, maxval = 3, dtype = tf.float32)
      #angle = tf.squeeze(angle)
      gamma = tf.random.uniform(shape = [1], minval = 0.5, maxval = 1.5, dtype = tf.float32)
      gamma = tf.squeeze(gamma)          
      images = tf.image.adjust_gamma(images, gamma)
      #images = tf.cond(tf.less(a, 0.5), lambda: tf.image.flip_left_right(images), lambda : images)  
      #masks = tf.cond(tf.less(a,0.5), lambda: tf.image.flip_left_right(masks), lambda : masks)   
      
      masks = tf.reduce_all(tf.equal(masks,[255,255,255]), axis = -1)
      #masks = tf.expand_dims(masks, axis = -1)
      
      
      images = tf.cast(images, tf.float32)
      masks = tf.cast(masks, tf.float32)
    
      images = tf.divide(images, 255)
    	  
      return images,masks

    def _extract_features2(example): 
      features = {
          "image_raw": tf.FixedLenFeature([],tf.string),
          "mask_raw": tf.FixedLenFeature([],tf.string)
      }
      parsed_example = tf.parse_single_example(example, features)                      
      images = tf.image.decode_jpeg(parsed_example['image_raw'])
      images = tf.cast(images,tf.float32)
      masks = tf.cast(tf.image.decode_png(parsed_example['mask_raw']),tf.float32)
      
      height = 576
      width = 1632
      
      images_shape = tf.stack([height,width,3])
      images = tf.reshape(images,images_shape)
    
      masks_shape = tf.stack([height,width,3])
      masks = tf.reshape(masks,masks_shape)
      masks = tf.reduce_all(tf.equal(masks,[255,255,255]), axis = -1)
    
      images = tf.cast(images, tf.float32)
      masks = tf.cast(masks, tf.float32)
     
    
      images = tf.divide(images, 255)
      
    		
      return images,masks


    filename1 = "lane_train.tfrecords"
    filename2= "lane_val.tfrecords"
    data_train = tf.data.TFRecordDataset(filename1)\
                                       .shuffle(buffer_size=10000)\
                                       .map(_extract_features1,num_parallel_calls=16)\
                                       .batch(args.batch_size)\
                                       .prefetch(buffer_size=args.batch_size)\
                                       .repeat()
                                       
    data_val = tf.data.TFRecordDataset(filename2).map(_extract_features2,num_parallel_calls=16)\
                                       .batch(args.batch_size)\
                                       .prefetch(buffer_size=args.batch_size)\
                                       #.repeat()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
  
    iter_train = data_train.make_initializable_iterator()
    iter_val = data_val.make_initializable_iterator()
  
    train_img,train_label = iter_train.get_next()
    val_img,val_label = iter_val.get_next()
    
    return train_img, train_label, iter_train, val_img, val_label, iter_val


def numpy_metrics(y_pred, y_true, n_classes):
    """
    Similar to theano_metrics to metrics but instead y_pred and y_true are now numpy arrays
    """

    # Put y_pred and y_true under the same shape
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    # We use not_void in case the prediction falls in the void class of the groundtruth

    I = np.zeros(n_classes)
    U = np.zeros(n_classes)

    for i in range(n_classes):
        y_true_i = y_true == i
        y_pred_i = y_pred == i

        I[i] = np.sum(y_true_i & y_pred_i)
        U[i] = np.sum((y_true_i | y_pred_i))

    accuracy = np.sum(I) / len(y_pred)
    return I, U, accuracy

def numpy_metrics_binary(y_pred, y_true, batch_size = 1):

    # Put y_pred and y_true under the same shape
    y_pred = y_pred>0.5
    y_pred = np.asarray(y_pred.flatten())
    y_true = np.asarray(y_true.flatten())

    # We use not_void in case the prediction falls in the void class of the groundtruth

    I = 0
    U = 0

    y_true_i = y_true == 1
    y_pred_i = y_pred == 1

    I = np.sum(y_true_i & y_pred_i)
    U = np.sum((y_true_i | y_pred_i))
    try:                                                                                                                                  
        accuracy = float(I) / (np.sum(y_true_i)+1)
        #print(I)
        #print(np.sum(y_true_i))
    except:
        print(I, y_true)
    return I, U, accuracy

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))
