# -*- coding: utf-8 -*-
from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess

def extractor(args):
 with tf.device('/cpu:0'):  
  def _extract_features1(example):
    with tf.device('/cpu:0'):   
      features = {
          "image_raw": tf.FixedLenFeature([],tf.string),
          "mask_orig_raw": tf.FixedLenFeature([],tf.string),
          "height": tf.FixedLenFeature([],dtype=tf.int64),
          "width": tf.FixedLenFeature([],dtype=tf.int64)
      }
      parsed_example = tf.parse_single_example(example, features)
      img_string = parsed_example['image_raw']
      print(img_string)                             
      images = tf.decode_raw(parsed_example['image_raw'],tf.uint8)
      images = tf.cast(images,tf.float32)
      masks = tf.cast(tf.decode_raw(parsed_example['mask_orig_raw'],tf.uint8),tf.float32)
      height = tf.cast(parsed_example['height'],tf.int32)
      width = tf.cast(parsed_example['width'],tf.int32)

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
        
      a = tf.random.uniform(shape = [1], minval = 0, maxval = 1, dtype = tf.float32)
      a = tf.squeeze(a)
      angle = tf.random.uniform(shape = [1], minval = -3, maxval = 3, dtype = tf.float32)
      angle = tf.squeeze(angle)
      gamma = tf.random.uniform(shape = [1], minval = 0.5, maxval = 1.5, dtype = tf.float32)
      gamma = tf.squeeze(gamma)          
                
      images = tf.cond(tf.less(a, 0.5), lambda: tf.image.flip_left_right(images), lambda : images)  
      masks = tf.cond(tf.less(a,0.5), lambda: tf.image.flip_left_right(masks), lambda : masks)   
      
      images = tf.contrib.image.rotate(images, angle)
      masks = tf.contrib.image.rotate(masks, angle)
                
      images = tf.image.adjust_gamma(images, gamma)
      images = tf.cast(images, tf.float32)
      masks = tf.cast(masks, tf.float32)

      images = tf.divide(images, 255)
	  
      return images,masks

  def _extract_features2(example):
   with tf.device('/cpu:0'):      
      features = {
          "image_raw": tf.FixedLenFeature([],tf.string),
          "mask_orig_raw": tf.FixedLenFeature([],tf.string),
          "height": tf.FixedLenFeature([],dtype=tf.int64),
          "width": tf.FixedLenFeature([],dtype=tf.int64)
      }
      parsed_example = tf.parse_single_example(example, features)
      img_string = parsed_example['image_raw']
      print(img_string)                             
      images = tf.decode_raw(parsed_example['image_raw'],tf.uint8)
      images = tf.cast(images,tf.float32)
      masks = tf.cast(tf.decode_raw(parsed_example['mask_orig_raw'],tf.uint8),tf.float32)
      height = tf.cast(parsed_example['height'],tf.int32)
      width = tf.cast(parsed_example['width'],tf.int32)

      images_shape = tf.stack([height,width,3])
      images = tf.reshape(images,images_shape)

      masks_shape = tf.stack([height,width,3])
      masks = tf.reshape(masks,masks_shape)


      images = tf.cast(images, tf.float32)
      masks = tf.cast(masks, tf.float32)
 

      images = tf.divide(images, 255)
      
		
      return images,masks


  filename1 = "lane_train.tfrecords"
  filename2= "v.tfrecords"
  dataset1 = tf.data.TFRecordDataset(filename1).shuffle(buffer_size=503)\
                                               .map(_extract_features1,num_parallel_calls=16)\
                                               .batch(args.batch_size)\
                                               .prefetch(buffer_size=20)\
                                               .repeat()\
  dataset2 = tf.data.TFRecordDataset(filename2).map(_extract_features2,num_parallel_calls=16).batch(args.batch_size).prefetch(buffer_size=40).repeat()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
  
  return dataset1, dataset2