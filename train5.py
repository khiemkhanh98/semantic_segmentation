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
sys.path.append("models")
from models.nukev8 import build_nukev8

from utils import utils


import matplotlib.pyplot as plt
print("change checkpoint 2")
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
args = parser.parse_args()

label_values = [[64,0,128], [128,128,64], [128,64,128], [0,0,0]]
num_classes = len(label_values)

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
      
      semantic_map=[]
      for colour in label_values:
          class_map = tf.reduce_all(tf.equal(masks, colour), axis=-1)
          semantic_map.append(class_map)
      masks = tf.stack(semantic_map, axis=-1)
	  
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

      
      #images=tf.image.random_crop(images,[args.crop_height,args.crop_width,3],seed=1)
      #masks=tf.image.random_crop(masks,[args.crop_height,args.crop_width,3],seed=1)


      # NOTE cast to tf.float32 because most neural networks operate in float32.

      images = tf.cast(images, tf.float32)
      masks = tf.cast(masks, tf.float32)
 

      images = tf.divide(images, 255)
      
      semantic_map=[]
      for colour in label_values:
          class_map = tf.reduce_all(tf.equal(masks, colour), axis=-1)
          semantic_map.append(class_map)
      masks = tf.stack(semantic_map, axis=-1)
		
      return images,masks


  filename1 = "train_vgu.tfrecords"
  filename2= "val_vgu.tfrecords"
  dataset1 = tf.data.TFRecordDataset(filename1).shuffle(buffer_size=503).map(_extract_features1,num_parallel_calls=16).batch(args.batch_size).prefetch(buffer_size=20).repeat()
  dataset2 = tf.data.TFRecordDataset(filename2).map(_extract_features2,num_parallel_calls=16).batch(args.batch_size).prefetch(buffer_size=40).repeat()

  iter1 = tf.data.Iterator.from_structure(dataset1.output_types,
                                             dataset1.output_shapes)

  net_input,net_output = iter1.get_next()
  train_init_op = iter1.make_initializer(dataset1)
  test_init_op = iter1.make_initializer(dataset2)


# Get the names of the classes so we can record the evaluation results


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


# Compute your softmax cross entropy loss


model_mode = tf.placeholder(tf.bool)
dp = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)
train_state = True
val_state = False

network  = build_nukev8(inputs = net_input, num_classes = num_classes, is_training = model_mode, dropout_p=dp)

reg_loss=tf.add_n(tf.losses.get_regularization_losses())
net_output = tf.cast(net_output, tf.float64)
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=network, onehot_labels=net_output)) + reg_loss
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print('finish update ops')
with tf.control_dependencies(update_ops):
    optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0001, decay = 0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())



# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())


# Load a previous checkpoint if desired
model_checkpoint_name = "./latest_model" + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    #saver.restore(sess, "079_trnlss_loss:0.228071_iou:0.695515_momentum_gnorm_/model.ckpt")
    saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")


print("\n***** Begin training *****")


avg_loss_per_epoch = []
avg_val_loss_per_epoch=[]
avg_scores_per_epoch = []
avg_iou_per_epoch = []
# Which validation images do we want

best_iou = 0.74
# Do the training here
for epoch in range(0, args.num_epochs):
    current_losses = []
    cnt=0
    # Equivalent to shuffling
    num_iters =np.round(503/args.batch_size)
    #num_iters=1
    st = time.time()
    epoch_st=time.time()
    with tf.device('/cpu:0'):
          sess.run([train_init_op])
    for i in range(int(num_iters)):
        with tf.device('/gpu:0'):
        # Do the training
            _,current=sess.run([optimizer,loss],feed_dict={model_mode: train_state, dp:0.2})

        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 60 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    print ('loss for epoch: ', mean_loss)
    avg_loss_per_epoch.append(mean_loss)
	
    if epoch % 1 == 0:
        print("Performing validation")
        
        I_tot = 0
        U_tot=0
        acc_tot=0
        n_imgs=0
        # Do the validation on a small set of validation images
        with tf.device('/cpu:0'):
            	sess.run(test_init_op)
        val_loss = []
        for ind in range(int(np.round(362/args.batch_size))): #we crop the original validation sets into 328 352x352 images
            # st = time.time()
            with tf.device('/gpu:0'):
                output_img, mask, iter_val_loss = sess.run([network,net_output, loss], feed_dict={model_mode: val_state, dp:0.0})
                val_loss.append(iter_val_loss)
                for i in range((mask.shape)[0]):
                    gt = np.argmax(mask[i,:,:,:], axis = -1)

                    output_image = np.array(output_img[i,:,:,:])

                    I, U, acc = utils.numpy_metrics(output_image, gt, 4)
                    I_tot += I
                    U_tot += U
                    acc_tot += acc 
                    n_imgs += 1
        
        mean_val_loss = np.mean(val_loss)
        print('val loss for epoch: ',mean_val_loss)
        
        labels = ['car', 'motorbike','road','void']
        
        iou= np.mean(I_tot / U_tot)
        accuracy = acc_tot / n_imgs
        for label, jacc in zip(labels, I_tot / U_tot):
            print('{} :\t{:.4f}'.format(label, jacc))
        print ('Mean Jaccard', iou)
        print ('Global accuracy', accuracy)
        
        avg_scores_per_epoch.append(accuracy)
        avg_iou_per_epoch.append(iou)
        avg_val_loss_per_epoch.append(mean_val_loss)
    
    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)
    if iou > best_iou:
        best_iou = iou
        print("Saving checkpoint for this epoch")
        saver.save(sess,"./%03d_trnlss_loss:%03f_iou:%03f/model.ckpt"%(epoch,mean_loss,iou))
        
    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []


    fig1, ax1 = plt.subplots(figsize=(11, 8))

    ax1.plot(range(epoch+1), avg_scores_per_epoch, color='green')
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. accuracy")


    plt.savefig('accuracy_vs_epochs.png')

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))
    ax2.plot(range(epoch+1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs.png')
    plt.clf()
    
    
    fig3, ax3 = plt.subplots(figsize=(11, 8))
    ax3.plot(range(epoch+1), avg_iou_per_epoch, color='green')
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")
    plt.savefig('iou_vs_epochs.png')
    
    fig4, ax4 = plt.subplots(figsize=(11, 8))
    ax4.plot(range(epoch+1), avg_val_loss_per_epoch)
    ax4.set_title("Average val loss vs epochs")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Current val loss")

    plt.savefig('val_loss_vs_epochs.png')
    plt.clf()
  
  