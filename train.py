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
from hardnet import *
from tools.utils import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
args = parser.parse_args()

train_img, train_label, iter_train, val_img, val_label, iter_val = extractor(args)
#train_img.set_shape((args.batch_size,args.crop_width,args.crop_height,3))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
writer = tf.summary.FileWriter('log/' , sess.graph)

lr = tf.placeholder(tf.float32)

predict  = hardnet(train_img, is_training = True)
inference = hardnet(val_img, is_training = False, reuse=True)
count_params()

reg_loss=tf.add_n(tf.losses.get_regularization_losses())
coef1 = tf.constant(10.83,dtype = tf.float32)
coef2 = tf.constant(0.34,dtype = tf.float32)
weight_mask = train_label*coef1
temp_mask = tf.math.equal(train_label, tf.zeros(tf.shape(train_label)))
weight_mask+=tf.cast(temp_mask,tf.float32)*coef2
prob = tf.nn.sigmoid(predict, name = 'prob')
num = tf.reduce_sum(prob*train_label)
denom = tf.reduce_sum(prob + train_label-prob*train_label)+1
dice_loss = ((num)/(denom))

train_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=predict, multi_class_labels=train_label, weights=weight_mask)) + reg_loss +1 - dice_loss
train_loss_str = tf.summary.scalar("train_loss", train_loss)
iou_loss_str = tf.summary.scalar("iou_loss", 1 - dice_loss)

val_iou = tf.placeholder(tf.float32)
val_iou_str = tf.summary.scalar("val_iou", val_iou)
val_accuracy = tf.placeholder(tf.float32)
val_accuracy_str = tf.summary.scalar("val_accuracy", val_accuracy)
merge_str = tf.summary.merge([val_accuracy_str, val_iou_str])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
global_step = tf.Variable(0, name='global_step',trainable=False)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0001, decay = 0.995).minimize(train_loss, var_list=[var for var in tf.trainable_variables()], global_step=global_step)

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

model_checkpoint_name = "checkpoint/latest_model" + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    #saver.restore(sess, "079_trnlss_loss:0.228071_iou:0.695515_momentum_gnorm_/model.ckpt")
    saver.restore(sess, model_checkpoint_name)



print("\n***** Begin training *****")


# Which validation images do we want

best_iou = 0.25
total_step = int(args.num_epochs*(88881/args.batch_size)/2000)
val_step = int(9676/args.batch_size)

sess.run([iter_train.initializer])

# Do the training here
for step in range(total_step):
    st = time.time()
    epoch_st=time.time()
    for i in range(2000):
        with tf.device('/gpu:0'):
        # Do the training
            _,train_loss_summary, iou_loss_summary=sess.run([optimizer,train_loss_str, iou_loss_str])
        writer.add_summary(train_loss_summary, 2000*step+i)
        writer.add_summary(iou_loss_summary, 2000*step+i)
    
    print("Time = %.2f"%(time.time()-st))
	
    sess.run([iter_val.initializer])    
    I_tot = 0
    U_tot=0
    acc_tot=0
    batch_imgs=0
    # Do the validation on a small set of validation images
    for i in range(val_step):
        with tf.device('/gpu:0'):
            output_img, mask = sess.run([inference,val_label])
            gt = np.asarray(mask)
            output_image = np.array(output_img)

            I, U, acc = numpy_metrics_binary(output_image, gt)
            I_tot += I
            U_tot += U
            acc_tot += acc 
            batch_imgs += 1
    
    iou = np.mean(float(I_tot )/ U_tot)
    accuracy = acc_tot / batch_imgs
    print('step: %d, iou: %.2f, accuracy: %.2f' %(step+1, iou,accuracy))
    
    #merge_summary = tf.summary.merge([val_accuracy_str, val_iou_str])
    #val_iou.assign(iou)
    #val_accuracy.assign(accuracy)   
    merge_summary = sess.run(merge_str,feed_dict={val_accuracy:accuracy,val_iou:iou})
    writer.add_summary(merge_summary, step)
    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)
    if iou > best_iou:
        best_iou = iou
        print("Saving checkpoint for this epoch")
        saver.save(sess,"checkpoint/%03d_trnlss_loss:%03f_iou:%03f/model.ckpt"%(epoch,mean_loss,iou))
        
  
  