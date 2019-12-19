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

#from models.nukev2_gnorm import build_nukev2
#from models.nukev2_gnorm import build_nukev2
from models.FC_DenseNet_Tiramisu import build_fc_densenet
#from models.nukev2 import build_nukev2
from models.nukev8_gnorm import build_nukev8
from utils import utils, helpers



import matplotlib.pyplot as plt
print("change checkpoint 2")
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
class_weight=tf.constant([0.2595,0.1826,4.5640,0.1417,0.9051,0.3826,9.6446,1.8418,0.6823,6.2478,7.3614,0], tf.float64)

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()


class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

with tf.device('/cpu:0'): 
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
      
      images = images[4:356,:,:]
      masks = masks[4:356,:,:]
      # NOTE cast to tf.float32 because most neural networks operate in float32.

      images = tf.cast(images, tf.float32)
      masks = tf.cast(masks, tf.float32)
 

      images = tf.image.per_image_standardization(images)
      
      semantic_map=[]
      for colour in label_values:
          class_map = tf.reduce_all(tf.equal(masks, colour), axis=-1)
          semantic_map.append(class_map)
      masks = tf.stack(semantic_map, axis=-1)
		
      return images,masks


  filename2= "test_camvid4.tfrecords"
  dataset2 = tf.data.TFRecordDataset(filename2).map(_extract_features2,num_parallel_calls=16).batch(args.batch_size).prefetch(buffer_size=40).repeat()

  iter1 = tf.data.Iterator.from_structure(dataset2.output_types,
                                             dataset2.output_shapes)

  net_input,net_output = iter1.get_next()
  net_input.set_shape([args.batch_size, args.crop_height, args.crop_width,3])
  net_output.set_shape([args.batch_size, args.crop_height, args.crop_width,12])
  test_init_op = iter1.make_initializer(dataset2)


# Get the names of the classes so we can record the evaluation results


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


# Compute your softmax cross entropy loss


model_mode = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)
train_state = True
val_state = False

#network = build_nukev2(inputs=net_input, num_classes=num_classes, preset_model='nukev2', n_filters_first_conv=48, n_pool=5, growth_rate=16, n_layers_per_block=4, dropout_p=0.2, l2=1e-4, is_training = model_mode)
#network = build_fc_densenet(inputs = net_input, num_classes = num_classes, preset_model='FC-DenseNet103', n_filters_first_conv=48, n_pool=5, growth_rate=16, n_layers_per_block=4, dropout_p=0.0, scope=None)
network  = build_nukev8(inputs = net_input, num_classes = num_classes, n_filters_first_conv=48, n_pool=5, growth_rate=16, n_layers_per_block=4, dropout_p=dp,is_training = model_mode)
net_output = tf.cast(net_output, tf.float64)

utils.count_params()

    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, var_list=[var for var in tf.trainable_variables()])
    #optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.1, decay = 0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
#saver=tf.train.Saver(tf.contrib.framework.get_trainable_variables()+ tf.contrib.framework.get_variables_by_suffix(suffix='moving_mean')+tf.contrib.framework.get_variables_by_suffix(suffix='moving_variance') )
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())



# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())


# Load a previous checkpoint if desired
model_checkpoint_name = "./latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, "489_trnlss_loss:0.212924_iou:0.611822_momentum_gnorm_/model.ckpt")
    #saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")


print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", 12)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []
# Which validation images do we want


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter( './',
                                      sess.graph)
# Do the training here
for epoch in range(0, args.num_epochs):
    if epoch % args.validation_step == 0:
        print("Performing validation")
        
        I_tot = 0
        U_tot=0
        acc_tot=0
        n_imgs=0
        # Do the validation on a small set of validation images
        with tf.device('/cpu:0'):
            	sess.run(test_init_op)
        for ind in range(int(np.round(223/args.batch_size))):
            # st = time.time()
            with tf.device('/gpu:0'):
                output_img, mask = sess.run([network,net_output], feed_dict={model_mode: val_state})
                #print(output_img.shape)
                #print((mask.shape)[0])
                for i in range((mask.shape)[0]):
                    gt = helpers.reverse_one_hot(mask[i,:,:,:])

                    output_image = np.array(output_img[i,:,:,:])

                    I, U, acc = utils.numpy_metrics(output_image, gt, 11, 11)
                    I_tot += I
                    U_tot += U
                    acc_tot += acc 
                    n_imgs += 1

        labels = ['sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree', 'sign', 'fence', 'car', 'pedestrian',
        'byciclist']
        
        iou= np.mean(I_tot / U_tot)
        accuracy = acc_tot / n_imgs
        for label, jacc in zip(labels, I_tot / U_tot):
            print('{} :\t{:.4f}'.format(label, jacc))
        print ('Mean Jaccard', iou)
        print ('Global accuracy', accuracy)