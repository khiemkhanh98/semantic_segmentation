import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from PIL import Image
from utils import utils
from models.nukev8 import build_nukev8





# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input1 = tf.placeholder(tf.float32,shape=[None,None,3])

net_input = tf.divide(net_input1 ,255)
net_input=tf.expand_dims(net_input, axis=0)

network  = build_nukev8(inputs = net_input, num_classes = 4, is_training = False, dropout_p=0.0)
sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
#saver.restore(sess, args.checkpoint_path)
saver.restore(sess, "079_trnlss_loss:0.228071_iou:0.695515_momentum_gnorm_/model.ckpt")

start=time.time()
loaded_image = np.asarray(Image.open('vid7_56.jpg'))
loaded_image = loaded_image[24:696,192 : 896,:] 
#input_image = np.expand_dims(np.float32(cropped_image),axis=0)

            # st = time.time()

output_image = sess.run(network,feed_dict={net_input1:loaded_image})
            
label_values = np.array([[64,0,128], [128,128,64], [128,64,128], [0,0,0]])
output_image = np.array(output_image[0,:,:,:])
output_image = np.argmax(output_image, axis = -1)
out_vis_image = label_values[output_image]
epoch_time=time.time()-start
file_name = 'test'
cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
cv2.imwrite("%s_ori.png"%(file_name),cv2.cvtColor(np.uint8(loaded_image), cv2.COLOR_RGB2BGR))

print("")
print("Finished!")
print("Wrote image " + "%s_pred.png"%(file_name))
print("time: ",(epoch_time))
