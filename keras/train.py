import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow import keras
import hardnet
import utils
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
filename = "train_dataset.tfrecord"
feature = {
    'img_raw': tf.io.FixedLenFeature(shape = [], dtype = tf.string),
    'img_mask': tf.io.FixedLenFeature([],tf.string)
    #'shape': tf.io.FixedLenFeature([3],tf.int64)
}
def decode_dataset(example):
    data = tf.io.parse_single_example(example,features=feature)
    img_raw = tf.io.decode_raw(data['img_raw'],tf.int8)
    img_mask = tf.io.decode_raw(data['img_mask'],tf.int8)
    img_raw = tf.cast(tf.reshape(img_raw,shape = [1024,2048,3]),dtype=tf.float32)/255.0
    img_mask = tf.reshape(img_mask,shape =[1024,2048])
    img_mask = tf.cast(tf.expand_dims(img_mask,axis = -1),dtype = tf.float32)
    return (img_raw,img_mask)
dataset =tf.data.TFRecordDataset("train_dataset.tfrecord").map(decode_dataset).shuffle(10).batch(1)
#dataset1 = dataset.take(1)
model = hardnet.HarDNet()
model.compile(optimizer=keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
,loss=utils.Binary_DICE_loss(),metrics=["accuracy"])
cb_list = [tf.keras.callbacks.ModelCheckpoint(
    "my_model", monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq=100),keras.callbacks.TensorBoard(log_dir = "graph",write_graph=True,write_images=True,histogram_freq=1,update_freq='batch')
]
#with tf.device('gpu:0'):
model.fit(dataset,epochs = 1,callbacks=cb_list)
