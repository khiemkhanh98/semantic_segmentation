import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np
class Conv_layer(layers.Layer):
  def __init__(self,depth,kernel_size = 3,stride = 1,dropout_rate =0.1,is_training = True):
    super(Conv_layer,self).__init__()
    self.is_training = is_training
    self.conv = keras.layers.Conv2D(depth,kernel_size,stride,padding='same',kernel_initializer=tf.initializers.he_uniform(seed = 1),
                                    kernel_regularizer=keras.regularizers.l2(1e-4))
    self.bn = layers.BatchNormalization(trainable= is_training,fused=True)
    self.relu = layers.ReLU()
    self.dropout = layers.Dropout(dropout_rate)
    self.convs = []
  def call(self,input,is_training = True):
    x = self.conv(input)
    x = self.bn(x,training = is_training)
    x = self.relu(x)
    x = self.dropout(x,training = is_training)    
    return x

#sess = tf.Session()   
#wow = Conv_layer(depth = 8,is_training= True)
#damn = np.ones(shape=(1,128,128,3)).astype(np.float32)
#damn1 = np.ones(shape=(1,128,128,8)).astype(np.float32)
#damn = np.expand_dims(damn,0).astype(np.float32)
#print(damn.shape)
#vc = wow(damn,is_training = True)
#
class HarDBlock(keras.layers.Layer):
    def __init__(self,base_out_channels,growth_rate,layers_num,grmul):
      # layers_num should be base 2 
      super(HarDBlock,self).__init__()
      self.base_out_channels = base_out_channels
      self.growth_rate = growth_rate
      self.layers_num = layers_num
      self.grmul = grmul
      self.conv_list = []
      self.skips = []
      self.out_chs = []
      for i in range(layers_num):
        #self.conv_list.append(Conv_layer(depth = base_out_channels,is_training = True))
        skip_ls,out_channels = self.create_link(i+1,self.base_out_channels,self.growth_rate,self.grmul) 
        self.conv_list.append(Conv_layer(depth = out_channels))
        self.skips.append(skip_ls)
        self.out_chs.append(out_channels)
      #for i in range 
    def create_link(self,layers_num,base_channels,growth_rate,grmul):
      if(layers_num == 0):
        return [] , base_channels
      skip_ls = []
      pow_2 = 1
      for i in range(10):
        if (layers_num % pow_2) == 0:
          skip_ls.append(layers_num - pow_2)
          pow_2 *= 2
          if i >0 :
            growth_rate *= grmul
          continue
        break
      return skip_ls, int(growth_rate)
    def call(self,input ,is_training = True):
        layers = []
        out_layers = []
        layers.append(input)
        for i in range(self.layers_num):
          links = []
          skip_ls = self.skips[i]
          #print("list is ",skip_ls)
          for j in range(len(skip_ls)):
            links.append(layers[skip_ls[j]])
            #print(links)
          x = tf.concat(links,axis = -1)
          #print(x.shape)
          #print(self.conv_list[i])
          #print("i is ",i)
          layers.append(self.conv_list[i](x))
         # Conv_layer.
          if( i % 2 == 1 or i == self.layers_num-1):
            out_layers.append(layers[i+1]) # well remember this clearly, if we put i not i+1,the gradient will skip through the last layer , boom no accuracy left
        #d = tf.gradients(out_layers,[x,layers])
        x =tf.concat(axis=-1,values=out_layers)
        return x
#wow = v(np.ones(shape = (1,128,128,3)).astype(np.float32))
def resize(x, size):
    return tf.image.resize(x,size)
def transition_up(x, skip_connection):
    x = resize(x, (tf.shape(skip_connection)[1], tf.shape(skip_connection)[2]))
    x = tf.concat([x, skip_connection], axis = -1)
    return x
class HarDNet(keras.models.Model):     
        def __init__(self):
          super(HarDNet,self).__init__()
          self.first_ch  = [16,24,32,48]
          self.depth_conv1 = [64, 96, 160, 224, 320]
          self.gr = [  10,16,18,24,32]
          self.grmul = 1.7
          self.n_layers = [4, 4, 8, 8, 8]
          self.up_sample = [126,238,374,534]
          self.blocks = []
          self.conv_ls = []
          self.conv_up = []
          self.blocks_up = []
          self.conv_last = Conv_layer(1,1)
          self.conv1 = Conv_layer(self.first_ch[0],stride = 2)
          self.conv2 = Conv_layer(self.first_ch[1])
          self.conv3 = Conv_layer(self.first_ch[2],stride = 2)
          self.conv4 = Conv_layer(self.first_ch[3])
          n_block = len(self.n_layers) - 1
          for i in range(len(self.n_layers)-1):
            ch = self.first_ch[3]
            self.blocks.append(HarDBlock(ch,self.gr[i],self.n_layers[i],self.grmul))
            self.conv_ls.append(Conv_layer(self.depth_conv1[i]))
          self.conv_ls.append(Conv_layer(self.depth_conv1[len(self.n_layers)-1]))
          for i in range(n_block -1,-1,-1):
              cur_channels_count = self.up_sample[i]//2
              self.conv_up.append(Conv_layer(cur_channels_count, kernel_size=1))
              self.blocks_up.append(HarDBlock(cur_channels_count,self.gr[n_block-1-i],self.n_layers[n_block-1-i],self.grmul))
          #print("block num created",len(self.blocks),len(self.conv_ls),len(self.blocks_up),len(self.conv_up))
        def call(self,input):
          skip_connections = []
          blks = len(self.n_layers)
          #print("shape is wow",input.shape)
          x = self.conv1(input)
          x = self.conv2(x)
          x = self.conv3(x)
          x = self.conv4(x)
          for i in range(blks):
            #print("shape is",x.shape)
            #print("iteration: ",i)
            #self.blocks.append(HarDBlock(ch,self.gr[i],self.n_layers[i],self.grmul)) # maybe incredibly slow here ok maybe a big error
            if i < blks-1:
                blk = self.blocks[i](x)
                skip_connections.append(blk)
            x = self.conv_ls[i](x)
            if i < blks-1:
              x = keras.layers.AveragePooling2D()(x)
            #ch = depth_conv1[i]
          #print("output from encoder is shape:",x.shape)
          #cur_channels_count = depth_conv1[len(depth_conv1)-1]
          n_blocks = blks-1
          for i in range(n_blocks-1,-1,-1):
              skip = skip_connections.pop()
              x = transition_up(x,skip)
              #print("input shape is :",x.shape)
              x = self.conv_up[i](x)
              x = self.blocks_up[i](x)
              #print("i is",i)
          x = self.conv_last(x)
          x = keras.activations.sigmoid(x)
          x = resize(x,(input.shape[1],input.shape[2]))
          #print("wow",x.shape)
          return x

#shit = HarDNet()
#vc = shit(np.ones(shape = (1,512,512,3)).astype(np.float32))
