import tensorflow as tf
import numpy as np
import time

l2 = 1e-4
a = 0
def get_link(layer, base_ch, growth_rate, grmul):
    if layer == 0:
        return base_ch, 0, []
    out_channels = growth_rate
    link = []
    
    for i in range(10):
        dv = 2 ** i
        if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
    
    out_channels = int(int(out_channels + 1) / 2) * 2
    
    return out_channels, link

def conv(x, depth, kernel_size=3, strides=1, padding = 'same'):
    global a
    a=a+1
    print(a)
    return tf.layers.conv2d(x, depth, (kernel_size,kernel_size), (strides,strides), padding= padding,
                           kernel_regularizer=tf.contrib.layers.l2_regularizer(l2),kernel_initializer = tf.initializers.he_uniform(seed = 1))

def batch_norm(x, is_training):
    return tf.layers.batch_normalization(x, training= is_training, fused=True)

def relu(x):
    return tf.nn.relu(x)

def resize(x, size):
    return tf.image.resize_bilinear(x, size)

def drop_out(x, rate, is_training):
    return tf.layers.dropout(x, rate, training = is_training)

def avg_pool(x, pool_size = 2, strides = 2):
    return tf.layers.average_pooling2d(x, pool_size, strides, padding = 'same')

def conv_layer(x, depth, kernel_size=3, strides=1, dropout_rate=0.1, is_training = False):
    x = conv(x, depth = depth, kernel_size=kernel_size, strides=strides)
    x = batch_norm(x, is_training)
    x = relu(x)
    x = drop_out(x, rate = (dropout_rate), is_training = is_training)
    return x

def hardblock(x, in_channels, growth_rate, grmul, n_layers):
    layers = []
    out_layers = []
    layers.append(x)
    out_channels = 0
    
    for i in range(n_layers):
        
        links = []
        out_ch, link = get_link(i+1, in_channels, growth_rate, grmul)
        if (i % 2 == 0) or (i == n_layers - 1):
            out_channels += out_ch
    
        for i in range(len(link)):
            links.append(layers[i])

        x = tf.concat(links, axis = -1)
        layers.append(conv_layer(x, out_channels))
        
    
    for i in range(len(layers)):
        if (i == i-1) or (i%2 == 1):
            out_layers.append(layers[i])
    
    x = tf.concat(out_layers, axis = -1)
    return x

def transition_up(x, skip_connection):
    x = resize(x, (tf.shape(skip_connection)[1], tf.shape(skip_connection)[2]))
    x = tf.concat([x, skip_connection], axis = -1)
    return x

def hardnet(input_img, is_training = False):
    first_ch  = [16,24,32,48]
    depth_conv1 = [64, 96, 160, 224, 320]
    grmul = 1.7
    gr       = [  10,16,18,24,32]
    n_layers = [   4, 4, 8, 8, 8]
    up_sample = [126,238,374,534]
    skip_connections = []
    blks = len(n_layers)
    
    
    x = conv_layer(input_img, first_ch[0], strides = 2)
    x = conv_layer(x, first_ch[1])
    x = conv_layer(x, first_ch[2], strides = 2)
    x = conv_layer(x, first_ch[3])
    
    ch = first_ch[3]
    for i in range(blks):
        blk = hardblock(x, ch, gr[i], grmul, n_layers[i])
        if i < blks-1:
            skip_connections.append(blk)
        x = conv_layer(x, depth_conv1[i])
        if i < blks-1:
          x = avg_pool(x)
        ch = depth_conv1[i]
    
    cur_channels_count = depth_conv1[len(depth_conv1)-1]
    n_blocks = blks-1
    
    for i in range(n_blocks-1,-1,-1):
        skip = skip_connections.pop()
        
        x = transition_up(x, skip)
        
        cur_channels_count = up_sample[i]//2
        x = conv_layer(x, cur_channels_count, kernel_size=1)
        blk = hardblock(x, cur_channels_count, gr[i], grmul, n_layers[i])

    x = conv_layer(x, depth = 1, kernel_size = 1)
    #x = tf.nn.sigmoid(x, name = 'prob')
    x = tf.image.resize_bilinear(x, (tf.shape(input_img)[1], tf.shape(input_img)[2]))
    return x

tf.reset_default_graph()
img = tf.placeholder(shape = (None,None,None,3), dtype = tf.float32, name = 'input')
sess = tf.Session()
model = hardnet(input_img = img)
sess.run(tf.global_variables_initializer())
sess.run([model], feed_dict={img:np.zeros((1,256,512,3))})
