class toyblock(keras.layers.Layer):
  def __init__(self,nothing=1):
    super(toyblock,self).__init__()
    self.conv_list = []
    self.skips = [[0],[1,0],[2],[3,2,0]]
    for i in range(4):
      self.conv_list.append(Conv_layer(depth =3))
  def call(self,input):
    layers = []
    #out_layers = []
    layers.append(input)
    links = []
    links.append(layers[0])
    x = tf.concat(links,axis = -1)
    layers.append(self.conv_list[0](x))
    links.append(layers[1])
    x = tf.concat(links,axis = -1)
    layers.append(self.conv_list[1](x))
    links = []
    skip_ls = self.skips[2]
    for i in range(len(skip_ls)):
      links.append(layers[skip_ls[i]])
    x = tf.concat(links,axis = -1)
    layers.append(self.conv_list[2](x))
    links = []
    skip_ls = self.skips[3]
    for i in range(len(skip_ls)):
      links.append(layers[skip_ls[i]])
    x = tf.concat(links,axis = -1)
    layers.append(self.conv_list[3](x))
    """for i in range(4):
          links = []
          skip_ls = self.skips[i]
          print("layer is:",layers)
          print("connect these layer:",skip_ls)
          for j in range(len(skip_ls)):
            links.append(layers[skip_ls[j]])
          x = tf.concat(links,axis = -1)
          print("i is ",i)
          layers.append(self.conv_list[i](x))
     """ 
    return layers[4]