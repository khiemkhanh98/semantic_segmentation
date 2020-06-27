import tensorflow as tf
from tensorflow import keras
class Binary_DICE_loss(keras.losses.Loss):
  # this is for classifying only 2 object types: lane vs non-lane
  """Arg :
  """
  def __init__(self,coef1 = 2,coef2 = 1): # as
    super(Binary_DICE_loss,self).__init__()
    self.coef1 = coef1
    self.coef2 = coef2
    self.bce = tf.keras.losses.BinaryCrossentropy()
  def call(self,y_true,y_pred):
    # this y_true must have shape [n,n,1] for loss func to work properly with weights
    #actually this loss func can also work properly with batch > 1
    print(y_pred)
    #y_true.set_shape(shape=(1,1024,2048,1))
    print(y_true)
    weight_mask = y_pred
    img_weight = y_true*self.coef1 #multiply lane_pixel (1) with big coef to punish mislabelling lane pixel harder.
    temp_mask = tf.cast(tf.equal(y_true,tf.zeros(shape = tf.shape(y_true))),dtype = tf.float32) # why tf.zeros not np.zeros
    temp_mask *= self.coef2
    weight_mask += temp_mask
    weight_mask  = keras.backend.squeeze(weight_mask,-1)
    x = self.bce(y_true,y_pred,sample_weight = weight_mask)
    prob = y_pred
    num = tf.reduce_sum(prob*y_true)
    denom = tf.reduce_sum(prob + y_true-prob*y_true)+1 #??? wtf is this denom? it is supposed to be prob^2 + train_label ^2 ? 
    x += num / denom
    return x