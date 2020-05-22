import tensorflow as tf
import numpy as np


def channel_shuffle(x,groups):
    batchsize, height, width, num_channels= x.shape.as_list()
    
    channels_per_group = num_channels // groups
    
    # reshape
    x = tf.reshape(x,[-1,height,width,groups,channels_per_group])
    
    x = tf.transpose(x,[0,1,2,4,3])
    
    # flatten
    x = tf.reshape(x,[-1,height,width,num_channels])
    
    return x

def SS_nbt_module(inputs,dilated,channels,dropprob,is_training=True):
    oup_inc = channels//2
    residual = inputs
    x1, x2 = tf.split(inputs,2,axis=3)

    output1 = tf.layers.conv2d(x1, oup_inc, [3,1], 1, 'same', use_bias=True,activation=None)
    output1 = tf.nn.relu(output1)
    output1 = tf.layers.conv2d(output1, oup_inc, [1,3], 1, 'same', use_bias=True,activation=None)
    output1 = tf.layers.batch_normalization(output1, training=is_training)
    output1 = tf.nn.relu(output1)

    output1 = tf.layers.conv2d(x1, oup_inc, [3,1], 1, 'same', use_bias=True,activation=None,dilation_rate=(dilated,1))
    ooutput1 = tf.nn.relu(output1)
    output1 = tf.layers.conv2d(output1, oup_inc, [1,3], 1, 'same', use_bias=True,activation=None)
    output1 = tf.layers.batch_normalization(output1, training=is_training)


    output2 = tf.layers.conv2d(x2, oup_inc, [1,3], 1, 'same', use_bias=True,activation=None)
    output2 = tf.nn.relu(output2)
    output2 = tf.layers.conv2d(output2, oup_inc, [3,1], 1, 'same', use_bias=True,activation=None)
    output2 = tf.layers.batch_normalization(output2, training=is_training)
    output2 = tf.nn.relu(output2)

    output2 = tf.layers.conv2d(output2, oup_inc, [1,3], 1, 'same', use_bias=True,activation=None)
    output2 = tf.nn.relu(output2)
    output2 = tf.layers.conv2d(output2, oup_inc, [3,1], 1, 'same', use_bias=True,activation=None)
    output2 = tf.layers.batch_normalization(output2, training=is_training)

    if (dropprob != 0):
        output1 = tf.layers.dropout(output1,rate = dropprob,training=is_training)
        output2 = tf.layers.dropout(output2,rate = dropprob,training=is_training)
        
    out = tf.concat([output1,output2],axis=3)
    out = tf.nn.relu(out+residual)
    
    #shuffle channels
    return channel_shuffle(out,2)

def downsampler_block(inputs,in_channel, out_channel,is_training):
    x1 = tf.layers.max_pooling2d(inputs,2,2)
    x2 = tf.layers.conv2d(inputs, out_channel-in_channel, [3,3], 2, 'same', use_bias=True,activation=None)

    diffY = x2.shape[1] - x1.shape[1]
    diffX = x2.shape[2] - x1.shape[2]

    x1 = tf.pad(x1,[[0,0],[diffX // 2, diffX - diffX // 2],
                    [diffY // 2, diffY - diffY // 2],[0,0]])

    output = tf.concat([x2, x1], axis=3)
    output = tf.layers.batch_normalization(output, training=is_training)
    output = tf.nn.relu(output)
    return output

def encoder(inputs,is_training = True):
    output = downsampler_block(inputs,3,32,is_training = is_training)
#     print(output)
    for i in range(0, 3):
        output = SS_nbt_module(output,1,32,0.03,is_training=is_training)
#         print(output)
    output = downsampler_block(output,32,64,is_training = is_training)
#     print(output)
    for i in range(0, 2):
        output = SS_nbt_module(output,1,64,0.03,is_training=is_training)
    output = downsampler_block(output,64,128,is_training = is_training)

    output = SS_nbt_module(output,1,128,0.3,is_training=is_training)
    output = SS_nbt_module(output,2,128,0.3,is_training=is_training)
    output = SS_nbt_module(output,5,128,0.3,is_training=is_training)
    output = SS_nbt_module(output,9,128,0.3,is_training=is_training)

    output = SS_nbt_module(output,2,128,0.3,is_training=is_training)
    output = SS_nbt_module(output,5,128,0.3,is_training=is_training)
    output = SS_nbt_module(output,9,128,0.3,is_training=is_training)
    output = SS_nbt_module(output,17,128,0.3,is_training=is_training)

#     if(is_training ==False):
#         output = tf.layers.conv2d(output, FLAGS.num_class, [1,1], 1, 'valid', use_bias=True,activation=None)
    return output

def conv2d_bn_relu(inputs,out_channels,kernel_size=1, stride=1,padding='same',is_training = True):
    out = tf.layers.conv2d(inputs, out_channels, kernel_size, stride, padding, use_bias=True,activation=None)
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out)
    return out

def apn_module(inputs,out_channels,is_training):
    b,h,w,c = inputs.shape.as_list()
    branch1 = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
    branch1 = conv2d_bn_relu(branch1,out_channels,kernel_size=1, stride=1,padding='valid',is_training=is_training)

    branch1 = tf.image.resize_bilinear(branch1,[h,w],name = 'branch1')
#     branch1 = tf.image.resize_bilinear(branch1, [FLAGS.image_height,FLAGS.image_width], name='upsample_branch1')
    
    #mid
    branch2 = conv2d_bn_relu(inputs,out_channels,kernel_size=1, stride=1,padding='valid',is_training=is_training)
    
    #branch3
    x1 = conv2d_bn_relu(inputs,1,kernel_size=7, stride=2,padding='same',is_training=is_training)
    x2 = conv2d_bn_relu(x1,1,kernel_size=5, stride=2,padding='same',is_training=is_training)
    x3 = conv2d_bn_relu(x2,1,kernel_size=3, stride=2,padding='same',is_training=is_training)
    x3 = conv2d_bn_relu(x3,1,kernel_size=3, stride=1,padding='same',is_training=is_training)

    x3 = tf.image.resize_bilinear(x3, [np.ceil(h/4).astype(np.int32),np.ceil(w/4).astype(np.int32)], name='upsample_branch3_x3')
    x2 = conv2d_bn_relu(x2,1,kernel_size=5, stride=1,padding='same',is_training=is_training)

    x = x2 + x3
    x = tf.image.resize_bilinear(x, [h//2,w//2], name='upsample_branch3_x2')
    x1 = conv2d_bn_relu(x1,1,kernel_size=7, stride=1,padding='same',is_training=is_training)
    x = x+x1
    x = tf.image.resize_bilinear(x, [h,w], name='upsample_branch3_x1')
    x = x * branch2

    x = x + branch1
    return x

def decode(encoded_output,image_height,image_width,is_training = True):
    apn_output = apn_module(encoded_output,2,is_training = is_training)
    decoded_output = tf.image.resize_bilinear(apn_output, [image_height,image_width], name='upsample_branch_final')
    return decoded_output


def LEDNet(inputs,image_height,image_width,num_class,is_training = True):
    encoded_output =  encoder(inputs,is_training = is_training)
    cls_result = tf.layers.conv2d(encoded_output, num_class, [3,3], 2, 'valid', use_bias=True,activation=None)
    cls_result = tf.reduce_mean(cls_result,axis=1)
    cls_result = tf.reduce_mean(cls_result,axis=1)
    decoded_output = decode(encoded_output,image_height,image_width,is_training = is_training)
    return decoded_output,cls_result
