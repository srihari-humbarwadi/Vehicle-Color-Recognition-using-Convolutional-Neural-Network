
# coding: utf-8

# In[1]:


from tensorflow.python.keras.layers import concatenate, Conv2D, MaxPooling2D, Dense, Input, Activation, BatchNormalization, Dropout, GlobalAveragePooling2D, Lambda
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model


# In[2]:


def conv_block(tensor, nfilters, size=3, block_type='normal', bn=False, pool=True, stride=1, padding='same', initializer='he_normal', name=''):
    if block_type == 'normal':
        y = Conv2D(filters=nfilters, kernel_size=size, strides=stride, padding=padding, kernel_initializer=initializer, name=f'{name}_conv')(tensor)
        if bn:
            y = BatchNormalization(name=f'{name}_Bn')(y)
        y = Activation('relu', name=f'{name}_relu')(y)
        if pool:
            y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f'{name}_pooling')(y)
        return y
    
    elif block_type == 'split':
        channels = K.int_shape(tensor)[-1]
        top_branch = Lambda(lambda x : x[:,:,:,:channels//2], name=f'{name}_split1')(tensor)
        bottom_branch = Lambda(lambda x : x[:,:,:,channels//2:], name=f'{name}_split2')(tensor)
        split_layers = [top_branch, bottom_branch]
        
        for i in range(2):
            split_layers[i] = Conv2D(filters=nfilters, kernel_size=size, strides=stride, padding=padding, kernel_initializer=initializer, name=f'{name}_conv_{i}')(split_layers[i])
            if bn:
                split_layers[i] = BatchNormalization(name=f'{name}_Bn_{i}')(split_layers[i])
            split_layers[i] = Activation('relu', name=f'{name}_relu_{i}')(split_layers[i])
            if pool:
                split_layers[i] = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f'{name}_pooling_{i}')(split_layers[i])            
        return split_layers[0], split_layers[1]
    else:
        print('You shouldn\'t be here')   


# In[3]:


def ColorModel(img_height=227, img_width=227, nclasses=5):
    print('Building network')
    _ = ['TOP', 'BOTTOM']
    image_input = Input(shape=(img_height, img_width, 3), name='image_input')
    image_output = []
    for i in range(2):
        x = conv_block(tensor=image_input, nfilters=48, size=11, stride=4, bn=True, padding='valid', name=f'{_[i]}_block_1')

        top, bottom = conv_block(tensor=x, nfilters=64, block_type='split', bn=True, pool=True, name=f'{_[i]}_block_2')

        top = conv_block(tensor=top, nfilters=96, size=3, bn=True, pool=False, name=f'{_[i]}_block_3t')
        bottom = conv_block(tensor=bottom, nfilters=96, size=3, bn=True, pool=False, name=f'{_[i]}_block_3b')
        x = concatenate([top, bottom], axis=-1, name=f'{_[i]}_block3_concatenate')

        top, bottom = conv_block(tensor=x, nfilters=96, block_type='split', pool=False, name=f'{_[i]}_block_4')

        top = conv_block(tensor=top, nfilters=64, size=3, bn=True, pool=True, name=f'{_[i]}_block_5t')
        bottom = conv_block(tensor=bottom, nfilters=64, size=3, bn=True, pool=True, name=f'{_[i]}_block_5b')
        image_output.append(top)
        image_output.append(bottom)
    
    image_output = concatenate(image_output, axis=-1, name='features_output')
    image_output = GlobalAveragePooling2D(name='GAP_layer')(image_output)
    image_output = Dense(nclasses, activation='softmax', kernel_initializer='he_normal', name='color_output')(image_output)
    print('Successful')
    return Model(inputs=image_input, outputs=image_output, name='Color-Model')

