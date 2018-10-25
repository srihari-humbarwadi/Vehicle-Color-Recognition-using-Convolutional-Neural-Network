
# coding: utf-8

# In[1]:


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Nadam, SGD
from tensorflow.python.keras.utils import plot_model, multi_gpu_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from model import ColorModel
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


img_height, img_width = 227, 227
epochs = 50
batch_size = 32
train_gen = ImageDataGenerator(rescale=1/255.)
val_gen = ImageDataGenerator(rescale=1/255.)
train_generator = train_gen.flow_from_directory('dataset/train', target_size=(img_height, img_width), batch_size=batch_size)
val_generator = train_gen.flow_from_directory('dataset/test', target_size=(img_height, img_width), batch_size=batch_size)
stepst = len(train_generator.filenames)//batch_size
stepsv = len(val_generator.filenames)//batch_size

print('\n\nImage input size : ', (img_height, img_width))
print('Batch size :', batch_size)
print('Steps per epoch (train) : ', stepst)
print('Steps per epoch (validation) : ', stepsv)


# In[17]:


tb = TensorBoard(log_dir='logs', histogram_freq=2,batch_size=batch_size,write_grads=True, write_graph=True)
chkpt = ModelCheckpoint('models/top_weights.h5', monitor='val_acc', save_best_only=True, save_weights_only=True,verbose=1)
callbacks = [tb, chkpt]


# In[14]:


colormodel = ColorModel(img_height, img_width)
# p_colormodel = multi_gpu_model(colormodel, 4)
colormodel.compile(optimizer=Nadam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[18]:

print(colormodel.count_params())
colormodel.fit_generator(train_generator, epochs=50, steps_per_epoch=stepst, validation_data=val_generator, validation_steps=stepsv, workers=8, callbacks=callbacks)

