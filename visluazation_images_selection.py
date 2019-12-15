import scipy.misc as misc
import numpy as np
import time
import numpy as np
from glob import glob
import os



def images_selection(file_name, image_width, image_channel, batch_size, num_generations, support_number):
    filenames = glob(os.path.join(file_name, '*.*'))
    fake_categories = len(filenames) * batch_size
    fake_images = np.zeros([fake_categories * num_generations,  image_width,  image_width,  image_channel])
    for i,image_path in enumerate(filenames):
        store_name = file_name + '_split/'
        if not os.path.exists(store_name):
            os.mkdir(store_name)
        current_x = misc.imread(image_path)
        image_size = int(np.shape(current_x)[0]/ batch_size)
        for j in range(batch_size):
            for k in range(support_number+num_generations):
                current_iamge = current_x[image_size*j:image_size*(j+1),image_size*(k):image_size*(k+1)]
                # if len(np.shape(current_iamge))<3:
                #     current_iamge = np.expand_dims(current_iamge,axis=-1)
                current_name =  store_name + image_path.split('/')[-1].split('png')[0] + 'batch{}_sample{}.png'.format(j,k) 
                misc.imsave(current_name, current_iamge)



file_name = './visualization_images/vggface'
image_width = 64
image_channel =3
batch_size = 32
num_generations = 32
support_number = 3
images_selection(file_name, image_width, image_channel, batch_size, num_generations, support_number)