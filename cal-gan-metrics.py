import argparse
import data_with_matchingclassifier as dataset
# from generation_builder import ExperimentBuilder
from GAN_Metrics_Tensorflow.frechet_kernel_Inception_distance import *
from GAN_Metrics_Tensorflow.inception_score import * 

import numpy as np
from glob import glob
import os
import cv2


parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
parser.add_argument('--batch_size', nargs="?", type=int, default=20, help='batch_size for experiment')
parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--experiment_title', nargs="?", type=str, default="densenet_generator_fc", help='Experiment name')
parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='continue from checkpoint of epoch')
parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
parser.add_argument('--dropout_rate_value', type=float, default=0.5, help='dropout_rate_value')
parser.add_argument('--num_generations', nargs="?", type=int, default=16, help='num_generations')
parser.add_argument('--support_number', nargs="?", type=int, default=3, help='num_support')
parser.add_argument('--use_wide_connections', nargs="?", type=str, default="False",
                    help='Whether to use wide connections in discriminator')

parser.add_argument('--matching', nargs="?", type=int, default=0)
parser.add_argument('--fce', nargs="?", type=int, default=0)
parser.add_argument('--full_context_unroll_k', nargs="?", type=int, default=4)
parser.add_argument('--average_per_class_embeddings', nargs="?", type=int, default=0)
parser.add_argument('--is_training', nargs="?", type=int, default=0)
parser.add_argument('--samples_each_category',type=int, default=20)
parser.add_argument('--classification_total_epoch',type=int, default=200)

parser.add_argument('--dataset',type=str, default='flowers')
parser.add_argument('--general_classification_samples',type=int, default=5)
parser.add_argument('--selected_classes',type=int, default=0)
parser.add_argument('--image_width', nargs="?", type=int, default=128)
parser.add_argument('--image_channel', nargs="?", type=int, default=3)







args = parser.parse_args()
batch_size = args.batch_size
num_gpus = args.num_of_gpus
support_num = args.support_number

# data = dataset.OmniglotDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
#                                     num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training)



if args.dataset == 'omniglot':
    print('omniglot')
    data = dataset.OmniglotDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes, image_size=args.image_width)

elif args.dataset == 'vggface':
    print('vggface')
    data = dataset.VGGFaceDAGANDataset(batch_size=batch_size, last_training_class_index=1600, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'miniimagenet':
    print('miniimagenet')
    data = dataset.miniImagenetDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'emnist':
    print('emnist')
    data = dataset.emnistDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'figr':
    print('figr')
    data = dataset.FIGRDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)
elif args.dataset == 'fc100':
    data = dataset.FC100DAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)
elif args.dataset == 'animals':
    data = dataset.animalsDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'flowers':
    data = dataset.flowersDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

elif args.dataset == 'flowersselected':
    data = dataset.flowersselectedDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)


elif args.dataset == 'birds':
    data = dataset.birdsDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                        num_of_gpus=num_gpus, gen_batches=1000, support_number=support_num,is_training=args.is_training,general_classification_samples=args.general_classification_samples,selected_classes=args.selected_classes,image_size=args.image_width)

    

    


def resize_image(image):
    image = cv2.resize(image,(299, 299), interpolation=cv2.INTER_LINEAR)
    return image

def get_real_fake_images(fake_images_path):
    real_images = data.get_total_batch_images('test',args.samples_each_category)
    real_images_after = np.zeros([np.shape(real_images)[0],299,299,np.shape(real_images)[3]])


    for i in range(np.shape(real_images)[0]):
        resized_image = resize_image(real_images[i])
        if len(np.shape(resized_image))<3:
            resized_image = np.expand_dims(resized_image,axis=-1)
        real_images_after[i] = resized_image
    real_images_after = 255*(real_images_after/np.max(real_images_after))
    real_images_after = np.transpose(real_images_after, axes=[0, 3, 1, 2])
    # print('real data shape',np.shape(real_images_after))

    filenames = glob(os.path.join(fake_images_path, '*.*'))
    fake_categories = len(filenames)*args.batch_size
    fake_images = np.zeros([fake_categories*args.num_generations,args.image_width, args.image_width, args.image_channel])
    for i,image_path in enumerate(filenames):
        ##(batchsize*28,num_generation+1+support_number)
        current_x = misc.imread(image_path)
        image_size = int(np.shape(current_x)[0]/ args.batch_size)
        for j in range(args.batch_size):
            for k in range(args.num_generations):
                current_iamge = current_x[image_size*j:image_size*(j+1),image_size*(1+args.support_number+k):image_size*(1+args.support_number+k+1)]
                if len(np.shape(current_iamge))<3:
                    current_iamge = np.expand_dims(current_iamge,axis=-1)
                fake_images[args.batch_size*i+j*args.num_generations+k] = current_iamge
    # fake_images = data.preprocess_data(fake_images)

    fake_images_after = np.zeros([np.shape(fake_images)[0],299,299,np.shape(fake_images)[3]])
    for i in range(np.shape(fake_images)[0]):
        resized_image = resize_image(fake_images[i])
        if len(np.shape(resized_image))<3:
            resized_image = np.expand_dims(resized_image,axis=-1)
        fake_images_after[i] = resized_image
    fake_images_after = 255*(fake_images_after/np.max(fake_images_after))
    fake_images_after = np.transpose(fake_images_after, axes=[0, 3, 1, 2])
    # print('fake data shape',np.shape(fake_images_after))

    ##### extending to three channel images for evaluation metrics
    if np.shape(real_images_after)[1]<3:
        three_channel_real_images = np.concatenate([real_images_after, real_images_after, real_images_after], axis=1)
        three_channel_fake_images = np.concatenate([fake_images_after, fake_images_after, fake_images_after], axis=1)
    else:
        three_channel_real_images = real_images_after
        three_channel_fake_images = fake_images_after

    print('real images',np.shape(three_channel_real_images))
    print('fake images',np.shape(three_channel_fake_images))
    return three_channel_real_images, three_channel_fake_images


def inception_score(fake_images_path) :
    _,images = get_real_fake_images(fake_images_path)
    

    BATCH_SIZE = 32
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    logits = inception_logits(inception_images)
    IS = get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)
    print()
    print("IS : ", IS)


def frechet_inception_distance(fake_images_path) :
    real_images, fake_images = get_real_fake_images(fake_images_path)
    BATCH_SIZE = 32
    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    activations = inception_activations(inception_images)
    FID = get_fid(fcd, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)

    print()
    print("FID : ", FID)


def kernel_inception_distance(fake_images_path) :
    real_images, fake_images = get_real_fake_images(fake_images_path)
    
    BATCH_SIZE = 32
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')
    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation, max_block_size=10)
    activations = inception_activations(inception_images)
    KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)

    print()
    print("KID_mean : ", KID_mean * 100)
    print("KID_stddev : ", KID_stddev * 100)

def mean_kernel_inception_distance() :
    filenames = glob(os.path.join('./real_source', '*.*'))
    real_source_images = [get_images(filename) for filename in filenames]
    real_source_images = np.transpose(real_source_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join('./real_target', '*.*'))
    real_target_images = [get_images(filename) for filename in filenames]
    real_target_images = np.transpose(real_target_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join('./fake', '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    BATCH_SIZE = 32

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation,
                                                                               max_block_size=10)
    activations = inception_activations(inception_images)

    FID = get_fid(fcd, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)

    mean_FID = get_fid(fcd, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)
    mean_KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)
    mean_KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)

    mean_FID = (FID + mean_FID) / 2.0
    mean_KID_mean = (KID_mean + mean_KID_mean) / 2.0
    mean_KID_stddev = (KID_stddev + mean_KID_stddev) / 2.0
    print()

    print("mean_FID : ", mean_FID)
    print("mean_KID_mean : ", mean_KID_mean * 100)
    print("mean_KID_stddev : ", mean_KID_stddev * 100)


fake_images_path = './RandomSim126221/flowers1way3shot6layersRandomTrainRandomTest/2connection/visual_outputs/testimages350/'
inception_score(fake_images_path)
frechet_inception_distance(fake_images_path)
kernel_inception_distance(fake_images_path)
mean_kernel_inception_distance()



