# Preparation file

# Imports
import pandas as pd
import matplotlib
from keras.datasets import cifar10
from keras import backend as K
import os
import csv
import numpy as np
import tensorflow as tf
from setup_cifar import CIFAR, CIFARModel

# Custom Networks
from networks.lenet import LeNet
from networks.resnet import ResNet

# Helper functions
import helper

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted)


matplotlib.style.use('ggplot')

results_folder = './results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Load dataset
data = CIFAR()
x_train = data.train_data
x_test = data.test_data
y_train = data.train_labels
y_test = data.test_labels
y_test = np.argmax(y_test, axis=1)
y_test = y_test.reshape(-1, 1)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load models 
cifar_ = CIFARModel('cifar')
cifar_100 = CIFARModel('cifar-distilled-100')

models = [cifar_, cifar_100]
modelNames = ['regular', 'distilled']

for i in range(len(models)):
    model_folder = f'{results_folder}/{modelNames[i]}'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

# Calculate Model Accuracies
network_stats, correct_imgs = helper.evaluate_models(models, modelNames, x_test, y_test)
correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy'])
# export to csv file
file_network_stats = f'{results_folder}/network_stats.csv'
network_stats.to_csv(file_network_stats, index=False) 

print(network_stats)
# CHANGE HERE HOW MANY SAMPLES YOU WANT TO ATTACK
n_samples = 500
n_classes = 10
seed = n_samples
np.random.seed(seed)

i = 0
for model in models:
    modelName = modelNames[i]
    valid = correct_imgs[correct_imgs.name == modelName]
    valid_imgs = correct_imgs[correct_imgs.name == modelName].img
    
    # block is image from class 1, image from class 2, ... image from class 10 (10 images/n_classes in a block)
    blocks = int(n_samples/n_classes)
    free_ids = valid_imgs
    free_imgs = valid
    img_samples = []
    
    for block in range(blocks):
        for cl in range(n_classes):
            valid_id_class = free_imgs[free_imgs.label == cl].img
            chosen_id = np.random.choice(valid_id_class, 1, replace=False)[0]
            img_samples.append(chosen_id)
            
            free_ids = [i for i in free_ids if i != chosen_id]
            free_imgs = free_imgs[free_imgs.img != chosen_id]


    # export to csv (save images that will be attacked)
    file_images_to_attack = f'{results_folder}/{modelName}/images_to_attack_idx.csv'
    f_img = open(file_images_to_attack, 'w')
    writer_img = csv.writer(f_img)
    header = ['image id', 'true label', 'confidence', 'prediction']
    writer_img.writerow(header)
    for img_id in img_samples:
        pred =  valid[valid.img == img_id].pred.values[0]
        row = [img_id, valid[valid.img == img_id].label.values[0], valid[valid.img == img_id].confidence.values[0], list(pred)]
        writer_img.writerow(row)
    f_img.close()

    # Create file for mean metrics 
    file_metrics_mean = f'{results_folder}/{modelName}/metrics_mean.csv'
    f_metrics = open(file_metrics_mean, 'w')
    writer_metrics = csv.writer(f_metrics)
    header = ['abordagem', 'success rate dataset mean']
    writer_metrics.writerow(header)
    f_metrics.close()

    # Create file for mean metrics of images
    file_metrics_mean_img = f'{results_folder}/{modelName}/metrics_mean_img.csv'
    f_metrics_img = open(file_metrics_mean_img, 'w')
    writer_metrics_img = csv.writer(f_metrics_img)
    header = ['img_counter', 'img_idx', 'abordagem', 'mean success rate', 'mean adv_prob']
    writer_metrics_img.writerow(header)
    f_metrics_img.close()

    # Create file for number of pixels information
    file_npixel = f'{results_folder}/{modelName}/number_pixels.csv'
    f_npixel = open(file_npixel, 'w')
    writer_pixel = csv.writer(f_npixel)
    header = ['abordagem', 'run', 'img_idx', 'img_counter', 'number of pixels']
    writer_pixel.writerow(header)
    f_npixel.close()

    i += 1