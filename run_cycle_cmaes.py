# Imports
import pickle
import numpy as np
import pandas as pd
import matplotlib
from keras.datasets import cifar10
from keras import backend as K
import os
import time
import csv
import matplotlib.pyplot as plt
import tensorflow as tf 
from math import log

# Custom Networks
from networks.lenet import LeNet
from networks.resnet import ResNet
from setup_cifar import CIFAR, CIFARModel

# Helper functions
from cmaes import cmaes
import helper
import gc

matplotlib.style.use('ggplot')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load dataset
data = CIFAR()
x_train = data.train_data
x_test = data.test_data
x_train = (x_train + 0.5) * 255
x_test= (x_test + 0.5) * 255

y_train = data.train_labels
y_test = data.test_labels
y_train = np.argmax(y_train, axis=1)
y_train = y_train.reshape(-1, 1)
y_test = np.argmax(y_test, axis=1)
y_test = y_test.reshape(-1, 1)


# Load models 
cifar_ = CIFARModel('cifar')
cifar_100 = CIFARModel('cifar-distilled-100')

models = [cifar_, cifar_100]
modelNames = ['regular','distilled']


(h, w, d) = x_test[0].shape
bounds = [[0, w - 1], [0, h - 1], [0, 255], [0, 255], [0, 255]]
bounds = np.array(bounds)

### CHANGE PARAMETERS HERE ### 
nruns = 10
seeds = list(range(nruns))
n_samples = 500

ind_size = 5
lambda_ = 400              #int(4 + 3 * log(ind_size)) # population size, number of offspring to be generated and evaluated in each generation
mu = int(lambda_/2)                                  # number of individuals selected as parents in each generation
sigma_ =  0.05                                        # initial step size or std of the search distribution
centroid = [0.5, 0.5, 0.5, 0.5, 0.5]

generations = 100                                    # 5000 if lambda is 8 (40000 evaluations)
pop_size = lambda_
pixels = 1

mn = 0

for model in models:
    modelName = modelNames[mn]

    # Create de folder
    cmaes_folder = f'./results/{modelName}/cmaes'
    if not os.path.exists(cmaes_folder):
        os.makedirs(cmaes_folder)

    # Time
    time_file = f'{cmaes_folder}/time.csv'
    if not os.path.exists(time_file):
        header_time = ['model', 'run', 'time']
        f_time = open(time_file, 'w')
        writer_time = csv.writer(f_time)
        writer_time.writerow(header_time)
        f_time.close()
    
    # Select images to attack
    images_df = pd.read_csv(f'./results/{modelName}/images_to_attack_idx.csv')
    images_idx = images_df['image id']
    images_labels = images_df['true label']

    # Write csv with metrics
    file_metrics = f'{cmaes_folder}/metrics.csv'
    f_metrics = open(file_metrics, 'w')
    writer_metrics = csv.writer(f_metrics)
    writer_metrics.writerow(['run', 'success rate dataset', 'time', 'success rate (per img)', 'adv prob label (per img)'])
    
    # Write csv with pixels covered
    file_cover = f'{cmaes_folder}/covered_pixels.csv'
    f_cover = open(file_cover, 'w')
    writer_cover = csv.writer(f_cover)
    writer_cover.writerow(['run', 'img_counter', 'img_id', 'number of covered pixels'])
    f_cover.close()

    # Mean (across runs) success rate (per img)
    dict_success_rate = {}
    dict_adv_prob1 = {}
    dict_adv_prob2 = {}
    
    success_rate_dataset_store = []

    for run in range(1, nruns+1):
        suc_samples = 0 # from n samples, how many were successfully attacked at least once?
        samples = []

        success_rate_per_img = []
        adv_prob_per_img = []

        # Create run folder
        print("---------------------------")
        print("Starting run ", run)
        run_folder = f'{cmaes_folder}/run_{run}'
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
        
        # File to store best individual per image
        file_bests =  f'{run_folder}/best_individuals.csv'
        f_bests = open(file_bests, 'a')
        writer_bests = csv.writer(f_bests)
        header_best = ['img_counter', 'img_id', 'best pixel', 'fitness', 'true label', 'predicted label', 'prior confidence in true label', 'post confidence true label', 'confidence wrong label']
        writer_bests.writerow(header_best)

        # Start timer
        start_time = time.time()

        # Attack
        for i in range(n_samples):
            print("\n--- Image ", i)
            img_idx = int(images_idx[i])
            img = x_test[img_idx]
            label = images_labels[i]
            samples.append(img_idx)

            if img_idx not in dict_success_rate:
                dict_success_rate[img_idx] = []
                dict_adv_prob1[img_idx] = []
                dict_adv_prob2[img_idx] = []

            img_folder = f'{run_folder}/img_{i}'
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            best_fit, avg_fit, best_ind, suc, suc_act_total, n_covered_pixels = cmaes(img, label, model, bounds, centroid, sigma_, lambda_, mu, generations, img_folder, seeds[nruns-1])
            
            fig = plt.figure(num=1, clear=True)
            plt.plot(list(range(len(best_fit))), best_fit)
            plt.plot(list(range(len(avg_fit))), avg_fit)
            plt.xlabel('Geração')
            plt.ylabel('Fitness')
            plt.title('Fitness overtime')
            plt.legend(['best', 'average'])
            fig.savefig(f"{img_folder}/fitness_evolution_{i}")
            fig.clear()
            plt.close(fig)
            
            # Save how many pixels were successful
            file_npixels =  f'./results/{modelName}/number_pixels.csv'
            f_npixels = open(file_npixels, 'a')
            writer_npixels = csv.writer(f_npixels)
            writer_npixels.writerow(['cmaes', run, img_idx, i, suc])
            f_npixels.close()

            # Save number of covered pixels
            file_cover = f'{cmaes_folder}/covered_pixels.csv'
            f_cover = open(file_cover, 'a')
            writer_cover = csv.writer(f_cover)
            writer_cover.writerow([run, i, img_idx, n_covered_pixels])
            f_cover.close()

            # Save best individual
            predicted_label = np.argmax(best_ind.confidence)
            activation = np.max(best_ind.confidence)
            prior_confidence_true_label = images_df['confidence'].values[label]
            post_confidence_true_label = best_ind.confidence[label]

            if best_ind.success:
                post_confidence_wrong_label = activation
            else:
                post_confidence_wrong_label = 0

            writer_bests.writerow([i, img_idx, best_ind.phenotype, best_ind.fitness.values[0], label, predicted_label, prior_confidence_true_label, post_confidence_true_label, post_confidence_wrong_label])

            # Metrics for img
            # For success rate dataset
            if suc > 0: 
                suc_samples += 1
                adv_prob2 = (suc_act_total / suc)
            else:
                adv_prob2 = 0
            success_rate = suc / (pop_size * generations) # number of successful attacks / number of evaluations
            adv_prob1 = suc_act_total / (pop_size * generations)

            success_rate_per_img.append(success_rate)
            adv_prob_per_img.append([adv_prob1, adv_prob2])

            # Save img metrics
            folder_metrics_img = f'{cmaes_folder}/metrics_img'
            if not os.path.exists(folder_metrics_img):
                os.mkdir(folder_metrics_img)
            
            
            file_metrics_img = f'{folder_metrics_img}/img_{i}.csv'
            if not os.path.exists(file_metrics_img):
                f_img = open(file_metrics_img, 'w')
                header = ['run', 'success rate', 'adv probability']
                writer_metrics_img = csv.writer(f_img)
                writer_metrics_img.writerow(header)
                f_img.close()

            f_img = open(file_metrics_img, 'a')
            writer_metrics_img = csv.writer(f_img)
            writer_metrics_img.writerow([run, success_rate, [adv_prob1, adv_prob2]])
            f_img.close()

            dict_success_rate[img_idx].append(success_rate)
            dict_adv_prob1[img_idx].append(adv_prob1)
            dict_adv_prob2[img_idx].append(adv_prob2)

        f_bests.close()

        # Run metrics
        success_rate_dataset = suc_samples / n_samples
        success_rate_dataset_store.append(success_rate_dataset)

        # End timer and write elapsed time into csv
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        f_time = open(time_file, 'a')
        writer_time = csv.writer(f_time)
        writer_time.writerow([modelName, run, elapsed_time])
        f_time.close()
        
        # success_rate_per_img e adv_prob_per_img sao listas
        writer_metrics.writerow([run, success_rate_dataset, elapsed_time, success_rate_per_img, adv_prob_per_img])
        gc.collect()
    f_metrics.close()

    # Write mean metrics
    file_means = f'./results/{modelName}/metrics_mean.csv'
    f_means = open(file_means, 'a')
    writer_means = csv.writer(f_means)
    mean_suc_rate_dataset = sum(success_rate_dataset_store) / nruns
    writer_means.writerow(['cmaes', mean_suc_rate_dataset])
    f_means.close()

    mn += 1

    # Specify the file path for the CSV
    csv_file = f'{cmaes_folder}/parameters.csv'

    # Create a list of parameter names and their corresponding values
    parameter_names = ["nruns", "seeds", "n_samples", "samples", "pop_size (lambda)", "generations", "pixels", "centroid", "sigma", "mu"]
    parameter_values = [nruns, seeds, n_samples, samples, pop_size, generations, pixels, centroid, sigma_, mu]

    # Write parameters to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(parameter_names)
        writer.writerow(parameter_values)
    
    # Write mean metrics for each image
    # dict = {img_idx: [success rate run 1, ..., success rate nruns]}
    file_metrics_mean_img = f'./results/{modelName}/metrics_mean_img.csv'
    f_metrics_img = open(file_metrics_mean_img, 'a')
    writer_metrics_img = csv.writer(f_metrics_img)
    counter = 0
    for img_idx, success_rate in dict_success_rate.items():
        mean_success_rate = sum(success_rate) / nruns  # Calculate the mean
        mean_adv_prob1 = sum(dict_adv_prob1[img_idx]) / nruns
        mean_adv_prob2 = sum(dict_adv_prob2[img_idx]) / nruns
        writer_metrics_img.writerow([counter, img_idx, 'cmaes', mean_success_rate, [mean_adv_prob1, mean_adv_prob2]])
        counter += 1
    f_metrics_img.close()