from hmac import new
import random
from tabnanny import check
import numpy as np
from deap import base, creator, tools, cma
from setup_cifar import CIFAR, CIFARModel
import pandas as pd
import helper
from differential_evolution import attack_success, in_dicio_total, perturb_image
import tensorflow as tf
from math import log
import csv
import os
from PIL import Image
import gc
import math
import copy

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.get_logger().setLevel(tf.compat.v1.logging.ERROR) 

def set_phenotype(ind, bounds):
    aux = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    ind = check_bounds(ind, aux)
    ind.phenotype = [ind[i] for i in range(len(ind))]
    ind.phenotype = scale_phenotype(ind.phenotype, bounds)
    ind.phenotype = round_phenotype_to_int(ind.phenotype)
    ind.phenotype = check_bounds(ind.phenotype, bounds)
    return ind

def scale_phenotype(phenotype, bounds):
    for i in range(len(phenotype)):
        phenotype[i] = (bounds[i][1] + 1) * phenotype[i]
    return phenotype

def round_phenotype_to_int(phenotype):
    for i in range(len(phenotype)):
        if math.isnan(phenotype[i]):
            phenotype[i] = 0
        else:
            phenotype[i] = int(round(phenotype[i]))
    return phenotype

def check_bounds(individual, bounds):
    for i in range(len(individual)):
        if individual[i] > bounds[i][1]:
            individual[i] = bounds[i][1]
        if individual[i] < bounds[i][0]:
            individual[i] = bounds[i][0]
    return individual

def generate_random_individual(bounds): # um pixel modificado
    ind = creator.Individual() 

    genotype = []

    for j in range(len(bounds)):
        genotype.append(np.random.choice([( 1 / (bounds[j][1] + 1) )*i for i in range(bounds[j][1] + 1)]))
    
    genotype = np.array(genotype)

    ind.extend(genotype)  
    ind.fitness.values = (0,) 
    ind.success = None
    ind.confidence = None
    ind.phenotype = None

    return ind
    
def dicio_total_add(dicio, pixel):
    string = ''
    for i in pixel.phenotype:
        string += (str(i) + '_')
    dicio[string] = [pixel.fitness.values, pixel.confidence, pixel.success]

def evaluate(popul, image, true_class, model, dicio_total_pixels):
    phenotypes = []
    new_pop = []
    for i in range(len(popul)):
        gene = popul[i].phenotype
        # check if ind matches with an ind in dict
        r = in_dicio_total(dicio_total_pixels, gene)
        if r != None:
           # ind has already been evaluated
           popul[i].fitness.values = r[0]
           popul[i].confidence = r[1]
           popul[i].success = r[2]
        else:
           # ind has not been evaluated
           phenotypes.append(gene)
           new_pop.append(popul[i])
    if len(phenotypes) > 0:
        success, confidence_x = attack_success(np.array(phenotypes), image, true_class, model, verbose=False)
        fitness(new_pop, success, confidence_x, image, dicio_total_pixels, true_class) # [true_class]


def fitness(population, success_x, confidence_x, image_orig, dicio, true_class): # x é um array do tipo [x, y, r, g, b]
  # max success + 1/perturbation
    for ind in range(len(population)):
        x = population[ind].phenotype
        # f needs individual to be integer in bounds
        # f = 1.0 / ( (abs(x[2] - image_orig[x[0]][x[1]][0]) + abs(x[3] - image_orig[x[0]][x[1]][1]) + abs(x[4] - image_orig[x[0]][x[1]][2])) + 1) + 1 * int(success_x[ind]) + (1.0 / (1 + confidence_x[ind][true_class]))
        # f = 1 * int(success_x[ind]) + (1.0 / confidence_x[ind][true_class])
        t_difs = (abs(x[2] - image_orig[x[0]][x[1]][0]) + abs(x[3] - image_orig[x[0]][x[1]][1]) + abs(x[4] - image_orig[x[0]][x[1]][2]))
        f = 1.0 / ( t_difs + 1) + 1 * int(success_x[ind]) + (1.0 /(confidence_x[ind][true_class]+1))
        population[ind].fitness.values = (f,)
        population[ind].confidence = confidence_x[ind]
        population[ind].success = success_x[ind]
        # add to dict
        dicio_total_add(dicio, population[ind])

def dicio_trues_add(dicio, individuo):
    x = individuo.phenotype
    novo = 0 
    string = ''
    for i in x:
        string += (str(i) + '_')
    r = dicio.get(string)
    if r == None:
        dicio[string] = 'succ'
        novo = 1
    return dicio, novo

def cmaes(img, label, model, bounds, centroid, sigma_, lambda_, mu, generations, folder_path, seed):
    np.random.seed(seed)

    dicio_total_pixels = {}
    dicio_success = {}

    # Count success
    suc = 0
    suc_act_total = 0

    # File to storage success
    header = ['gen', 'phenotype', 'true label', 'predicted label',' confidence in wrong label']
    file_suc = f'{folder_path}/success_file.csv'
    f_suc = open(file_suc, 'w')
    writer_suc = csv.writer(f_suc)
    writer_suc.writerow(header)

    # File for evolution overview
    header = ['gen', 'best fitness', 'best individual', 'best confidence', 'best success', 'true label', 'predicted label', 'average fitness', 'std fitness', 'prediction']
    file_gen = f'{folder_path}/evolution_overview.csv'
    f_gen_info = open(file_gen, 'w')
    writer_gen_info = csv.writer(f_gen_info)
    writer_gen_info.writerow(header)

    # This folder holds generation files that have all individuals 
    gen_folder = f'{folder_path}/generations_files'
    if not os.path.exists(gen_folder):
        os.makedirs(gen_folder)

    # Storage
    best_fit = []
    avg_fit = []

    # Define the DEAP creator for the individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax, success=bool, confidence=list, phenotype=list)

    # Define the DEAP toolbox
    dicio_total_pixels = {}
    toolbox = base.Toolbox()
    toolbox.register("individual", generate_random_individual, bounds)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("evaluate", evaluate, img, label, models[0])

    # Run the CMA-ES algorithm for a specified number of generations
    gen = 0

    while gen < generations:
        if gen == 0:
            population = [generate_random_individual(bounds) for i in range(lambda_)]
            copy_pop = np.array(population).T
            centroid_ = np.mean(copy_pop, axis=1)
            strategy = cma.Strategy(centroid=centroid_, sigma=sigma_, lambda_=lambda_, mu=mu)
            toolbox.register("generate", strategy.generate, creator.Individual)
            toolbox.register("update", strategy.update)

            halloffame = tools.HallOfFame(1)

        else:
            population = toolbox.generate()
        population = [set_phenotype(ind, bounds) for ind in population]
        print(population)
        # Generation
        print("---------------------------")
        print("Generation ", gen)

        evaluate(population, img, label, model, dicio_total_pixels)
    
        halloffame.update(population)
        best_ind = halloffame[0]
        best_fit.append(best_ind.fitness.values[0])
        avg = sum([ind.fitness.values[0] for ind in population])/len(population)
        avg_fit.append(avg)

        print(f"\tFitness max: {best_ind.fitness.values[0]} \n\tFitness avg: {avg}")

        # Write for overview
        predicted_label = np.argmax(best_ind.confidence)
        activation = np.max(best_ind.confidence)
        writer_gen_info.writerow([gen, best_ind.fitness.values[0], best_ind.phenotype, activation, best_ind.success, label, predicted_label, avg_fit[0], np.std([ind.fitness.values[0] for ind in population]), list(best_ind.confidence)])

        # Write entire population 
        # header_pergen = ['phenotype', 'fitness', 'success', 'confidence']
        # file_pergen = f'{gen_folder}/gen0.csv'
        # f_pergen = open(file_pergen, 'w')
        # writer_pergen = csv.writer(f_pergen)
        # writer_pergen.writerow(header_pergen)
        for m in range(len(population)):
            ind = population[m]
            # writer_pergen.writerow([ind, ind.fitness.values[0], ind.success, list(ind.confidence)])
            if ind.success:
                dicio_success, novo = dicio_trues_add(dicio_success, ind)
                if novo:
                    suc += 1
                    predicted_label = np.argmax(ind.confidence)
                    activation = np.max(ind.confidence)
                    suc_act_total += activation
                    writer_suc.writerow([gen, ind.phenotype, label, predicted_label, activation])
        #f_pergen.close()

        try:
            toolbox.update(population)
        except np.linalg.LinAlgError as e:
            print("LinAlgError: Eigenvalues did not converge")
            avg_fit = avg_fit[0:-1]
            best_fit = avg_fit[0:-1]
            break
        gen += 1
    
    # Close overview and success
    f_gen_info.close()
    f_suc.close()

    # Save original image and image perturbed by best individual
    perturbed_image = perturb_image(np.array(best_ind.phenotype), img)[0]

    # Save perturbed image
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    perturbed_pil_image = Image.fromarray(perturbed_image)
    scaled_perturbed_pil_image = perturbed_pil_image.resize((320, 320))
    scaled_perturbed_pil_image.save(f'{folder_path}/best_perturbed.png')

    # Save original image
    img = np.clip(img, 0, 255).astype(np.uint8)
    original_pil_image = Image.fromarray(img)
    scaled_original_pil_image = original_pil_image.resize((320, 320))
    scaled_original_pil_image.save(f'{folder_path}/original_image.png')

    del population
    del f_gen_info
    del f_suc
    #del f_pergen
    gc.collect()

    return best_fit, avg_fit, best_ind, suc, suc_act_total, len(dicio_total_pixels)