import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import helper
from PIL import Image
import gc
import tensorflow as tf
import keras
from helper import perturb_image

def attack_success(x, img, true_class, model, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    perturbed = perturb_image(x, img) # x pode ser 1 pixel ou lista de pixeis
    perturbed = (perturbed / 255) - 0.5 # para os modelos do carlini
    
    list_confidence = model.predict(perturbed) # leak! com problemas [ [0 for i in range(10)] for k in perturbed] #fake eval all 0 for each class #model.predict(perturbed)
    
    # Apply softmax to the logits
    # Calculate the maximum logits for each sample
    max_logits = np.max(list_confidence, axis=1, keepdims=True)

    # Subtract the maximum logits from the original logits to improve numerical stability
    shifted_logits = list_confidence - max_logits

    # Apply softmax to the shifted logits for all predictions
    exp_logits = np.exp(shifted_logits)
    list_confidence = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    _ = gc.collect() 
    keras.backend.clear_session()


    list_success = []

    for confidence in list_confidence:
      predicted_class = np.argmax(confidence)
      if (predicted_class != true_class):
          list_success.append(True)
      else:
          list_success.append(False)

    return list_success, list_confidence

def generate_random_individual(bounds):
  x = np.random.randint(bounds[0][0], bounds[0][1])
  y = np.random.randint(bounds[1][0], bounds[1][1])
  red = np.random.randint(bounds[2][0], bounds[2][1])
  green = np.random.randint(bounds[3][0], bounds[3][1])
  blue = np.random.randint(bounds[4][0], bounds[4][1])
  pixel = np.array([x, y, red, green, blue]) # pixel = x,y,r,g,b
  return {'genotype': pixel, 'fitness': None, 'confidence': None, 'success': None}


def generate_random_individual_uniform_gaussian(bounds):
    x = np.random.randint(bounds[0][0], bounds[0][1])
    y = np.random.randint(bounds[1][0], bounds[1][1])
    red = int(np.random.normal(128, 127))  # Generate RGB values using a Gaussian distribution
    green = int(np.random.normal(128, 127))
    blue = int(np.random.normal(128, 127))
    pixel = np.array([x, y, red, green, blue])
    return {'genotype': pixel, 'fitness': None, 'confidence': None, 'success': None}

def generate_initial_population_uniform_gaussian(POPULATION_SIZE, bounds):
    population = []
    for i in range(POPULATION_SIZE):
        ind = generate_random_individual_uniform_gaussian(bounds)
        population.append(ind)
    return np.array(population)

def in_dicio_total(dicio, gene):
    string = ''
    for i in gene:
        string += (str(i) + '_')
    r = dicio.get(string)
    if r == None:
      return None
    else:
       return dicio[string]
    
def dicio_total_add(dicio, pixel):
    string = ''
    for i in pixel['genotype']:
        string += (str(i) + '_')
    dicio[string] = [pixel['fitness'], pixel['confidence'], pixel['success']]

def evaluate(popul, image, true_class, model, dicio_total_pixels):
    genotypes = []
    new_pop = []
    for i in range(len(popul)):
        gene = popul[i]['genotype']
        # check if ind matches with an ind in dict
        r = in_dicio_total(dicio_total_pixels, gene)
        if r != None:
           # ind has already been evaluated
           popul[i]['fitness'] = r[0]
           popul[i]['confidence'] = r[1]
           popul[i]['success'] = r[2]
        else:
           # ind has not been evaluated
           genotypes.append(popul[i]['genotype'])
           new_pop.append(popul[i])
    if len(genotypes) > 0:
        success, confidence_x = attack_success(np.array(genotypes), image, true_class, model, verbose=False)
        fitness(new_pop, success, confidence_x, image, dicio_total_pixels, true_class) # [true_class]

def fitness(population, success_x, confidence_x, image_orig, dicio, true_label): # x é um array do tipo [x, y, r, g, b]
  # max success + 1/perturbation
  for ind in range(len(population)):
    x = population[ind]['genotype']
    # f = 1.0 / ( (abs(x[2] - image_orig[x[0]][x[1]][0]) + abs(x[3] - image_orig[x[0]][x[1]][1]) + abs(x[4] - image_orig[x[0]][x[1]][2])) + 1) + 1 * int(success_x[ind]) + (1.0 /confidence_x[ind][true_label])
    # f = 1 * int(success_x[ind]) + (1.0 /confidence_x[ind][true_label])
    t_difs = (abs(x[2] - image_orig[x[0]][x[1]][0]) + abs(x[3] - image_orig[x[0]][x[1]][1]) + abs(x[4] - image_orig[x[0]][x[1]][2]))
    f = 1.0 / ( t_difs + 1) + 1 * int(success_x[ind]) + (1.0 /(confidence_x[ind][true_label]+1))
    population[ind]['fitness'] = f
    population[ind]['confidence'] = confidence_x[ind]
    population[ind]['success'] = success_x[ind]
    # add to dict
    dicio_total_add(dicio, population[ind])

def mutation(a, b, c, F):
    new_pixel = []
    for i in range(len(a['genotype'])):
        new_pixel.append(a['genotype'][i] + F * (b['genotype'][i] - c['genotype'][i]))
    new_pixel = [int(x) for x in new_pixel]
    return {'genotype': np.array(new_pixel), 'fitness': None, 'confidence': None, 'success': None}

def check_bounds(mutated, bounds):
    mutated_pixel = mutated['genotype']
    mutated_bound = [np.clip(mutated_pixel[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return {'genotype': np.array(mutated_bound), 'fitness': None, 'confidence': None, 'success': None}

def dicio_trues_add(dicio, individuo):
    x = individuo['genotype']
    novo = 0 
    string = ''
    for i in x:
        string += (str(i) + '_')
    r = dicio.get(string)
    if r == None:
        dicio[string] = 'succ'
        novo = 1
    return dicio, novo

def differential_evolution_no_cx(img, true_label, model, popsize, generations, mut, bounds, folder_path, seed):
    np.random.seed(seed)
    dicio_total_pixels = {}
    dicio_success = {}
    # Count success
    suc = 0
    suc_act_total = 0

    # File to storage success
    header = ['gen', 'genotype', 'true label', 'predicted label',' confidence in wrong label']
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
    
    # Generation 0
    print("---------------------------")
    print("Generation 0")

    population = generate_initial_population_uniform_gaussian(popsize, bounds)
    evaluate(population, img, true_label, model, dicio_total_pixels)
    # population.sort(key=lambda x: x['fitness'], reverse=True)
    population = np.array(sorted(population, key=lambda x: x['fitness'], reverse=True))
    best = population[0]
    best_fit.append(best['fitness'])
    avg = sum([ind['fitness'] for ind in population])/popsize
    avg_fit.append(avg)

    print(f"\tFitness max: {best['fitness']} \n\tFitness avg: {avg}")

    # Write for overview
    predicted_label = np.argmax(best['confidence'])
    activation = np.max(best['confidence'])

    writer_gen_info.writerow([0, best_fit[0], best['genotype'], activation, best['success'], true_label, predicted_label, avg_fit[0], np.std([ind['fitness'] for ind in population]), list(best['confidence'])])
    
    # Write entire population 
    # header_pergen = ['genotype', 'fitness', 'success', 'confidence']
    # file_pergen = f'{gen_folder}/gen0.csv'
    # f_pergen = open(file_pergen, 'w')
    # writer_pergen = csv.writer(f_pergen)
    # writer_pergen.writerow(header_pergen)
    for m in range(len(population)):
        ind = population[m]
       # writer_pergen.writerow([ind['genotype'], ind['fitness'], ind['success'], list(ind['confidence'])])
        if ind['success']:
            dicio_success, novo = dicio_trues_add(dicio_success, ind)
            if novo:
                suc += 1
                predicted_label = np.argmax(ind['confidence'])
                activation = np.max(ind['confidence'])
                suc_act_total += activation
                writer_suc.writerow([0, ind['genotype'], true_label, predicted_label, activation])
    # f_pergen.close()

    for i in range(1, generations):
        print("Generation ", i)

        mutants = []
        for j in range(popsize):
            candidates = [candidate for candidate in range(popsize) if candidate != j]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            mutated = mutation(population[a], population[b], population[c], mut) 
            mutated = check_bounds(mutated, bounds)
            mutants.append(mutated)

        evaluate(mutants, img, true_label, model, dicio_total_pixels)
        
        for j in range(popsize):
            if population[j]['fitness'] < mutants[j]['fitness']:
                population[j] = mutants[j]
                if mutants[j]['success']:
                    dicio_success, novo = dicio_trues_add(dicio_success, mutants[j])
                    if novo:
                        suc += 1
                        predicted_label = np.argmax(mutants[j]['confidence'])
                        activation = np.max(mutants[j]['confidence'])
                        suc_act_total += activation
                        writer_suc.writerow([i, mutants[j]['genotype'], true_label, predicted_label, activation])
        
        # Update best, sort (reverse) and storage 
        # population.sort(key=lambda x: x['fitness'], reverse=True)
        population = np.array(sorted(population, key=lambda x: x['fitness'], reverse=True))
        best = population[0]
        best_fit.append(best['fitness'])
        avg = sum([ind['fitness'] for ind in population])/popsize
        avg_fit.append(avg)

        print(f"\tFitness max: {best['fitness']} \n\tFitness avg: {avg}")

        # Write for overview
        predicted_label = np.argmax(best['confidence'])
        activation = np.max(best['confidence'])
        writer_gen_info.writerow([i, best['fitness'], best['genotype'], activation, best['success'], true_label, predicted_label, avg_fit[i], np.std([ind['fitness'] for ind in population]), list(best['confidence'])])

        # Write entire population 
        # header_pergen = ['genotype', 'fitness', 'success', 'confidence']
        # file_pergen = f'{gen_folder}/gen{i}.csv'
        # f_pergen = open(file_pergen, 'w')
        # writer_pergen = csv.writer(f_pergen)
        # writer_pergen.writerow(header_pergen)
        # for m in range(len(population)):
           # ind = population[m]
            # writer_pergen.writerow([ind['genotype'], ind['fitness'], ind['success'], list(ind['confidence'])])
        # f_pergen.close()

    # Close overview and success
    f_gen_info.close()
    f_suc.close()

    # Save original image and image perturbed by best individual
    perturbed_image = perturb_image(np.array(best['genotype']), img)[0]

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
    # del f_pergen
    gc.collect()

    return best_fit, avg_fit, best, suc, suc_act_total, len(dicio_total_pixels)
