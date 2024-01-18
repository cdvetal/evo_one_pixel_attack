import random
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import gc
from PIL import Image
import keras
from differential_evolution import evaluate
from helper import perturb_image 

def generate_random_individual(w, h): # um pixel modificado
  x = random.randint(0, w - 1)
  y = random.randint(0, h - 1)
  red = random.randint(0, 255)
  green = random.randint(0, 255)
  blue = random.randint(0, 255)
  pixel = np.array([x, y, red, green, blue]) # pixel = x,y,r,g,b
  return {'genotype': pixel, 'fitness': None, 'confidence': None, 'success': None}

def generate_initial_population(POPULATION_SIZE, w, h):
    for i in range(POPULATION_SIZE):
        yield generate_random_individual(w, h)

def trues_add(trues, pixel, suc, suc_act):
    x = list(pixel['genotype'])
    novo = 0

    if x not in trues:
        trues.append(x)
        suc += 1
        suc_act += np.max(pixel['confidence'])
        novo = 1

    return trues, suc, suc_act, novo

def with_random(img, true_class, model, pop_size, generations, bounds, folder_path, SEED):
    random.seed(SEED)
    dicio_total_pixels = {}
    # Boundaries
    w = bounds[0][1]
    h = bounds[1][1]

    NUMERO_DE_ALEATORIOS = generations * pop_size

    ## exemplo com N gerados aleatóriamente
    population = list(generate_initial_population(NUMERO_DE_ALEATORIOS, w, h))

    ################################

    # parte do modelo
    #true_class = y_test[image_id, 0]
    #prior_confidence = model.predict_one(img)[true_class]

    ### deixei de usar esta função, o que vem a seguir deve ser adaptado
    ### o atack succes só confirma um pixel de cada vez o que é estranho.
    #success = attack_success(pixel, x_test[image_id], true_class, model, verbose=True)

    # confiança do modelo antes do ataque
    #print('Prior confidence', prior_confidence)

    # faz perturbação de um array de pixels
    #resultados = perturb_image(population, img) # resultados é uma lista de imagens

    # File to storage success
    header = ['gen', 'genotype', 'true label', 'predicted label',' confidence in wrong label']
    file_suc = f'{folder_path}/success_file.csv'
    f_suc = open(file_suc, 'w')
    writer_suc = csv.writer(f_suc)
    writer_suc.writerow(header)

    verbose = False
    #i = 0
    suc = 0
    suc_act = 0
    num_parts = generations
    part_size = NUMERO_DE_ALEATORIOS // num_parts  # Calculate the size of each part

    for i in range(num_parts):
        start_idx = i * part_size
        end_idx = (i + 1) * part_size if i < num_parts - 1 else NUMERO_DE_ALEATORIOS
        
        # Slice the population to get the current part
        current_part = population[start_idx:end_idx]
        
        # Evaluate the current part
        evaluate(current_part, img, true_class, model, dicio_total_pixels)
    
    # Write entire population 
    # header_pop = ['genotype', 'fitness', 'success', 'confidence']
    # file_pop = f'{folder_path}/population.csv'
    # f_pop = open(file_pop, 'w')
    # writer_pop = csv.writer(f_pop)
    # writer_pop.writerow(header_pop)
    # #for m in range(len(population)):
    # for m in range(pop_size):
    #     ind = population[m]
    #     writer_pop.writerow([ind['genotype'], ind['fitness'], ind['success'], list(ind['confidence'])])
    # f_pop.close()

    # Write new success pixels found
    trues = list()
    for i in range(NUMERO_DE_ALEATORIOS):
        ind = population[i]
        predicted_class = np.argmax(ind['confidence'])
    #    suc_act += ind['confidence']
    #    suc = (predicted_class == true_class)
        activation = np.max(ind['confidence'])
        if predicted_class != true_class:
            trues, suc, suc_act, novo = trues_add(trues, ind, suc, suc_act)
            if novo:
                writer_suc.writerow([i, population[i]['genotype'], true_class, predicted_class, activation])

    # Close sucess
    f_suc.close()

    population.sort(key=lambda x: x['fitness'])
    best = population[-1]

    # copy from GA code
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
    del f_suc
    # del f_pop
    gc.collect()

    output_file = 'output.txt'
    f_output_file = open(output_file, 'w')
    print(dicio_total_pixels.keys(), file=f_output_file)
    return best, suc, suc_act, len(dicio_total_pixels)     # devolve o melhor individuo, a fitness desse, nº de sucesso e suc_act