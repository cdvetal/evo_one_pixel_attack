# Computação Evolucionária

import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
#import cv2
import math
from skimage import io
import helper
import os
import gc
import keras
from PIL import Image
from differential_evolution import evaluate
from helper import perturb_image

def predict_classes(xs, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def generate_random_individual(w, h): # um pixel modificado
  #image = img
  #print(i + 1)
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

def mapping(genotype, image): # genotype = x,y,r,g,b
  image = perturb_image(np.array(genotype), image)
  helper.plot_image(image)


def choose_indiv(population, TOURNAMENT):     # cópia do 1º projeto - inalterado
    pool = random.sample(population, TOURNAMENT)  # escolher aleatoriamente TOURNAMENT pixeis
    pool.sort(key=lambda i: i['fitness'])         # organizar de acordo com o fitness
    return copy.deepcopy(pool[-1])

def crossover(p1, p2):
  genotype = []

  cut_point = random.randint(1, 4)  # nao faz sentido o cutpoint ser no 0 ou no 4 (1 a 4 inclusive)

  for i in range(0, cut_point):
    genotype.append(p1['genotype'][i])
  for i in range(cut_point, 5):
    genotype.append(p2['genotype'][i])

  return {'genotype': genotype, 'fitness': None, 'confidence': None, 'success': None}

# Funções de mutação para cada gene
def mutate_por_gene(p, w, h, PROB_MUTATION):
  p = copy.deepcopy(p)
  p['fitness'] = None

  for i in range(5):
    if random.random() > PROB_MUTATION:
      if i == 0:        # posiçao 0 -> x
        p['genotype'][0] = random.randint(0, w - 1)
      elif i == 1:      # posiçao 1 -> y
        p['genotype'][1] = random.randint(0, h - 1)
      else:                 # valores de red, green e blue
        p['genotype'][i] = random.randint(0, 255)

  return p

def mutate_por_gene_gauss(p, desvio, w, h, PROB_MUTATION):
  p = copy.deepcopy(p)
  p['fitness'] = None

  x1 = random.random()
  x2 = random.random()
  y1 = math.sqrt(-2.0 * math.log(x1)) * math.cos(2.0 * math.pi * x2)


  for i in range(5):
    gene = int(y1 * desvio + p['genotype'][i])

    if random.random() < PROB_MUTATION:
      if i == 0:
        if gene > w - 1:
          gene = w - 1
        elif gene < 0:
          gene = 0
      elif i == 1:
        if gene > h - 1:
          gene = h - 1
        elif gene < 0:
          gene = 0
      else:
        if gene > 255:
          gene = 255
        elif gene < 0:
          gene = 0
      p['genotype'][i] = gene

  return p

# u ficheiro por populaçao
def infos_populacao_fich_v2(populacao, it):
    with open("/content/drive/MyDrive/UNI/Bolsa_dados/Populaçoes/populacoes_individuos_" + str(it), "w") as f:
      writer = csv.writer(f)
      writer.writerow(['index', 'fitness', 'confidence', 'success', 'genotype:'])
      for i in range(len(populacao)):
        writer.writerow([i, populacao[i]['fitness'], populacao[i]['confidence'], populacao[i]['success'], populacao[i]['genotype']])
      f.close()

def dicio_trues_add(dicio, gene, soma_dif, image_orig, suc, suc_act):
  x = gene['genotype']
  novo = 0 
  dif = (abs(x[2] - image_orig[x[0]][x[1]][0]) + abs(x[3] - image_orig[x[0]][x[1]][1]) + abs(x[4] - image_orig[x[0]][x[1]][2])) / 3
  string = ''
  for i in x:
      string += (str(i) + '_')

  r = dicio.get(string)
  if r == None:
    dicio[string] = dif
    suc += 1
    suc_act += np.max(gene['confidence'])
    novo = 1

  return dicio, soma_dif, suc, suc_act, novo

def genetic_algorithm(image, true_class, model, POPULATION_SIZE, NUMBER_OF_ITERATIONS, PROB_MUTATION, PROB_CROSSOVER, TOURNAMENT, bounds, folder_path, SEED):
    random.seed(SEED)
    dicio_total_pixels = {}
    # Boundaries
    w = bounds[0][1]
    h = bounds[1][1]

    # Count success
    suc = 0
    suc_act = 0
    
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
    x = []
    x.extend(range(0, NUMBER_OF_ITERATIONS)) # array com os primeiros N inteiros
    
    # Create a initial population randomly
    population = list(generate_initial_population(POPULATION_SIZE, w, h))
    dicio_trues = dict()
    it = 0
    soma_dif = 0

    # Evaluate how good the individuals are (problem dependent)
    for it in range(it, NUMBER_OF_ITERATIONS):
        
        #o filtra_por novos deve ser um funcao que filtra os elementos da populacao que nao foram ainda vistos.
        # evaluate(filtra_por_novos(population), image, true_class, model)
        evaluate(population, image, true_class, model, dicio_total_pixels)

        population.sort(key=lambda x: x['fitness'])
        best = population[-1]
        best_fit.append(best['fitness'])
      ## avaliar se é adversarial depois da avaliaçao
        for ni in population:
          if ni['success'] == True:
            dicio_trues, soma_dif, suc, suc_act, novo = dicio_trues_add(dicio_trues, ni, soma_dif, image, suc, suc_act)
            if novo:
              predicted_label = np.argmax(ni['confidence'])
              activation = np.max(ni['confidence'])
              writer_suc.writerow([it, ni['genotype'], true_class, predicted_label, activation])


        # Colocar o best e a média nesta iteração

        #bests.append(best)
        print("Best at", it, best)

        # Write for overview
        predicted_label = np.argmax(best['confidence'])
        activation = np.max(best['confidence'])

        avg = sum([ind['fitness'] for ind in population])/POPULATION_SIZE
        avg_fit.append(avg)

        # informaçao desta geraçao
        writer_gen_info.writerow([it, best['fitness'], best['genotype'], activation, best['success'], true_class, predicted_label, avg_fit[it], np.std([ind['fitness'] for ind in population]), list(best['confidence'])])
        #writer_gen_info.writerow([it, best['fitness'], best['genotype'], activation, best['success'], true_class, predicted_label, avg_fit[it], np.std([ind['fitness'] for ind in population]), best['confidence']])

        # Write entire population 
        # header_pergen = ['genotype', 'fitness', 'success', 'confidence']
        # file_pergen = f'{gen_folder}/gen{it}.csv'
        # f_pergen = open(file_pergen, 'w')
        # writer_pergen = csv.writer(f_pergen)
        # writer_pergen.writerow(header_pergen)
        # for m in range(len(population)):
        #     ind = population[m]
        #     writer_pergen.writerow([ind['genotype'], ind['fitness'], ind['success'], list(ind['confidence'])])
        #     #writer_pergen.writerow([ind['genotype'], ind['fitness'], ind['success'], ind['confidence']])
        # f_pergen.close()

        # elitismo
        new_population = [best]
        #print("Populaçao inicial", population)
        ###### Operadores de variaçao e seleçao de descendentes 
        while len(new_population) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:    # random.random() -> probability:[0.0 ; 1.0[
                # Parent Selection
                p1 = choose_indiv(population, TOURNAMENT)
                p2 = choose_indiv(population, TOURNAMENT)
                # nao fazer crossover quando p1 e p2 sao iguais
                while(np.array_equal(p2['genotype'], p1['genotype'])):
                   p2 = choose_indiv(population, TOURNAMENT)
                # Recombination
                ni = crossover(p1, p2)
                #evaluate(ni, image, true_class, model)
            else:
                ni = choose_indiv(population, TOURNAMENT)
            # Mutation
                        # mutacao por genes - funçoes mutate_por_gene() e mutate_por_gene_gauss()
            #mutate_por_gene(ni, w, h, PROB_MUTATION)
            ni = mutate_por_gene_gauss(ni, 3, w, h, PROB_MUTATION)
            #evaluate([ni], image, true_class, model)
            
            new_population.append(copy.deepcopy(ni)) # para garantir
        population = new_population
        
    print("Final: ", best)
    bestie = perturb_image(np.array(best['genotype']), image)
    # helper.plot_image(bestie)
    lista_trues = list(dicio_trues.keys())
    print("Trues: ", lista_trues)
    print("Pixeis encontrados: ", len(lista_trues))    # convem q este valor seja igual ao 'suc' (success - numero de bem sucedidos)

    # diferença media dos pixeis encontrados e o pixel original
    for i in dicio_trues.values():
      soma_dif += i
    if len(lista_trues) != 0:
      media_difs = soma_dif  / len(lista_trues)

    # Close overview and sucess
    f_gen_info.close()
    f_suc.close()

    perturbed_image = perturb_image(np.array(best['genotype']), image)[0]
    # Save perturbed image
    perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
    perturbed_pil_image = Image.fromarray(perturbed_image)
    scaled_perturbed_pil_image = perturbed_pil_image.resize((320, 320))
    scaled_perturbed_pil_image.save(f'{folder_path}/best_perturbed.png')

    # Save original image
    image = np.clip(image, 0, 255).astype(np.uint8)
    original_pil_image = Image.fromarray(image)
    scaled_original_pil_image = original_pil_image.resize((320, 320))
    scaled_original_pil_image.save(f'{folder_path}/original_image.png')

    del population
    del f_gen_info
    del f_suc
    # del f_pergen
    gc.collect()

    return best_fit, avg_fit, best, suc, suc_act, len(dicio_total_pixels)