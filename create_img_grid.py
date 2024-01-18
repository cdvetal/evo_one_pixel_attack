import tensorflow as tf
import matplotlib.pyplot as plt
from setup_cifar import CIFAR, CIFARModel
from differential_evolution import perturb_image
from PIL import Image
import numpy as np
import pandas as pd
import os
import ast

modelName = 'regular'
abordagens = ['de', 'ga', 'cmaes']
results_path = './results'
n_samples = 500
nruns = 10

succ_samples = []

path_id = f'{results_path}/{modelName}/metrics_mean_img.csv'
ids = pd.read_csv(path_id)["img_idx"].tolist()

# Load dataset
data = CIFAR()
x_train = data.train_data
x_test = data.test_data
x_train = (x_train + 0.5) * 255
x_test= (x_test + 0.5) * 255

list_ = [2, 5, 10, 13, 16, 17, 24, 30, 31, 38, 42, 43, 53, 54, 55, 59, 62, 63, 68, 72, 73, 75, 77, 78, 82, 83, 84, 85, 86, 91, 92, 94, 95, 99, 101, 103, 104, 105, 106, 112, 114, 115, 118, 121, 122, 123, 125, 126, 130, 133, 140, 143, 147, 152, 153, 154, 155, 157, 161, 164, 170, 172, 173, 177, 178, 183, 187, 188, 190, 192, 193, 194, 195, 199, 202, 203, 205, 210, 212, 213, 214, 215, 216, 217, 222, 223, 226, 228, 230, 232, 233, 234, 235, 236, 239, 241, 243, 250, 251, 252, 253, 255, 263, 264, 268, 271, 274, 280, 282, 283, 286, 293, 294, 296, 299, 302, 303, 304, 306, 307, 308, 309, 310, 311, 313, 322, 323, 326, 327, 331, 333, 335, 340, 342, 343, 344, 345, 349, 353, 357, 358, 367, 373, 374, 375, 378, 384, 388, 392, 395, 398, 400, 401, 403, 405, 410, 411, 413, 415, 422, 423, 429, 432, 437, 439, 442, 443, 444, 445, 448, 449, 452, 455, 458, 459, 463, 465, 468, 470, 472, 473, 475, 478, 480, 482, 483, 484, 485, 488, 490, 493, 494, 497, 498]
for sample in range(n_samples):
    failed_abordagens = []
    runs = []
    for abordagem in abordagens:
        file = f"{results_path}/{modelName}/{abordagem}/metrics_img/img_{sample}.csv"
        df = pd.read_csv(file)

        # Check if all values in the second column (success rate) are 0
        if (df['success rate'] == 0).all():
            failed_abordagens.append(abordagem)
        
        else: 
             # Get the value in the "run" column when success rate is not 0
            run_value = df.loc[df['success rate'] != 0, 'run'].values[0]
            runs.append(run_value)

    if len(failed_abordagens) == 0 and sample in list_:
        succ_samples.append({'sample': sample, 'id': ids[sample], 'runs': runs})



# Open a text file for writing
with open(f'./images_{modelName}/images_output.txt', 'w') as output_file:
    for img in succ_samples:
        img_counter = img['sample']
        runs = img['runs']
        idx = img['id']
        original_image = x_test[idx]

        file = f"{results_path}/{modelName}/images_to_attack_idx.csv"
        conf = pd.read_csv(file)
        row = conf[conf['image id'] == idx]
        act = row['confidence'].values
        
        # Redirect print to the text file
        print(' ', file=output_file)
        print(f"original image counter {img_counter}, index {idx}, confidence in true {act}", file=output_file)

        path_save_img = f'./images_{modelName}/img_{img_counter}'
        if not os.path.exists(path_save_img):
            os.mkdir(path_save_img)
        for i in range(len(abordagens)):
            abordagem = abordagens[i]
            run = runs[i]
            min_value = 255*3
            success_file = f"{results_path}/{modelName}/{abordagem}/run_{run}/img_{img_counter}/success_file.csv"
            new_success_file = f"{results_path}/{modelName}/{abordagem}/run_{run}/img_{img_counter}/new_success_file.csv"
            
            # find pixel with min distortion in new_success_file
            df = pd.read_csv(new_success_file)
            df2 = pd.read_csv(success_file)
            min_index = df['dif'].idxmin()
            dist = df.loc[min_index, 'dif']
            pixel = df.loc[min_index, 'genotype']
            pixel = pixel[1:-1]   # remove '[' and ']'
            if ',' in pixel:
                pixel = pixel.split(',')
            else:
                pixel = pixel.split(" ")
            while '' in pixel:
                pixel.remove('')
            for i in range(len(pixel)):
                pixel[i] = int(pixel[i])

            act = df2.loc[min_index, ' confidence in wrong label']
            true_label = df2.loc[min_index, 'true label']
            pred_label = df2.loc[min_index, 'predicted label']
            
            x = pixel[0]
            y = pixel[1]
            rgb = original_image[x, y]
            original_pixel = np.array([x, y, rgb[0], rgb[1], rgb[2]])

            # Redirect print to the text file
           # print(abordagem, file=output_file)
            print(f"{abordagem}, label {true_label} -> {pred_label}, confidence in wrong {act}, distortion {dist}, pixel {pixel}, original pixel {original_pixel}", file=output_file)
            
            #print(original_pixel, file=output_file)
            
            original_image_ = perturb_image(np.array(original_pixel), original_image.copy())[0]
            original_image_ = np.clip(original_image_, 0, 255).astype(np.uint8)
            original_image_ = Image.fromarray(original_image_)
            original_image_ = original_image_.resize((320, 320))
            original_image_.save(f"./images_{modelName}/img_{img_counter}/{modelName}_original_{img_counter}.png")

            show_pixel = np.array([x, y, 255, 0, 0])

            show_image = perturb_image(np.array(show_pixel), original_image.copy())[0]
            show_image = np.clip(show_image, 0, 255).astype(np.uint8)
            show_image = Image.fromarray(show_image)
            show_image = show_image.resize((320, 320))
            show_image.save(f'./images_{modelName}/img_{img_counter}/{modelName}_where_{img_counter}_{abordagem}.png')

            perturbed_image = perturb_image(np.array(pixel), original_image.copy())[0]
            perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
            perturbed_image = Image.fromarray(perturbed_image)
            perturbed_image = perturbed_image.resize((320, 320))
            perturbed_image.save(f'./images_{modelName}/img_{img_counter}/{modelName}_{img_counter}_{abordagem}.png')
