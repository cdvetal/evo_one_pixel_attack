# Calculate distortion of each pixel attack


import pandas as pd
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from setup_cifar import CIFAR

def main():
    modelNames = ['regular', 'distilled']
    # abordagens = ['de', 'ga', 'ra', 'cmaes']
    abordagens = ['de', 'ga']
    #nruns = 10
    #n_imgs = 500
    nruns = 10
    n_imgs = 500

    # Load dataset
    data_cifar = CIFAR()
    x_test = data_cifar.test_data
    x_test = (x_test + 0.5) * 255

    for modelName in modelNames:
        path_model = f'./results/{modelName}'
        print(f"\n\n\t\t\tMODELO: {modelName}\n\n")

        for abordagem in abordagens:
            print(f"\n\t\tABORDAGEM: {abordagem}")
            for i_run in range(1, nruns+1):
                print(f"\n\tRUN: {i_run}\n")
                for i_img in range(n_imgs):
                    print(f"IMAGEM: {i_img}")
                        
                    # path the metrics_mean_img.csv - ficheiro q usamos para buscar o id
                    path_id = f'{path_model}/metrics_mean_img.csv'
                    ids = pd.read_csv(path_id)["img_idx"].tolist()
                    
                    # buscar o id da imagem atual
                    id_atual = ids[i_img + n_imgs * abordagens.index(abordagem)]

                    # imagem atual
                    img = x_test[id_atual]

                    # BUSCAR DADOS AO FICHEIRO
                    # aqui n sei pq quando fiz download os modelos nao apareceram, com os modelos é o primeiro path
                    file_success = f"{path_model}/{abordagem}/run_{i_run}/img_{i_img}/success_file.csv"
                    # this_path = path + f"/{modelName}/{abordagem}/run_{i_run + 1}/img_{i_img}/success_file.csv"
                    # this_path = path + f"//{abordagem}/run_{i_run + 1}/img_{i_img}/success_file.csv"
                    data = pd.read_csv(file_success)
                    print(data)
                    # listas das colunas
                    gen = data["gen"].tolist()
                    genotype = data["genotype"].tolist()
                    true_label = data["true label"].tolist()
                    predicted_label = data["predicted label"].tolist()
                    confidence = data.iloc[:, 3].tolist()

                        # CALCULAR A DIFERENÇA
                    dif = list()

                    for gt in genotype:
                        # gt é uma string - transformar em lista
                        gt = gt[1:-1]   # retirar os parentesis retos '[' e ']'
                        if ',' in gt:
                            gt = gt.split(',')
                        else:
                            gt = gt.split(" ")
                        while '' in gt:
                            gt.remove('')
                        for i in range(len(gt)):
                            gt[i] = int(gt[i])

                        # dif = (abs(x[2] - image_orig[x[0]][x[1]][0]) + abs(x[3] - image_orig[x[0]][x[1]][1]) + abs(x[4] - image_orig[x[0]][x[1]][2]))
                        d = (abs(gt[2] - img[gt[0]][gt[1]][0]) + abs(gt[3] - img[gt[0]][gt[1]][1]) + abs(gt[4] - img[gt[0]][gt[1]][2]))
                        dif.append(d)   # colocar na lista de diferentes

                    # Testar q os comprimentos estao todos iguais
                    #print(f"gen: {len(gen)}\ngenotype: {len(genotype)}\ntrue label: {len(true_label)}\npredicted label: {len(predicted_label)}\nconfidence: {len(confidence)}\ndif: {len(dif)}")

                    d = {"gen": gen, "genotype": genotype, "true label": true_label, "predicted label": predicted_label, "confidence in wrong label": confidence, "dif": dif}
                    df = pd.DataFrame(d)
                    #print(df)
                    new_success_file = f"{path_model}/{abordagem}/run_{i_run}/img_{i_img}/new_success_file.csv"
                    df.to_csv(new_success_file, index = False) 
                    

if __name__ == "__main__":
    main()