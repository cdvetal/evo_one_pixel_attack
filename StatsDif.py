# Calculate distortion statis per image in a run

import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np

def main():
    models = ['regular', 'distilled']
    abordagens = ['de', 'ga', 'ra', 'cmaes']
    runs = 10
    n_imgs = 500
    path = './results'

    for model in models:
        print(f"\n\n\t\t\tMODELO: {model}\n\n")
        for abordagem in abordagens:
            print(f"\n\t\tABORDAGEM: {abordagem}")
            for i_run in range(runs):
                print(f"\n\tRUN: {i_run}\n")
                media_list = []
                mini_list = []
                maxi_list = []
                desvio_padrao_list = []
                n_suc_list = []
                for i_img in range(n_imgs):
                    print(f"IMAGEM: {i_img}")
                        
                    # path the metrics_mean_img.csv - ficheiro q usamos para buscar o id
                    path_id = f'{path}/{model}/metrics_mean_img.csv'
                    ids = pd.read_csv(path_id)["img_idx"].tolist()

                    # BUSCAR DADOS AO FICHEIRO
                    # aqui n sei pq quando fiz download os modelos nao apareceram, com os modelos é o primeiro path
                    this_path = path + f"/{model}/{abordagem}/run_{i_run + 1}/img_{i_img}/"
                    run_path = path + f"/{model}/{abordagem}/run_{i_run + 1}/"
                    # this_path = path + f"//{abordagem}/run_{i_run + 1}/"
                    data = pd.read_csv(this_path + "new_success_file.csv")

                    # listas das colunas
                    dif = data["dif"].tolist()
                    
                    # STATISTICS
                    if len(dif) == 0:
                        mini = 0
                        maxi = 0
                        desvio_padrao = 0
                        media = 0
                    else:
                        mini = min(dif)
                        maxi = max(dif)
                        desvio_padrao = statistics.pstdev(dif)
                        media = statistics.mean(dif)
                    n_suc = len(dif)
                    media_list.append(media)
                    mini_list.append(mini)
                    maxi_list.append(maxi)
                    desvio_padrao_list.append(desvio_padrao)
                    n_suc_list.append(n_suc)
                    

                d = {"img_counter":np.arange(len(media_list)), "mean": media_list, "min": mini_list, "max": maxi_list, "standard deviation": desvio_padrao_list, "number success pixeis": n_suc_list}
                df = pd.DataFrame(d)
                #print(df)
                dif_file = f'{run_path}/difStats.csv'
                df.to_csv(dif_file, index = False)
                    #print(df)
                    #df.to_csv(this_path, index = False)

if __name__ == "__main__":
    main()