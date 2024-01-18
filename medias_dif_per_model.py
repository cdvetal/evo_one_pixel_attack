import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    models = ['regular']
    abordagens = ['de', 'ga', 'ra', 'cmaes']
    # abordagens = ['ra']
    # runs = 10
    # n_imgs = 500
    runs = 10
    n_imgs = 500

    path_results = './our_results'
    
    for model in models:
        print(f"\n\n\t\t\tMODELO: {model}\n\n")
        path = f'{path_results}/{model}'
        medias_list = []
        medis_min_list = []
        std_medias_list = []
        std_min_list = []
        for abordagem in abordagens:
            print(f"\n\t\tABORDAGEM: {abordagem}")
            mean_mean_per_run = []
            mean_min_per_run = []
            for i_run in range(runs):
                print(f"\n\tRUN: {i_run}\n")

                # BUSCAR DADOS AO FICHEIRO
                this_path = f"{path}/{abordagem}/run_{i_run + 1}/difStats.csv"
                data = pd.read_csv(this_path)
                data = data[data["mean"] > 0]
                # Media de imagens por run
                mean_mean_per_run.append(sum(data["mean"])/len(data))
                mean_min_per_run.append(sum(data["min"])/len(data))
            # Calcular media de todas as runs de uma abordagem
            media = np.mean(mean_mean_per_run)
            std_media = np.std(mean_mean_per_run)
            media_min = np.mean(mean_min_per_run)
            std_min = np.std(mean_min_per_run)
            # Guardar na lista de medias por abordagem
            medias_list.append(media)
            medis_min_list.append(media_min)
            std_medias_list.append(std_media)
            std_min_list.append(std_min)

        d = {"abordagem": abordagens, "mean das means": medias_list, "mean do minimo": medis_min_list, "std das means": std_medias_list, "std do min": std_min_list}
        df = pd.DataFrame(d)
        df.to_csv(path + "/difMeans.csv", index=False) # index = True
        #print(df)
        #df.to_csv(this_path, index = False)

if __name__ == "__main__":
    main()