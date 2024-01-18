import pandas as pd
import numpy as np

modelNames = ["regular"]
abordagens = ["ra"]
results_path = './results'
nruns = 10
n_samples = 500
pop_size = 400

# Create a text file for output
output_file_path = "./results/output.txt"
with open(output_file_path, "w") as output_file:

    # Analyse covered pixels
    print("---------- Mean number of covered pixels ----------", file=output_file)
    # mean das runs por imagem e depois mean das imagens
    for modelName in modelNames:
        print(modelName, file=output_file)
        for abordagem in abordagens:
            print(abordagem)
            file = f"{results_path}/{modelName}/{abordagem}/covered_pixels.csv"
            data = pd.read_csv(file)
            result = data.groupby(['run'])['number of covered pixels'].mean().reset_index()
            print(result['number of covered pixels'], file=output_file)
            print(abordagem, np.mean(result['number of covered pixels']), np.std(result['number of covered pixels']), file=output_file)

    # Mean adv per img
    print("---------- Mean quantity of adversarial images found for an original image ----------", file=output_file)
    # mean das imagens por run e depois mean das runs
    for modelName in modelNames:
        model_path = f'{results_path}/{modelName}'

        for abordagem in abordagens:
            print(modelName, " - ", abordagem, file=output_file)
            mean_runs = []
            for run in range(1, 1 + nruns):    
                adv_per_image = []

                for img in range(n_samples):
                    success_data = pd.read_csv(f'{model_path}/{abordagem}/run_{run}/img_{img}/success_file.csv')
                    n_advs = len(success_data)
                    adv_per_image.append(n_advs)

                mean = np.mean(adv_per_image)
                print(mean, file=output_file)
                mean_runs.append(mean)
            print("mean runs ", np.mean(mean_runs), np.std(mean_runs), file=output_file)

    # Nevals for first success
    print("---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file)
    print("skiping images with no success", file=output_file)
    # mean das imagens por run e depois mean das runs
    for modelName in modelNames:
        model_path = f'{results_path}/{modelName}'

        for abordagem in abordagens:
            print(modelName, " - ", abordagem, file=output_file)
            mean_runs = []
            for run in range(1, 1 + nruns):    
                nevals_for_success = []

                for img in range(n_samples):
                    success_data = pd.read_csv(f'{model_path}/{abordagem}/run_{run}/img_{img}/success_file.csv')
                    has_rows = len(success_data) > 0
                    if not has_rows:
                        continue
                    gen_first_success = success_data['gen'][0]
                    if abordagem == 'ra':
                        nevals_for_success.append(gen_first_success)
                    else:
                        nevals_for_success.append(gen_first_success*pop_size)

                mean = np.mean(nevals_for_success)
                mean_runs.append(mean)
                print(mean, file=output_file)
            print("mean runs: ", np.mean(mean_runs), np.std(mean_runs), file=output_file)

    print("counting as 40 000 evals when no success", file=output_file)
    for modelName in modelNames:
        model_path = f'{results_path}/{modelName}'

        for abv in abordagens:
            print(modelName, " - ", abv, file=output_file)
            mean_runs = []
            for run in range(1, 1 + nruns):    
                nevals_for_success = []

                for img in range(n_samples):
                    success_data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/success_file.csv')
                    has_rows = len(success_data) > 0
                    if not has_rows:
                        nevals_for_success.append(40000)
                    else:
                        gen_first_success = success_data['gen'][0]
                        if abordagem == 'ra':
                            nevals_for_success.append(gen_first_success)
                        else:
                            nevals_for_success.append(gen_first_success*pop_size)

                mean = np.mean(nevals_for_success)
                mean_runs.append(mean)
                print(mean, file=output_file)
            print("mean runs: ", np.mean(mean_runs), np.std(mean_runs), file=output_file)

    # Success rate per dataset (das n_samples, quantas conseguiu achar advs?)
    print("success rate", file=output_file)
    for modelName in modelNames:
        model_path = f'{results_path}/{modelName}'
        print(modelName, file=output_file)

        for abv in abordagens:
            print(abv, file=output_file)
            data = pd.read_csv(f'{model_path}/{abv}/metrics.csv')
            suc_rate = data.iloc[:, 0]
            print(suc_rate, file=output_file)
            print(np.mean(suc_rate), np.std(suc_rate), file=output_file)
