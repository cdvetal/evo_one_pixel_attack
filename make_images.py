import csv
from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from collections import Counter
import seaborn as sns

modelNames = ['regular', 'distilled']
abordagens = ['differential evolution', 'genetic algorithm', 'cmaes']
abordagens_abv = ['de', 'ga', 'cmaes']
results_path = './results'
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
nruns = 10
n_samples = 500
n_gen = 100

graphics_folder = f"{results_path}/graphics"
if not os.path.exists(graphics_folder):
    os.mkdir(graphics_folder)

prunpimg = False
prunmimg = False
mrunpimg = False
mrunmimg = True


sns.set(style="white", rc={'figure.figsize':(8, 6), 'pdf.fonttype': 42, 'ps.fonttype': 42})

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Per run, per image
if prunpimg:
    for modelName in modelNames:
        model_folder = f"{graphics_folder}/{modelName}"
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        per_run_per_img = f"{model_folder}/per_run_per_img"
        if not os.path.exists(per_run_per_img):
            os.mkdir(per_run_per_img)

        model_path = f'{results_path}/{modelName}'
        
        for run in range(1, nruns+1):
            for img in range(n_samples):
                # Create a figure and axes for each plot
                fig_avg, ax_avg = plt.subplots()
                fig_adv, ax_adv = plt.subplots()
                fig_best, ax_best = plt.subplots()

                for abord in range(len(abordagens_abv)):
                    abv = abordagens_abv[abord]
                    abord_name = abordagens[abord]
                    data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/evolution_overview.csv')
                    fit_avg = data['average fitness']
                    fit_best = data['best fitness']
                    
                    success_data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/success_file.csv')
                    success_gen = success_data['gen']
                    generations = np.arange(len(fit_avg))
                    counts = {value: 0 for value in generations}
                    count_dict = dict(Counter(success_gen))
                    counts.update(count_dict)

                    adv_values = list(counts.values())
                    adv_values_accumulated = [adv_values[0]]  # Initialize with the first value from adv_values
                    for value in adv_values[1:]:
                        adv_values_accumulated.append(adv_values_accumulated[-1] + value)

                    # Plot fit_avg and fit_best for each approach in separate figures
                    ax_avg.plot(fit_avg, label=f'{abord_name}')
                    ax_best.plot(fit_best, label=f'{abord_name}')
                    ax_adv.plot(generations, adv_values, label=f'{abord_name} (new)')
                    ax_adv.plot(generations, adv_values_accumulated, linestyle='--', label=f'{abord_name} (accumulated)')

                # Set axis labels and legend
                ax_avg.set_xlabel('Generation')
                ax_avg.set_ylabel('Fitness')
                ax_avg.set_xticks(np.arange(0, len(fit_avg), 11))
                ax_avg.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
                ax_avg.set_title(f'Fitness Evolution - {modelName} - Image {img} - run {run} - Fit Avg')
                ax_avg.legend(loc='upper left', bbox_to_anchor=(1, 1))


                ax_best.set_xlabel('Generation')
                ax_best.set_ylabel('Fitness')
                ax_best.set_xticks(np.arange(0, len(fit_avg), 11))
                ax_best.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
                ax_best.set_title(f'Fitness Evolution - {modelName} - Image {img} - run {run} - Fit Best')
                ax_best.legend(loc='upper left', bbox_to_anchor=(1, 1))

                ax_adv.set_xlabel('Generation')
                ax_adv.set_ylabel('Number of adversarials')
                ax_adv.set_xticks(np.arange(0, len(fit_avg), 11))
                ax_adv.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
                ax_adv.set_title(f'Evolution of Adversarials - {modelName} - Image {img} - run {run}')
                ax_adv.legend(loc='upper left', bbox_to_anchor=(1, 1))

                # Save the plots to files or display them
                fig_avg.savefig(f'{per_run_per_img}/{modelName}_run_{run}_img_{img}_fit_avg.png',  bbox_inches='tight')
                fig_best.savefig(f'{per_run_per_img}/{modelName}_run_{run}_img_{img}_fit_best.png',  bbox_inches='tight')
                fig_adv.savefig(f'{per_run_per_img}/{modelName}_run_{run}_img_{img}_adv.png',  bbox_inches='tight')

                # Close the figures to avoid overlapping plots
                plt.close(fig_avg)
                plt.close(fig_best)
                plt.close(fig_adv)

# Per run, mean across images
if prunmimg:
    for modelName in modelNames:
        model_path = f'{results_path}/{modelName}'

        model_folder = f"{graphics_folder}/{modelName}"
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        per_run_mean_img = f"{model_folder}/per_run_mean_img"
        if not os.path.exists(per_run_mean_img):
            os.mkdir(per_run_mean_img)
        
        for run in range(1, 1 + nruns):
            fig_avg, ax_avg = plt.subplots()
            fig_best, ax_best = plt.subplots()
            fig_adv, ax_adv = plt.subplots()

            for abord in range(len(abordagens_abv)):
                abv = abordagens_abv[abord]
                abord_name = abordagens[abord]
                fit_avgs = []
                fit_bests = []
                adv_values_s = []
                adv_values_accumulated_s = []

                for img in range(n_samples):
                    data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/evolution_overview.csv')
                    fit_avg = data['average fitness']
                    fit_best = data['best fitness']
                    success_data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/success_file.csv')
                    fit_avgs.append(fit_avg)
                    fit_bests.append(fit_best)

                    success_data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/success_file.csv')
                    success_gen = success_data['gen']
                    generations = np.arange(len(fit_avg))
                    counts = {value: 0 for value in generations}
                    count_dict = dict(Counter(success_gen))
                    counts.update(count_dict)

                    adv_values = list(counts.values())
                    adv_values_s.append(adv_values)

                    adv_values_accumulated = [adv_values[0]]  # Initialize with the first value from adv_values
                    for value in adv_values[1:]:
                        adv_values_accumulated.append(adv_values_accumulated[-1] + value)
                    adv_values_accumulated_s.append(adv_values_accumulated)

                # Transpose fit_avgs and fit_bests for calculating the mean across generations
                fit_avgs = np.array(fit_avgs).T
                fit_bests = np.array(fit_bests).T
                adv_values_s = np.array(adv_values_s).T
                adv_values_accumulated_s = np.array(adv_values_accumulated_s).T
                
                # Calculate the mean across generations
                mean_fit_avgs = np.mean(fit_avgs, axis=1)
                mean_fit_bests = np.mean(fit_bests, axis=1)
                mean_adv_values = np.mean(adv_values_s, axis=1)
                mean_adv_values_accumulated = np.mean(adv_values_accumulated_s, axis=1)

                # Plot mean fit_avg and mean fit_best for each approach
                ax_avg.plot(mean_fit_avgs, label=f'{abord_name}')
                ax_best.plot(mean_fit_bests, label=f'{abord_name}')
                ax_adv.plot(generations, mean_adv_values, label=f'{abord_name} (new)')
                ax_adv.plot(generations, mean_adv_values_accumulated, linestyle='--', label=f'{abord_name} (accumulated)')

            # Set axis labels and legend
            ax_avg.set_xlabel('Generation')
            ax_avg.set_ylabel('Fitness')
            ax_avg.set_title(f'Fitness Evolution (avg between images) - {modelName } - run {run}')
            ax_avg.set_xticks(np.arange(0, len(fit_avg), 11))
            ax_avg.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
            ax_avg.legend(loc='upper left', bbox_to_anchor=(1, 1))

            ax_best.set_xlabel('Generation')
            ax_best.set_ylabel('Fitness')
            ax_best.set_title(f'Fitness Evolution (avg between images) - {modelName} - run {run}')
            ax_best.set_xticks(np.arange(0, len(fit_avg), 11))
            ax_best.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
            ax_best.legend(loc='upper left', bbox_to_anchor=(1, 1))

            ax_adv.set_xlabel('Generation')
            ax_adv.set_ylabel('Number of adversarials')
            ax_adv.set_xticks(np.arange(0, len(fit_avg), 11))
            ax_adv.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
            ax_adv.set_title(f'Evolution of Adversarials (avg between images) - {modelName} - run {run}')
            ax_adv.legend(loc='upper left', bbox_to_anchor=(1, 1))

            # Save the plots to files or display them
            fig_avg.savefig(f'{per_run_mean_img}/{modelName}_run_{run}_fit_avg.png', bbox_inches='tight')
            fig_best.savefig(f'{per_run_mean_img}/{modelName}_run_{run}_fit_best.png',  bbox_inches='tight')
            fig_adv.savefig(f'{per_run_mean_img}/{modelName}_run_{run}_adv.png',  bbox_inches='tight')

            # Close the figures to avoid overlapping plots
            plt.close(fig_avg)
            plt.close(fig_best)
            plt.close(fig_adv)

# Mean across runs, per image
if mrunpimg:
    for modelName in modelNames:
        model_path = f'{results_path}/{modelName}'

        model_folder = f"{graphics_folder}/{modelName}"
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        
        mean_run_per_img = f"{model_folder}/mean_run_per_img"
        if not os.path.exists(mean_run_per_img):
            os.mkdir(mean_run_per_img)

        for img in range(n_samples):
            fig_avg, ax_avg = plt.subplots()
            fig_best, ax_best = plt.subplots()
            fig_adv, ax_adv = plt.subplots()

            for abord in range(len(abordagens_abv)):
                abv = abordagens_abv[abord]
                abord_name = abordagens[abord]
                fit_avgs = []
                fit_bests = []
                adv_values_s = []
                adv_values_accumulated_s = []
                
                for run in range(1, 1 + nruns):
                    data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/evolution_overview.csv')
                    fit_avg = data['average fitness']
                    fit_best = data['best fitness']
                    success_data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/success_file.csv')
                    fit_avgs.append(fit_avg)
                    fit_bests.append(fit_best)


                    if len(fit_avg) < n_gen:
                        fit_avg.append(fit_avg[-1]*(n_gen - len(fit_avg)))
                    if len(fit_best) < n_gen:
                        fit_best.append(fit_best[-1]*(n_gen - len(fit_best)))


                    success_data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/success_file.csv')
                    success_gen = success_data['gen']
                    generations = np.arange(len(fit_avg))
                    counts = {value: 0 for value in generations}
                    count_dict = dict(Counter(success_gen))
                    counts.update(count_dict)

                    adv_values = list(counts.values())
                    adv_values_s.append(adv_values)

                    adv_values_accumulated = [adv_values[0]]  # Initialize with the first value from adv_values
                    for value in adv_values[1:]:
                        adv_values_accumulated.append(adv_values_accumulated[-1] + value)
                    adv_values_accumulated_s.append(adv_values_accumulated)

                # Transpose fit_avgs and fit_bests for calculating the mean across generations
                fit_avgs = np.array(fit_avgs).T
                fit_bests = np.array(fit_bests).T
                adv_values_s = np.array(adv_values_s).T
                adv_values_accumulated_s = np.array(adv_values_accumulated_s).T
                    
                # Calculate the mean across generations
                mean_fit_avgs = np.mean(fit_avgs, axis=1)
                mean_fit_bests = np.mean(fit_bests, axis=1)
                mean_adv_values = np.mean(adv_values_s, axis=1)
                mean_adv_values_accumulated = np.mean(adv_values_accumulated_s, axis=1)

                # Plot mean fit_avg and mean fit_best for each approach
                ax_avg.plot(mean_fit_avgs, label=f'{abord_name}')
                ax_best.plot(mean_fit_bests, label=f'{abord_name}')
                ax_adv.plot(generations, mean_adv_values, label=f'{abord_name} (new)')
                ax_adv.plot(generations, mean_adv_values_accumulated, linestyle='--', label=f'{abord_name} (accumulated)')

            # Set axis labels and legend
            
            ax_avg.set_xlabel('Generation')
            ax_avg.set_ylabel('Fitness')
            ax_avg.set_title(f'Average Fitness Evolution (avg between runs) - {modelName} - Image {img}')
            ax_avg.set_xticks(np.arange(0, len(fit_avg), 11))
            ax_avg.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
            ax_avg.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax_avg.set_ylim(bottom=0)
            ax_best.set_xlabel('Generation')
            ax_best.set_ylabel('Fitness')
            ax_best.set_title(f'Best Fitness Evolution (avg between runs) - {modelName} - Image {img}')
            ax_best.set_xticks(np.arange(0, len(fit_avg), 11))
            ax_best.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
            ax_best.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax_best.set_ylim(bottom=0)

            ax_adv.set_xlabel('Generation')
            ax_adv.set_ylabel('Number of adversarials')
            ax_adv.set_xticks(np.arange(0, len(fit_avg), 11))
            ax_adv.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45)
            ax_adv.set_title(f'Evolution of Adversarials (avg between runs) - {modelName} - Image {img}')
            ax_adv.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax_adv.set_ylim(bottom=0)

            # Save the plots to files or display them
            fig_avg.savefig(f'{mean_run_per_img}/{modelName}_img_{img}_fit_avg.png', bbox_inches='tight')
            fig_best.savefig(f'{mean_run_per_img}/{modelName}_img_{img}_fit_best.png', bbox_inches='tight')
            fig_adv.savefig(f'{mean_run_per_img}/{modelName}_img_{img}_adv.png',  bbox_inches='tight')

            # Close the figures to avoid overlapping plots
            plt.close(fig_avg)
            plt.close(fig_best)
            plt.close(fig_adv)
font_size = 25
# Mean across runs, mean across imgs
if mrunmimg:
    for modelName in modelNames:
        model_path = f'{results_path}/{modelName}'

        model_folder = f"{graphics_folder}/{modelName}"
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        
        mean_run_mean_img = f"{model_folder}/mean_run_mean_img"
        if not os.path.exists(mean_run_mean_img):
            os.mkdir(mean_run_mean_img)
        
         # Create figures and axes with specified font size
        fig_avg, ax_avg = plt.subplots()
        ax_avg.tick_params(axis='both', labelsize=font_size)
        
        fig_best, ax_best = plt.subplots()
        ax_best.tick_params(axis='both', labelsize=font_size)
        
        fig_adv, ax_adv = plt.subplots()
        ax_adv.tick_params(axis='both', labelsize=font_size)
        
        fig_acc, ax_acc = plt.subplots()
        ax_acc.tick_params(axis='both', labelsize=font_size)
        
        fig_fit, ax_fit = plt.subplots()
        ax_fit.tick_params(axis='both', labelsize=font_size)

        for abord in range(len(abordagens_abv)):
            abv = abordagens_abv[abord]
            abord_name = abordagens[abord]
            fit_avgs = []
            fit_bests = []
            adv_values_s = []
            adv_values_accumulated_s = []

            for img in range(n_samples):
                img_fit_avgs = []
                img_fit_bests = []
                img_adv_values_s = []
                img_adv_values_accumulated_s = []
            
                for run in range(1, 1 + nruns):
                    success_data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/success_file.csv')
                    success_gen = success_data['gen']
                    generations = np.arange(n_gen)
                    counts = {value: 0 for value in generations}
                    count_dict = dict(Counter(success_gen))
                    counts.update(count_dict)

                    adv_values = list(counts.values())
                    img_adv_values_s.append(adv_values)

                    adv_values_accumulated = [adv_values[0]]  # Initialize with the first value from adv_values
                    for value in adv_values[1:]:
                        adv_values_accumulated.append(adv_values_accumulated[-1] + value)
                    img_adv_values_accumulated_s.append(adv_values_accumulated)
                    print(adv_values)
                    if adv_values == 0:
                        print('zero')
                    data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/evolution_overview.csv')
                    fit_avg = data['average fitness']
                    fit_best = data['best fitness']
                    if len(fit_avg) < n_gen:
                        last_value = fit_avg.iloc[-1]
                        fit_avg = fit_avg.append(pd.Series([last_value] * (n_gen - len(fit_avg))), ignore_index=True)
                        # fit_avg = np.append(fit_avg, [fit_avg.values[-1]*(n_gen - len(fit_avg))])
                    if len(fit_best) < n_gen:
                        last_value = fit_avg.iloc[-1]
                        fit_best = fit_best.append(pd.Series([last_value] * (n_gen - len(fit_best))), ignore_index=True)
                        # fit_best = np.append(fit_best, [fit_best.values[-1]*(n_gen - len(fit_best))])

                    img_fit_avgs.append(fit_avg)
                    img_fit_bests.append(fit_best)

                    
                # Transpose fit_avgs and fit_bests for calculating the mean across generations
                img_fit_avgs = np.array(img_fit_avgs).T
                img_fit_bests = np.array(img_fit_bests).T
                img_adv_values_s = np.array(img_adv_values_s).T
                img_adv_values_accumulated_s = np.array(img_adv_values_accumulated_s).T
                
                # Calculate the mean across generations
                mean_fit_avgs = np.mean(img_fit_avgs, axis=1)
                mean_fit_bests = np.mean(img_fit_bests, axis=1)
                mean_adv_values = np.mean(img_adv_values_s, axis=1)
                mean_adv_values_accumulated = np.mean(img_adv_values_accumulated_s, axis=1)

                fit_avgs.append(list(mean_fit_avgs))
                fit_bests.append(list(mean_fit_bests))
                adv_values_s.append(list(mean_adv_values))
                adv_values_accumulated_s.append(list(mean_adv_values_accumulated))
                

            # Transpose fit_avgs and fit_bests for calculating the mean across generations
            fit_avgs = np.array(fit_avgs).T
            fit_bests = np.array(fit_bests).T
            adv_values_s = np.array(adv_values_s).T
            adv_values_accumulated_s = np.array(adv_values_accumulated_s).T

            # Calculate the mean across generations
            mean_mean_fit_avgs = np.mean(fit_avgs, axis=1)
            mean_mean_fit_bests = np.mean(fit_bests, axis=1)
            mean_mean_adv_values = np.mean(adv_values_s, axis=1)
            mean_mean_adv_values_accumulated = np.mean(adv_values_accumulated_s, axis=1)

            # Plot mean fit_avg and mean fit_best for each approach
            ax_avg.plot(mean_mean_fit_avgs, label=f'{(abordagens_abv[abord]).upper()}', color=colors[abord])
            ax_best.plot(mean_mean_fit_bests, label=f'{(abordagens_abv[abord]).upper()}', color=colors[abord])
            ax_adv.plot(generations, mean_mean_adv_values, label=f'{(abordagens_abv[abord]).upper()}', color=colors[abord])
            ax_acc.plot(generations, mean_mean_adv_values_accumulated, label=f'{(abordagens_abv[abord]).upper()}', color=colors[abord])
            ax_fit.plot(mean_mean_fit_bests, label=f'{(abordagens_abv[abord]).upper()} best', color=colors[abord])
            ax_fit.plot(mean_mean_fit_avgs, label=f'{(abordagens_abv[abord]).upper()} average', linestyle='--', color=colors[abord])

        # Set axis labels and legend
        ax_avg.set_xlabel('Generation',  fontsize=font_size)
        ax_avg.set_ylabel('Fitness',  fontsize=font_size)
        # ax_avg.set_title(f'Average Fitness Evolution (avg between runs, avg between img) - {modelName}')
        ax_avg.set_xticks(np.arange(0, len(fit_avg), 11))
        ax_avg.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45,  fontsize=font_size)
        ax_avg.legend(loc='upper left', bbox_to_anchor=(1, 1),  fontsize=font_size)
        ax_avg.set_ylim(bottom=0)

        ax_best.set_xlabel('Generation',  fontsize=font_size)
        ax_best.set_ylabel('Fitness',  fontsize=font_size)
        # ax_best.set_title(f'Best Fitness Evolution (avg between runs, avg between img) - {modelName}')
        ax_best.set_xticks(np.arange(0, len(fit_avg), 11))
        ax_best.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45,  fontsize=font_size)
        ax_best.legend(loc='upper left', bbox_to_anchor=(1, 1),  fontsize=font_size)
        ax_best.set_ylim(bottom=0)

        ax_fit.set_xlabel('Generation',  fontsize=font_size)
        ax_fit.set_ylabel('Fitness',  fontsize=font_size)
        # ax_fit.set_title(f'Fitness Evolution')
        ax_fit.set_xticks(np.arange(0, len(fit_avg) + 1, 10))
        ax_fit.set_xticklabels(np.arange(0, len(fit_avg) + 1, 10), rotation=45,  fontsize=font_size)
        ax_fit.legend(loc='upper left',  bbox_to_anchor=(1, 1),  fontsize=font_size)
        ax_fit.set_ylim(bottom=0.4, top=1.75)
        ax_fit.grid(True)


        ax_adv.set_xlabel('Generation',  fontsize=font_size)
        ax_adv.set_ylabel('Number of new adversarials',  fontsize=font_size-3)
        ax_adv.set_xticks(np.arange(0, len(fit_avg), 11))
        ax_adv.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45,  fontsize=font_size)
        # ax_adv.set_title(f'Evolution of New Adversarials (avg between runs, avg between img) - {modelName}')
        ax_adv.legend(loc='upper left', bbox_to_anchor=(1, 1),  fontsize=font_size)
        ax_adv.set_ylim(bottom=0)

        ax_acc.set_xlabel('Generation',  fontsize=font_size)
        ax_acc.set_ylabel('Number of accumulated adversarials',  fontsize=font_size-7)
        ax_acc.set_xticks(np.arange(0, len(fit_avg), 11))
        ax_acc.set_xticklabels(np.arange(0, len(fit_avg), 11), rotation=45,  fontsize=font_size)
        # ax_acc.set_title(f'Evolution of Accumulated Adversarials (avg between runs, avg between img) - {modelName}')
        ax_acc.legend(loc='upper left', bbox_to_anchor=(1, 1),  fontsize=font_size)
        ax_acc.set_ylim(bottom=0)

        # Save the plots to files or display them
        fig_avg.savefig(f'{mean_run_mean_img}/{modelName}_fit_avg.png', bbox_inches='tight')
        fig_best.savefig(f'{mean_run_mean_img}/{modelName}_fit_best.png', bbox_inches='tight')
        fig_adv.savefig(f'{mean_run_mean_img}/{modelName}_adv.png', bbox_inches='tight')
        fig_acc.savefig(f'{mean_run_mean_img}/{modelName}_acc.png', bbox_inches='tight')
        fig_fit.savefig(f'{mean_run_mean_img}/{modelName}_fit.png', bbox_inches='tight')
        # Close the figures to avoid overlapping plots
        plt.close(fig_avg)
        plt.close(fig_best)
        plt.close(fig_adv)
        plt.close(fig_acc)
        plt.close(fig_fit)