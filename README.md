# Evolutionary Adversarial One-Pixel Attacks

This repository contains the implementation of the Evolutionary Adversarial One-Pixel Attacks discussed in the paper "A Comparative Analysis of Evolutionary
Adversarial One-Pixel Attacks".

The file setup_cifar is taken from https://github.com/carlini/nn_robust_attacks/blob/master/setup_cifar.py . It is used to retrieve the target models architecture and the CIFAR-10 images.

The file preparation_file.py tests the models and select the samples to attack.

To carry out the attacks, the user only needs to run the files: run_cycle_de.py, run_cycle_cmaes.py, run_cycle_ga.py, run_cycle_random.py.

The files de.py, ga.py, cmaes.py contain the implementation of the evolutionary algorithms. The file random_.py contains the implementation of the random attack.

The file statistical_tests.ipnyb contains the statistical tests conducted.

The file make_images.py is used to make graphics.

The file create_img_grid.py is to visualize attacks.

The files results_analysis.py, CountDif_orig_img.py, medias_dif_per_model.py, StatsDif.py are for analysis. The three latter are for distortion analysis, and results_analysis.py for the other metrics.

