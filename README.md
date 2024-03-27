# Evolutionary Adversarial One-Pixel Attacks

This repository contains the implementation of the Evolutionary Adversarial One-Pixel Attacks discussed in the paper "A Comparative Analysis of Evolutionary
Adversarial One-Pixel Attacks".

The file setup_cifar is taken from https://github.com/carlini/nn_robust_attacks/blob/master/setup_cifar.py . It is used to retrieve the target models architecture and the CIFAR-10 images.

The file preparation_file.py tests the models and select the samples to attack.

To carry out the attacks, the user only needs to run the files: run_cycle_de.py, run_cycle_cmaes.py, run_cycle_ga.py, run_cycle_random.py.

The files differential_evolution.py, genetic_algorithm.py, cmaes.py contain the implementation of the evolutionary algorithms. The file random_.py contains the implementation of the random attack.

The file statistical_tests.ipnyb contains the statistical tests conducted.

The file make_images.py is used to make graphics.

The file create_img_grid.py is to visualize attacks.

The files results_analysis.py, CountDif_orig_img.py, medias_dif_per_model.py, StatsDif.py are for analysis. The three latter are for distortion analysis, and results_analysis.py for the other metrics.

## Cite this project

If you use this project in your research work or publication, please cite it using the following BibTeX entry:

```bibtex
@InProceedings{10.1007/978-3-031-56855-8_9,
  author="Clare, Luana
  and Marques, Alexandra
  and Correia, Jo{\~a}o",
  editor="Smith, Stephen
  and Correia, Jo{\~a}o
  and Cintrano, Christian",
  title="A Comparative Analysis of Evolutionary Adversarial One-Pixel Attacks",
  booktitle="Applications of Evolutionary Computation",
  year="2024",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="147--162",
  abstract="Adversarial attacks pose significant challenges to the robustness of machine learning models. This paper explores the one-pixel attacks in image classification, a black-box adversarial attack that introduces changes to the pixels of the input images to make the classifier predict erroneously. We use a pragmatic approach by employing different evolutionary algorithms - Differential Evolution, Genetic Algorithms, and Covariance Matrix Adaptation Evolution Strategy - to find and optimise these one-pixel attacks. We focus on understanding how these algorithms generate effective one-pixel attacks. The experimentation was carried out on the CIFAR-10 dataset, a widespread benchmark in image classification. The experimental results cover an analysis of the following aspects: fitness optimisation, number of evaluations to generate an adversarial attack, success rate, number of adversarial attacks found per image, solution space coverage and level of distortion done to the original image to generate the attack. Overall, the experimentation provided insights into the nuances of the one-pixel attack and compared three standard evolutionary algorithms, showcasing each algorithm's potential and evolutionary computation's ability to find solutions in this strict case of the adversarial attack.",
  isbn="978-3-031-56855-8"
}
