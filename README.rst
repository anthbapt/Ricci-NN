=============================================================================================
Ricci-NN : Ricci flow-like process for Neural-Network
=============================================================================================

This repository contains both the R and Python codes to run the analysis decribed in the following article:

# Add link to arXiv link

The R codes have been created by Christopher Banerji and the Python codes by Anthony Baptista.

---------------------------------------------------
R version
---------------------------------------------------

* ``fMNIST_DNN_training.r``: Code to train the different Neural Network architectures
* ``fMNIST_kNN_RCoef_eval.r``: Exploration of the k value for the k-nearest-neighbours (knn) graph construction. The graphs construct based on the knn are used to computed the Ricci flow-low like process

---------------------------------------------------
Python version
---------------------------------------------------

* ``fmnist_extraction.py``: Code to extract from the FMNIST dataset the test and train sub-dataset for the cloths labelled 5 (Sandal) and 9 (Ankle Boot). The raw data can be found at the following link: https://www.kaggle.com/datasets/zalando-research/fashionmnist
* ``training.py``: Code to train the different Neural Network architectures
* ``knn.py``: Exploration of the k value for the k-nearest-neighbours (knn) graph construction. The graphs construct based on the knn are used to computed the Ricci flow-low like process

---------------------------------------------------
Applications on FMINST
---------------------------------------------------
