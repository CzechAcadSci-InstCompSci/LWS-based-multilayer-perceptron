# LWS-based robust multilayer perceptron (LWS-MLP), version 1.0

The software implemented in Keras allows to perform a robust training of multilayer 
perceptrons (MLPs) in a very elegant way. It is able to modify standard steps of 
training of MLPs by a mere replacement of the standard loss function by a 
non-standard version. This has the form of the sum of implicitly weighted squared 
residuals, as inspired by the least weighted squares estimator from linear regression. 
In fact, the standard tools for training MLPs are so complex, that the main contribution 
of the software is the idea which part of the codes should be modified (robustified).

Feel free to use or modify the code.

## Requirements

You need to install Python, its library NumPy, its math module, TensorFlow, and Keras  (which itself is an open-source library written in Python).

## Usage

* The usage of the code is straightforward. The training of the robust MLP is called in the same way as habitually
used calling of a standard (non-robust) MLP.

## Authors
  * Jan Tichavský, The Czech Academy of Sciences, Institute of Computer Science
  * Jan Kalina, The Czech Academy of Sciences, Institute of Computer Science

## Contact

Do not hesitate to contact us (tichavsk@seznam.cz) or write an Issue.

## How to cite

When refering to the LWS-MLP method, please consider citing the following:

Kalina J, Vidnerová P (2020): Robust multilayer perceptrons: Robust loss functions and their derivatives. Proceedings of the 21st
EANN (Engineering Applications of Neural Networks) 2020 Conference. Proceedings of the International Neural Networks Society, 
vol. 2, Springer, Cham, pp. 546-557.

## Acknowledgement

This work was supported by the Czech Science Foundation grant GA19-05704S.