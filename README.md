<h1 align='center'> 🧠 multilayer-perceptron</h1>

> ⚠️ This tutorial assumes you have done [dslr](https://github.com/leogaudin/dslr) and that you have a good understanding of the basics of machine learning and linear algebra.

## Table of Contents

- [Introduction](#introduction) 👋
- [Layers](#layers) 📚
- [Activations](#activations) 🧠
- [Backpropagation](#backpropagation) 🔙
- [Losses](#losses) 📉
- [Optimizers](#optimizers) 🚀
- [Resources](#resources) 📖

## Introduction

`multilayer-perceptron` gives us a problem to solve that is fairly similar to the one we had in `dslr`, if not simpler: we are given a dataset of fine-needle aspirates of breast mass, and we have to predict whether the mass is benign or malignant.

The project is not really about the dataset, but rather about the implementation of a neural network, and how modular it is. Here, we will cover the various "modules" of the neural network, and how they interact with each other.

> 💡 I recommend you watch *sentdex*'s [Neural Networks from Scratch](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3) first, as it provides great explanations on each component of a NN and their respective roles.

## Layers

A neural network is made of layers, and layers are made of neurons, which are the basic unit of computation in a neural network.

A layer is characterized by its input and output sizes (e.g. 4 neurons in, 3 neurons out), and its weights and biases. The weights and biases are the parameters of the layer, and they are what the neural network will learn.

The simplest type is the `Dense` layer, which is a fully connected layer, that is to say that every neuron in the layer is connected to every neuron in the previous one.

## Activations

The activation functions are what determines the output of a neuron.

In `dslr`, we used the **sigmoid** function, which squashes the output of the neuron between 0 and 1.

However, it is now considered a bit old-fashioned, and we will use the **ReLU** function instead, which is much simpler and faster to compute.

## Backpropagation

Backpropagation is the algorithm that allows us to train the neural network.

It is based on the chain rule of calculus, and it allows us to compute the gradient of the loss with respect to the parameters of the network.

## Resources

- [📺 YouTube − Neural Networks from Scratch](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
- [📺 YouTube − Backpropagation, step-by-step | DL3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- [📺 YouTube − Backpropagation calculus | DL4](https://www.youtube.com/watch?v=tIeHLnjs5U8)

- [📖 ]()
- [💬 ]()
