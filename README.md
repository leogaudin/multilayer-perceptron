<h1 align='center'> 🧠 multilayer-perceptron</h1>

> ⚠️ This tutorial assumes you have done [dslr](https://github.com/leogaudin/dslr) and that you have a good understanding of the basics of machine learning and linear algebra.

## Table of Contents

- [Introduction](#introduction) 👋
- [Layers](#layers) 📚
- [Forward Propagation](#forward-propagation) ➡️
- [Back Propagation](#back-propagation) 🔙
- [Resources](#resources) 📖
<!-- - [Losses](#losses) 📉
- [Optimizers](#optimizers) 🚀 -->

## Introduction

`multilayer-perceptron` gives us a problem to solve that is fairly similar to the one we had in `dslr`, if not simpler: we are given a dataset of fine-needle aspirates of breast mass, and we have to predict whether the mass is benign or malignant.

The project is not really about the dataset, but rather about the implementation of a neural network, and how modular it is. Here, we will cover the various "modules" of the neural network, and how they interact with each other.

> 💡 I recommend you watch *sentdex*'s [Neural Networks from Scratch](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3) first, as it provides great explanations on each component of a NN and their respective roles.

## Layers

A neural network is made of layers, and layers are made of neurons, which are the basic unit of computation in a neural network.

A layer is characterized by its input and output sizes (e.g. 4 neurons in, 3 neurons out), along with its weights and biases.

The weights and biases are the parameters of the layer, and they are what the neural network will tweak during its training.

There are several types of layers, but we will explain only two of them here.

### Dense

A **dense** layer, or **fully connected** layer, is a type of layer where each neuron is connected to each neuron in the previous layer.

Each weight in the layer is associated with a connection between two neurons (i.e. what impact will the output of neuron $i$ have on neuron $j$).

### Activation

Activations are actually functions, as they do not depend on any external weight but only on the output of the neurons. However, we will treat them as some kind of intermiediate layer here, as it will be more relevant when we get to forward and back propagation.

In `dslr`, our activation function was the **sigmoid** function, which squashes the output of the neuron between 0 and 1.

It is now considered a bit old-fashioned, so you would rather use the **ReLU** function, which is much simpler and faster to compute.

> 💡 The ReLU function is defined as $f(x) = \max(0, x)$. Simpler than $\frac{1}{1 + e^{-x}}$ to be honest.

## Forward Propagation

Forward propagation is the simple process of passing an input through the layers of the neural network.

You pass an input to the input layer, that will pass it to a hidden layer that will compute an output, passed to an activation layer, passed to another hidden layer, and so on, until you reach the output layer.

The output layer will then give the prediction of the neural network.

Simple.

## Back Propagation

With forward propagation, we now have a prediction. However, we need to know how good this prediction is, and how we can improve it.

In `dslr`, we directly calculated the derivative of the loss with respect to the weights and biases, and updated them accordingly, because each weight had a direct impact on the output.

However, in a neural network with multiple layers, the weights of the first layer have an indirect impact on the output, and it gets trickier to derive.

That is where **backpropagation** and the **chain rule** come in.

Suppose we want to calculate the impact of $x$ on $y$. However, $x$ only has an impact on $u$, that has an impact on $v$, that finally has an impact on $y$.

In this case, the chain rule states that the impact of $x$ on $y$ is the product of the impacts of $x$ on $u$, $u$ on $v$, and $v$ on $y$:

$$
\frac{dy}{dx} = \frac{dy}{dv} \cdot \frac{dv}{du} \cdot \frac{du}{dx}
$$

> 💡 Back propagation is one of the hardest concepts of this project, so be sure to check out the resources, particularly [this video](https://www.youtube.com/watch?v=pauPCy_s0Ok) that decomposes the problem amazingly.

Just as forward propagation goes from start to finish, passing inputs sequentially, back propagation goes from finish to start, passing gradients sequentially.

We simply need to compute the gradient of an output with respect to an input, for each layer.

So, let's do so step by step. Let's say our neural network output a probability distribution $[0.7, 0.3]$ for a sample, and the true distribution is $[1, 0]$.

If we used the **mean squared error** loss, the loss would be:

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where $y_i$ is the true value and $\hat{y}_i$ is the predicted value.

The gradient of the loss with respect to the output would be:

$$
\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{n} \cdot (\hat{y}_i - y_i)
$$

Here:

$$
\frac{\partial L}{\partial Y} = \frac{2}{n} \odot \begin{bmatrix} 0.7 - 1 \\ 0.3 - 0 \end{bmatrix} = \begin{bmatrix} -0.6 \\ 0.3 \end{bmatrix}
$$

And that is the first gradient we will pass to the output layer.

## Resources

- [📺 YouTube − Neural Networks from Scratch](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3) : understand the various components of a neural network and how to create one in a modular way.
- [📺 YouTube − Backpropagation, step-by-step | DL3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) : understand how backpropagation works.
- [📺 YouTube − Backpropagation calculus | DL4](https://www.youtube.com/watch?v=tIeHLnjs5U8) : understand the math behind backpropagation.
- [📺 YouTube − Backpropagation Algorithm | Neural Networks](https://www.youtube.com/watch?v=sIX_9n-1UbM) : have a good insight of the chain rule's logic in backpropagation.
- [📺 YouTube − Neural Network from Scratch | Mathematics & Python Code](https://www.youtube.com/watch?v=pauPCy_s0Ok) : understand the actual implementation of backpropagation in code.

- [📖 ]()
- [💬 ]()
