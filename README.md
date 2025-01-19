<h1 align='center'> 🧠 multilayer-perceptron</h1>

> ⚠️ This tutorial assumes you have done [dslr](https://github.com/leogaudin/dslr) and that you have a good understanding of the basics of machine learning and linear algebra.

## Table of Contents

- [Introduction](#introduction) 👋
- [Layers](#layers) 📚
    - [Dense](#dense)
    - [Activation](#activation)
- [Forward Propagation](#forward-propagation) ➡️
- [Backward Propagation](#backward-propagation) 🔙
    - [Softmax + cross-entropy loss](#softmax--cross-entropy-loss)
    - [Activation layers](#activation-layers)
        - [Sigmoid](#sigmoid)
        - [ReLU](#relu)
    - [Dense layers](#dense-layers)
    - [Putting it all together](#putting-it-all-together)
- [Early stopping](#early-stopping) 🛑
- [Complex optimizations](#complex-optimizations) 🚀
    - [Momentum](#momentum)
    - [RMSprop](#rmsprop)
    - [Adam](#adam)
- [Resources](#resources) 📖

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

## Backward Propagation

With forward propagation, we now have a prediction. However, we need to know how good this prediction is, and how we can improve it.

In `dslr`, we directly calculated the derivative of the loss w.r.t. the weights and biases, and updated them accordingly, because each weight had a direct impact on the output.

However, in a neural network with multiple layers, the weights of the first layer have an indirect impact on the output, and it gets trickier to derive.

That is where **backpropagation** and the **chain rule** come in.

We basically decompose the task of computing the derivatives per layer, and pass them sequentially from the output layer to the input layer.

That is to say, each element will need to implement the derivative of its output error w.r.t. its input, formally $\frac{\partial E}{\partial X}$, based on the derivative of the error w.r.t. its output, $\frac{\partial E}{\partial Y}$.

<p align='center'>
  <img src='./assets/propagation.webp' alt='Forward and backward propagation' />
</p>

> *Forward and backward propagation visualized for one layer.*
>
> 💡 Back propagation is one of the hardest concepts of this project, so be sure to check out the resources, particularly [this video](https://www.youtube.com/watch?v=pauPCy_s0Ok) that decomposes the problem amazingly, also available as  [an article](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65).

Naturally, the first derivative we will need to compute is the derivative of the loss w.r.t. the output of the neural network.

Here, the network's output passes through a **softmax** function, which turns a vector of raw scores into a vector of probabilities (e.g. $[0.1, 0.2, 0.7]$).

Then, it is passed to a **cross-entropy** loss function, which computes the difference between the predicted probabilities and the actual probabilities (e.g. the actual result was $[0, 0, 1]$).

The derivative of the cross-entropy loss w.r.t. its input is quite tricky, and the softmax function is even trickier.

However, the derivative of the **softmax function + cross-entropy loss** is quite simple, and is given by the formula $Y_{pred} - Y_{true}$. You can find the full derivation [here](https://www.pinecone.io/learn/cross-entropy-loss/#Derivative-of-the-Softmax-Cross-Entropy-Loss-Function).

> 💡 Capital letters denote vectors, and normal letters are scalars. $Y$ is a vector $[y_1, y_2, y_3]$ for example.

We will now see how to compute $\frac{\partial E}{\partial X}$ for each layer.

In the following, we will call $\frac{\partial E}{\partial Y}$ the **output gradient** of the layer.

> ⚠️ Here, all the derivatives are given already reduced, but you should watch the whole process of deriving them in [this video](https://www.youtube.com/watch?v=pauPCy_s0Ok) mentioned above.

### Softmax + cross-entropy loss

We just saw that the derivative of the softmax + cross-entropy loss is $Y_{pred} - Y_{true}$.

### Activation layers

The derivative of activation layers is given by its output gradient, multiplied element-wise by the derivative of the activation function used.

$$
\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} \odot f'(X)
$$

Let's see some derivatives $f'(x)$ for common activation functions.

#### Sigmoid

The derivative of the sigmoid function is given by:

$$
f'(x) = f(x) (1 - f(x))
$$

> 💡 Here for instance, $\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} \odot f(x) (1 - f(x))$.

#### ReLU

The derivative of the ReLU function is quite simple:

$$
f'(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{otherwise}
\end{cases}
$$

### Dense layers

Dense layers are a bit more complex, because aside from computing the input gradient based on the output gradient, **we also need to compute the weights and biases gradients in order to update them**.

Here, we assume the weights are a matrix of shape $(n, m)$, where $n$ is the number of neurons in the previous layer, and $m$ is the number of neurons in the current layer. That is to say, it takes an input of size $n$ and outputs a vector of size $m$.

The input gradient is given by:

$$
\frac{\partial E}{\partial X} = \frac{\partial E}{\partial Y} \cdot W^T
$$

The weights gradient is given by:

$$
\frac{\partial E}{\partial W} = X^T \cdot \frac{\partial E}{\partial Y}
$$

The biases gradient is simply the output gradient $\frac{\partial E}{\partial Y}$. Note that in the case of batch-GD, you will need to sum the gradients over the batch dimension.

### Putting it all together

Now, let's take the following neural network:

- Input layer of size 4
- ReLU activation layer
- Hidden layer of size 3
- ReLU activation layer
- Output layer of size 2
- Softmax activation layer

During a single epoch, backpropagation will look like this:

1. Compute the output of the neural network and pass it through softmax to get $Y_{pred}$.
2. Compute the loss gradient $Y_{pred} - Y_{true}$.
3. Pass it to the output layer to get the gradient of its loss w.r.t. its input.
4. Pass this gradient to the ReLU activation layer to get the gradient of its loss w.r.t. its input.
5. Pass this gradient to the hidden layer to get the gradient of its loss w.r.t. its input, and allow it to update its weights and biases.
6. And so on, until we reach the input layer.

> 💡 You might end up fighting with NumPy's broadcasting when working with batches. Do not sum or average the batches without reason. You should be able to implement this without doing any other sum than the biases update's.

## Early stopping

When training a neural network, it is good practice to keep track of the loss on the training data, obviously, but also on some **validation data**, that is data that the neural network has never seen before.

Typically, if we plot the training loss and the validation loss, we will see that the training loss will decrease over time, while the validation loss will decrease at first, but then increase again.

<p align='center'>
  <img src='./assets/overfitting.webp' alt='Overfitting' />
</p>

This is called **overfitting**, and it happens when the neural network learns the training data too well, and starts to "learn the noise" in the data.

Although the loss on the training data is still decreasing, it is basically useless to keep going, as it will never perform that well on new data.

That is why we use **early stopping**, which consists of stopping the training when the validation loss starts to increase again.

It is fairly simply to implement, you just need to add the following logic to your training loop:

1. Initialize a counter and a variable to keep track of the best loss.
```python
patience_counter = 0
best_loss = float("inf")
```
2. If the loss is lower than the previous best loss, save it.
```python
if val_loss < best_loss:
    best_loss = val_loss
```

3. If the loss is higher than the previous best loss, increment the counter.
```python
else:
    patience_counter += 1
```

4. If the counter reaches a certain threshold, stop the training.
```python
if patience_counter >= patience:
    break
```

> 💡 You can also save the model's weights when the validation loss is the lowest, and load them back when you stop the training.

That's it!

## Complex optimizations

### Momentum

Vanilla gradient descent consists of simply updating the weights and biases based on the gradients.

The concept of momentum introduces some inertia in the updates, by adding a fraction of the previous update to the current update.

To implement it, you can simply store the previous update in a variable, and update it as follows:

```python
if velocity is None:
    velocity = np.zeros_like(weights)

velocity = momentum * velocity + learning_rate * gradient
weights -= velocity
```

Note that the only difference with the vanilla gradient descent is the addition of the `momentum * velocity` term, as vanilla gradient descent would simply be:

```python
weights -= learning_rate * gradient
```

### RMSprop

RMSprop is a more complex optimization algorithm that adapts the learning rate based on the gradients.

It is based on AdaGrad, which adapts the learning rate based on the sum of the squared gradients. However, it is considered too aggressive, as the learning rate decreases too quickly.

RMSprop introduces a **decay factor** to the sum of the squared gradients, which allows the learning rate to decrease more slowly.

We can introduce some "velocity" term as we did before, that is defined as follows:

$$
v_t = \beta \cdot v_{t - 1} + (1 - \beta) \cdot {\nabla J(\theta)}^2
$$

Where:

- $v_t$ is the velocity at time $t$
- $\beta$ is the decay factor (around 0.9)
- $\nabla J(\theta)$ is the gradient of the loss w.r.t. the weights

> 💡 ${\nabla J(\theta)}^2$ is of course a Hadamard product, we could have noted it $\nabla J(\theta) \odot \nabla J(\theta)$.

Then, we can update the weights as follows:

$$
\theta = \theta - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot \nabla J(\theta)
$$

As you can see, the velocity is based on

Where:

- $\eta$ is the learning rate
- $\epsilon$ is a small number (e.g. $10^{-7}$), to avoid division by zero

The implementation logic is the same as before, only the update formula changes!

You can check out this [interesting article](https://medium.com/@piyushkashyap045/understanding-rmsprop-a-simple-guide-to-one-of-deep-learnings-powerful-optimizers-403baeed9922) for a more in-depth explanation.

### Adam

Adam is a combination of momentum and RMSprop, and is considered one of the best optimization algorithms for neural networks. It stands for **Ada**ptive **M**oment Estimation.

It introduces **two** momentum terms, that we will denote $m_t$ and $v_t$.

$m_t$ is the one that will **add some inertia to the updates**, and is defined as follows:

$$
m_t = \beta_1 \cdot m_{t - 1} + (1 - \beta_1) \cdot \nabla J(\theta)
$$

Where:

- $m_t$ is the momentum at time $t$
- $\beta_1$ is the decay factor (around 0.9)
- $\nabla J(\theta)$ is the gradient of the loss w.r.t. the weights

$v_t$, on its side, will **adapt the learning rate** based on the gradients, and is defined as follows:

$$
v_t = \beta_2 \cdot v_{t - 1} + (1 - \beta_2) \cdot {\nabla J(\theta)}^2
$$

Where:

- $v_t$ is the momentum at time $t$
- $\beta_2$ is the decay factor (around 0.999)
- $\nabla J(\theta)$ is the gradient of the loss w.r.t. the weights

Then, we can update the weights as follows:

$$
\theta = \theta - \frac{\eta}{\epsilon + \sqrt{v_t}} \cdot m_t
$$

Where:

- $\eta$ is the learning rate
- $\epsilon$ is a small number (e.g. $10^{-7}$), to avoid division by zero

## Resources

- [📺 YouTube − Neural Networks from Scratch](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3) : understand the various components of a neural network and how to create one in a modular way.
- [📺 YouTube − Backpropagation, step-by-step | DL3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) : understand how backpropagation works.
- [📺 YouTube − Backpropagation calculus | DL4](https://www.youtube.com/watch?v=tIeHLnjs5U8) : understand the math behind backpropagation.
- [📺 YouTube − Backpropagation Algorithm | Neural Networks](https://www.youtube.com/watch?v=sIX_9n-1UbM) : have a good insight of the chain rule's logic in backpropagation.
- [📺 YouTube − Neural Network from Scratch | Mathematics & Python Code](https://www.youtube.com/watch?v=pauPCy_s0Ok) : understand the actual implementation of backpropagation in code.
- [📖 Medium − Neural Network from scratch in Python](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65) : the corresponding article to the previous video.
- [📖 Pinecone − Cross-Entropy Loss: make predictions with confidence](https://www.pinecone.io/learn/cross-entropy-loss/#Derivative-of-the-Softmax-Cross-Entropy-Loss-Function) : the full derivation of the softmax + cross-entropy loss.
- [📺 YouTube − Optimization for Deep Learning (Momentum, RMSprop, AdaGrad, Adam)](https://www.youtube.com/watch?v=NE88eqLngkg)
- [📖 Medium − Understanding RMSProp: A Simple Guide to One of Deep Learning’s Powerful Optimizers](https://medium.com/@piyushkashyap045/understanding-rmsprop-a-simple-guide-to-one-of-deep-learnings-powerful-optimizers-403baeed9922)

- [💬 ]()
