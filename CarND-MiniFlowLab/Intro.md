#Basic Neural Network from scratch
## Introduction
In this project, we will build a library which will be a simplified version of TensorFlow! We will calll this library Miniflow. 

TensorFlow is one of the most popular open source neural network libraries, built by the team at Google Brain over just the last few years.

The goal is to demystify two concepts at the heart of neural networks - backpropagation and differentiable graphs.

Backpropagation is the process by which neural networks update the weights of the network over time. 

Differentiable graphs are graphs where the nodes are differentiable functions. They are also useful as visual aids for understanding and calculating complicated derivatives. This is the fundamental abstraction of TensorFlow - it's a framework for creating differentiable graphs.

With graphs and backpropagation, you will be able to create your own nodes and properly compute the derivatives. Even more importantly, you will be able to think and reason in terms of these graphs.

Now, let's take the first peek under the hood...

##NeuralNetworks and graphs
A neural network is a graph of mathematical functions such as linear combinations and activation functions. The graph consists of nodes, and edges.

Nodes in each layer (except for nodes in the input layer) perform mathematical functions using inputs from nodes in the previous layers. For example, a node could represent $f(x, y) = x + yf(x,y)=x+y$, where $xx$ and $yy$ are input values from nodes in the previous layer.

Similarly, each node creates an output value which may be passed to nodes in the next layer. The output value from the output layer does not get passed to a future layer (because it is the final layer).

Layers between the input layer and the output layer are called hidden layers.

The edges in the graph describe the connections between the nodes, along which the values flow from one layer to the next. These edges can also apply operations to the values that flow along them, such as multiplying by weights and adding biases. MiniFlow won't use a separate class for edges - instead, its nodes will perform both their own calculations and those of their input edges. This will be more clear as you go through these lessons.

###Forward Propagation
By propagating values from the first layer (the input layer) through all the mathematical functions represented by each node, the network outputs a value. This process is called a forward pass.
[Missingexample]
###Graphs
The nodes and edges create a graph structure. Though the example above is fairly simple, it isn't hard to imagine that increasingly complex graphs can calculate . . . well . . . almost anything.

There are generally two steps to create neural networks:

1.Define the graph of nodes and edges.
2.Propagate values through the graph.

MiniFlow will work the same way. We'll define the nodes and edges of our network with one method and then propagate values through the graph with another method. 
