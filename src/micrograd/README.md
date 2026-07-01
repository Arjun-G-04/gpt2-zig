# How does this work?

- Value is a struct which contains the functionalities to store grad, dependant Values, etc.
- Value derived from another Value or two Values will store the pointers to those Value(s), the backward function needed to calculate the derivative, and misc.
- derivative of Values is calculated using chain rule. grad stores the rate of change of the final Value in the graph with change in current Value. grad of final Value is 1.
- topo sort is used to find the final nodes before each nodes. reversed list allows to calc grad for the furthest nodes, then the next furthest node and so on.
- a Neuron consists of weights and bias (all of which are Values).
- the Neuron can take an input list (x) of Values and calculate output by bias + sum(wi * xi).
- a Layer consists of a list of Neurons. it receives an input list, feeds the list to all the Neurons, and sends back the output list.
- a Multi-Layer Perceptron (MLP) is a list of Layers. it also receives an input list, feeds to each Layer, takes the output from each Layer and feeds it on to the next one.
- all operations done within the Neural Network components are via Value ops. thus, all operations are connected and it forms a Directed Acyclic Graph. 
- the NN is present in a persistent memory heap since all weights and biases must be present through each epoch. wherelese, when computing the values for each step, they are created in a ephemeral memory heap and cleared after each run. 
- IMPORTANT NOTE: the grad must be set to zero every run since its value must be calculated freshly via backpropagation.
- given input is passed to the MLP. using the output, a loss Value is determined. we run back propagation on this loss Value and apply negative grad times step to the weights and biases.
- it is repeated until loss is minimized.