import numpy as np


class Node(object):
    """Base class that defines a node of a graph."""

    def __init__(self, inbound_nodes=[]):
        # Nodes from which this node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this node passes values
        self.outbound_nodes = []
        # For each node in the inbound node, add the present node as an
        # outbound node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
            # Output value of the node
            self.value = None

    def forward(self):
        """
        Forward propagation method

        Compute the output value based on inbound nodes and store the output in
        self.value.

        This method will depend on the type of node and thus will be
        implemented case-by-case in subclasses.
        """
        raise NotImplemented(
            'Forward method should not be called for the base class Node. This is probably due to a wrong definition of the node.')


class Input(Node):
    """Subclass specifically for input nodes. Performs no calculation and has
    no input node."""

    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    def forward(self, value=None):
        # NOTE: Input node is the only node where the value
        # may be passed as an argument to forward().
        #
        # All other node implementations should get the value
        # of the previous node from self.inbound_nodes
        #
        # Example:
        # val0 = self.inbound_nodes[0].value

        # Overwrite value if one is passed in.
        if value is not None:
            self.value = value


class Add(Node):
    """Subclass specific for the addition of two values."""

    def __init__(self, *inputs):
        # there are two inbounds nodes in this case.
        Node.__init__(self, inputs)

    def forward(self):
        # The output value is defined as the sum of the values of the inbound
        # nodes
        self.value = 0
        for n in self.inbound_nodes:
            self.value += n.value

    def backward(self):
        # To be implmented
        # TODO : implement backward method for add node.
        raise NotImplemented


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        inputs = self.inbound_nodes[0]
        weights = self.inbound_nodes[1]
        bias = self.inbound_nodes[2]
        self.value = np.dot(inputs.value, weights.value) + bias.value


class Sigmoid(Node):
    """
    Method That implements a sigmoid node. Has only one inbound node
    and produces the sigmoid of the input value as output.
    """

    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used later with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        return 1/(1+np.exp(-x))

    def forward(self):
        """
        Set the value of this node to the result of the
        sigmoid function, `_sigmoid`.

        """
        self.value = self._sigmoid(self.inbound_nodes[0].value)


class Relu(Node):
    """
    Method That implements a relu node. Has only one inbound node
    and produces max(0,input) as output.
    """

    def __init__(self, node):
        Node.__init__(self, [node])

    def forward(self):
        """
        Set the value of this node to the input if it is positive, to 0 other
        wise.
        """
        self.value = np.max(0,self.inbound_nodes[0].value)


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.


    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
