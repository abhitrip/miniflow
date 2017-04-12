import numpy as np
class Node(object):
    def __init__(self,inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound node , add this node as outbound node
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        # A calculated value
        self.value = None

    # Placeholder for forward propagation
    def forward(self):
        """
        forward propagation
        Compute the output value based on inbound_nodes and
        store result in self.value
        """ 
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # An input Node has no inbound nodes
        # so no need to pass any to node instantiator
        Node.__init__(self)

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    #
    # of the previous node from self.inbound_nodes
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self,value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

class Add(Node):
    def __init__(self,x,y):
        # You could access `x` and `y` in forward with
        # self.inbound_nodes[0] (`x`) and self.inbound_nodes[1] (`y`)
        Node.__init__(self,[x,y])
    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of its inbound_nodes.
        """
        xvalue = self.inbound_nodes[0].value
        yvalue = self.inbound_nodes[1].value
        self.value = xvalue+yvalue


class Linear(Node):
    """
    Class to implement linear functionality
    """
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.
    def forward(self):
        """
        Set self.value to linear function's output 
        """
        inputs =  self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias    = self.inbound_nodes[2].value
        output = np.dot(inputs,weights)+bias
        # Non numpy solution
        # output = bias
        #for x,w in zip(inputs,weights):
        #    output+= x*w
        self.value = output
         

class Sigmoid(Node):
    """
    Class to implement Sigmoid functionality
    """
    def __init__(self,node):
        Node.__init__(self,[node])
    
    def _sigmoid(self,x):
        """
        This is different from forward() as it will be used in both forward and backward
        """
        return 1./(1.+np.exp(-x))
    
    def forward(self):
        """
        Set the value of this node to output of _sigmoid activation 
        """
        x = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)


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

