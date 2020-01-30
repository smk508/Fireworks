from fireworks.extensions import explain as dx
import torch
from torch.nn import Module, Parameter
from torch.nn.modules import Linear, ReLU, CELU, Sigmoid
import numpy as np
import matplotlib.pyplot as plt 
from itertools import count 

class SimpleModel(Module):

    def __init__(self, activation):
        super().__init__()
        self.layer1 = Linear(2,2)
        self.layer1.weight = Parameter(torch.Tensor([[1.0, -1.0], [-1.0, 1.0]]))
        self.layer1.bias = Parameter(torch.Tensor([1.5, -1.0]))
        self.layer2 = Linear(2,2)         
        self.layer2.weight = Parameter(torch.Tensor([[1.1, 1.4], [-.5, 1.0]]))
        self.layer2.bias = Parameter(torch.Tensor([0.0, 2.0]))        
        self.activation = activation 

    def forward(self, X):
        super().__init__()
        out = self.layer1(X)
        out = self.activation(out)
        out = self.layer2(out)
        return out 

class LinearModel(Module):
    """
    Implements y = m*x+b
    """
    def __init__(self,m=3.,b=4.):
        super().__init__()
        self.m = Parameter(torch.tensor([m]), requires_grad=False) 
        self.b = Parameter(torch.tensor([b]), requires_grad=False)

    def forward(self, X):
        return self.m * X + self.b 

class MultiLinearModel(Module):
    """
    Implements y = m*x+b
    """
    def __init__(self,m=3.,b=4.):
        super().__init__()
        self.m = Parameter(torch.tensor([m]), requires_grad=False) 
        self.b = Parameter(torch.tensor([b]), requires_grad=False)

    def forward(self, X):

        return torch.sum(self.m * X + self.b, 1)


class SimplerModel(Module):
    """
    Implements ReLU( ReLU(x1-1) - ReLU(x2) )
    .
    .
    """
    def __init__(self):
        super().__init__()
        self.activation = ReLU()
        self.layer1 = Linear(2,2)
        self.layer1.weight = Parameter(torch.Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=False))
        self.layer1.bias = Parameter(torch.Tensor([-1.0, 0], requires_grad=False))
        self.layer2 = Linear(2,2, bias=False)
        self.layer2.weight = Parameter(torch.Tensor([[1.0], [-1.0]], requires_grad=False))
        

    def forward(self, X):

        out = self.layer1(X)
        out = self.activation(out)
        out = self.layer2(X)
        return out 

class MinModel(Module):
    """
    Implements min(xi)
    """
    
    def forward(self, X):
        
       return torch.min(v)

class SimpleMultiInputsModel(Module):
    """
    Implements ReLu (3*x1|2*x2) | is a concat op
    """
    def __init__(self):
        super().__init__()
        self.layer_a = Linear(2,2, bias=False)      
        self.layer_a.weight = Parameter(torch.Tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=False))
        self.layer_b = Linear(2,2, bias=False)
        self.layer_b.weight = Parameter(torch.Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=False))
        self.activation = Sigmoid

    def forward(self, X1, X2):
        a = self.layer_a(X1)
        b = self.layer_b(X2)
        cat = torch.concat(a, b, 1)
        out = self.activation(cat)
        return out 

class XORModel(Module):
    """
    Preseeded model for training a XOR function.
    """
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(2,2)
        self.layer1.weight = Parameter(torch.Tensor([[0.10711301, -0.0987727], [-1.57625198, 1.34942603]]))
        self.layer1.bias = Parameter(torch.Tensor([-0.30955192, -0.14483099]))
        self.layer2 = Linear(2,1)
        self.layer2.weight = Parameter(torch.Tensor([[0.69259691], [-0.16255915]]).reshape(1,-1))
        self.layer2.bias = Parameter(torch.Tensor([1.53952825]))
        self.activation = CELU()

    def forward(self, X):

        out = self.layer1(X)
        out = self.activation(out) 
        out = self.layer2(out)
        out = self.activation(out)

        return out

def TrainXOR():

    # Construct model and criterion
    model = XORModel()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    all_params = list(model.parameters())
    # Generate random dataset 
    
    x = np.random.randint(0, 2, size=(1000, 2))
    y = np.expand_dims(np.logical_or(x[:, 0], x[:, 1]), -1)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    model.train()
    if torch.cuda.is_available():
        model.cuda()
        x = x.cuda()
        y = y.cuda()
    initial_loss = loss(model(x), y)
    batch_size = 100
    
    for i in range(1000):
        optimizer.zero_grad()
        start = i*batch_size % 1000
        end = (i+1)*batch_size % 1000        
        if start > end:
            input = torch.cat((x[start:],x[0:end]))
            reference = torch.cat((y[start:], y[0:end]))
        else:
            input = x[start:end]
            reference = y[start:end]
        output = model(input)
        current_loss = loss(output, reference)
        current_loss.backward()
        
        optimizer.step() 

    final_loss = current_loss.detach().cpu().numpy()
    
    return model, (np.abs(final_loss) < 0.01)


def test_GradientxInput_linear_model():
    
    m = 5.
    model = LinearModel(m=m)
    X = np.random.randint(0, 2, size=(10, 2))
    X = torch.Tensor(X)
    if torch.cuda.is_available():
        model.cuda()
        model.m = model.m.cuda()
        model.b = model.b.cuda()
        X = X.cuda() 
    
    Y =  model(X)
    
    
    if torch.cuda.is_available():
        Y = Y.cuda()

    
    attributions = dx.GradientxInput(model, X)
    
    # Check that the attributions are correct
    for x, y, row in zip(X, Y, count()):
        if x[0] == 0 :
            assert (attributions[row,0] == torch.tensor([0., 0.])).all()
        else:
            assert (attributions[row,0] == torch.tensor([m, 0.])).all()
        if x[1] == 0:
            assert (attributions[row,1] == torch.tensor([0., 0.])).all()
        else:
            assert (attributions[row,1] == torch.tensor([0., m])).all()


def test_GradientxInput_multilinear_model():

    m = 5.
    model = MultiLinearModel(m=m)
    X = np.random.randint(0, 2, size=(10, 2))
    X = torch.Tensor(X)

    if torch.cuda.is_available():
        model.cuda()
        model.m = model.m.cuda()
        model.b = model.b.cuda()
        X = X.cuda() 

    Y =  model(X)
    
    if torch.cuda.is_available():
        Y = Y.cuda()


    attributions = dx.GradientxInput(model, X)
    
    X = X.cpu()
    # Check that the attributions are correct 
    for x, y, row in zip(X, Y, count()):
        if (x == torch.Tensor([0., 0.])).all():
            assert (attributions[row] == torch.tensor([0., 0.])).all()
        elif (x == torch.Tensor([0., 1.])).all():
            assert (attributions[row] == torch.tensor([0., m])).all()
        elif (x == torch.Tensor([1.,0.])).all():
            assert (attributions[row] == torch.tensor([m, 0.])).all()
        elif (x == torch.Tensor([1., 1.])).all():
            assert (attributions[row] == torch.tensor([m, m])).all()

            
def test_GradientxInput_nn_model():
    
    X = np.random.randint(0, 2, size=(10, 2))    
    Y = np.expand_dims(np.logical_or(X[:, 0], X[:, 1]), -1)
    
    X = torch.FloatTensor(X.astype(float))
    
    Y = torch.tensor(Y)

    model, is_trained = TrainXOR()
    if torch.cuda.is_available():
        model.cuda()
        X = X.cuda()
        Y = Y.cuda()

    assert is_trained

    attributions = dx.GradientxInput(model, X)
    assert attributions.shape == torch.Size([10,1,2])

def test_IntegratedGradients_linear_model():
    """ For linear models, Integrated Gradients is equivalent to GxI """
    m = 5.
    model = LinearModel(m=m)
    X = np.random.randint(0, 2, size=(10, 2))
    X = torch.Tensor(X)
    if torch.cuda.is_available():
        model.cuda()
        model.m = model.m.cuda()
        model.b = model.b.cuda()
        X = X.cuda() 
    
    Y =  model(X)
    
    
    if torch.cuda.is_available():
        Y = Y.cuda()

    
    attributions = dx.IntegratedGradients(model, X)
    
    # Check that the attributions are correct
    for x, y, row in zip(X, Y, count()):
        if x[0] == 0 :
            assert (attributions[row,0] == torch.tensor([0., 0.])).all()
        else:
            assert (attributions[row,0] == torch.tensor([m, 0.])).all()
        if x[1] == 0:
            assert (attributions[row,1] == torch.tensor([0., 0.])).all()
        else:
            assert (attributions[row,1] == torch.tensor([0., m])).all()


def test_IntegratedGradients_multilinear_model():
    """ For linear models, Integrated Gradients is equivalent to GxI """
    m = 5.
    model = MultiLinearModel(m=m)
    X = np.random.randint(0, 2, size=(10, 2))
    X = torch.Tensor(X)

    if torch.cuda.is_available():
        model.cuda()
        model.m = model.m.cuda()
        model.b = model.b.cuda()
        X = X.cuda() 

    Y =  model(X)
    
    if torch.cuda.is_available():
        Y = Y.cuda()

    attributions = dx.GradientxInput(model, X)
    
    X = X.cpu()
    # Check that the attributions are correct 
    for x, y, row in zip(X, Y, count()):
        if (x == torch.Tensor([0., 0.])).all():
            assert (attributions[row] == torch.tensor([0., 0.])).all()
        elif (x == torch.Tensor([0., 1.])).all():
            assert (attributions[row] == torch.tensor([0., m])).all()
        elif (x == torch.Tensor([1.,0.])).all():
            assert (attributions[row] == torch.tensor([m, 0.])).all()
        elif (x == torch.Tensor([1., 1.])).all():
            assert (attributions[row] == torch.tensor([m, m])).all()

def test_IntegratedGradients_nn_model():
    """ Integrated Gradients should not be equivalent to GxI for neural nets. """
    X = np.random.randint(0, 2, size=(10, 2))    
    Y = np.expand_dims(np.logical_or(X[:, 0], X[:, 1]), -1)
    
    X = torch.FloatTensor(X.astype(float))
    
    Y = torch.tensor(Y)

    model, is_trained = TrainXOR()
    if torch.cuda.is_available():
        model.cuda()
        X = X.cuda()
        Y = Y.cuda()

    assert is_trained

    attributions = dx.IntegratedGradients(model, X)
    attributions2 = dx.GradientxInput(model, X)
    assert attributions.shape == torch.Size([10,1,2])
    assert attributions2.shape == torch.Size([10,1,2])
    assert not (attributions == attributions2).all()

def test_Occlusion_1_linear_model():
    """ For linear models, Occlusion-1 is equivalent to GxI """
    m = 5.
    model = LinearModel(m=m)
    X = np.random.randint(0, 2, size=(10, 2))
    X = torch.Tensor(X)
    if torch.cuda.is_available():
        model.cuda()
        model.m = model.m.cuda()
        model.b = model.b.cuda()
        X = X.cuda() 
    
    Y =  model(X)
    
    
    if torch.cuda.is_available():
        Y = Y.cuda()

    attributions = dx.Occlusion_1(model, X)
    # Check that the attributions are correct
    for x, y, row in zip(X, Y, count()):        
        if x[0] == 0 :
            assert (attributions[row,0] == torch.tensor([0., 0.])).all()
        else:
            assert (attributions[row,0] == torch.tensor([m, 0.])).all()
        if x[1] == 0:
            assert (attributions[row,1] == torch.tensor([0., 0.])).all()
        else:
            assert (attributions[row,1] == torch.tensor([0., m])).all()

def test_Occlusion_1_multilinear_model():
    """ For linear models, Occlusion-1 is equivalent to GxI """
    m = 5.
    model = MultiLinearModel(m=m)
    X = np.random.randint(0, 2, size=(10, 2))
    X = torch.Tensor(X)

    if torch.cuda.is_available():
        model.cuda()
        model.m = model.m.cuda()
        model.b = model.b.cuda()
        X = X.cuda() 

    Y =  model(X)
    
    if torch.cuda.is_available():
        Y = Y.cuda()

    attributions = dx.Occlusion_1(model, X)
    
    X = X.cpu()
    # Check that the attributions are correct 
    for x, y, row in zip(X, Y, count()):
        if (x == torch.Tensor([0., 0.])).all():
            assert (attributions[row] == torch.tensor([0., 0.])).all()
        elif (x == torch.Tensor([0., 1.])).all():
            assert (attributions[row] == torch.tensor([0., m])).all()
        elif (x == torch.Tensor([1.,0.])).all():
            assert (attributions[row] == torch.tensor([m, 0.])).all()
        elif (x == torch.Tensor([1., 1.])).all():
            assert (attributions[row] == torch.tensor([m, m])).all()

def test_Occlusion_1_nn_model():
    """ Occlusion-1 should not be equivalent to GxI for neural nets. """
    X = np.random.randint(0, 2, size=(10, 2))    
    Y = np.expand_dims(np.logical_or(X[:, 0], X[:, 1]), -1)
    
    X = torch.FloatTensor(X.astype(float))
    
    Y = torch.tensor(Y)

    model, is_trained = TrainXOR()
    if torch.cuda.is_available():
        model.cuda()
        X = X.cuda()
        Y = Y.cuda()

    assert is_trained

    attributions = dx.Occlusion_1(model, X)
    attributions2 = dx.GradientxInput(model, X)
    assert attributions.shape == torch.Size([10,1,2])
    assert attributions2.shape == torch.Size([10,1,2])
    assert not (attributions == attributions2).all()