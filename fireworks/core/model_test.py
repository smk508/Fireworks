from .model import Model, model_from_module
from fireworks.utils.exceptions import ParameterizationError
from fireworks.toolbox.pipes import BatchingPipe, LoopingPipe, ShufflerPipe, RepeaterPipe
from fireworks.utils.test_helpers import DummyModel, DummyMultilinearModel, LinearJunctionModel, LinearModule, RandomJunction, generate_linear_model_data, generate_multilinear_model_data, DummyUpdateModel, ComplexModel
from fireworks import Message, Junction
import random
import torch
from torch.nn import Parameter
from random import randint
import numpy as np
import os
import shutil

loss = torch.nn.MSELoss()

def match(A, B, equivalence = lambda a,b: a is b):
    """
    Returns all elements of A that are exactly the same as an element of B. ie. a is b for some b in B.
    You can change the equivalence relation used.
    """
    matches = []
    for a in A:
        for b in B:
            if equivalence(a, b):
                matches.append(a)
                break
    return matches

def unique(A):
    """
    Returns elements of A that are unique as defined by 'is' an equivalence relation.
    """
    unique = []
    for a1 in A:
        count = 0
        for a2 in A:
            if a1 is a2:
                count +=1
                if count > 1:
                    break
        if count == 1:
            unique.append(a1)

    return unique

def train_model(model, data, models = None, predicted='y', label='y_true'):

    # Initialize model for training
    # Define loss function and learning algorithm
    models = models or [model]
    parameters = [filter(lambda p: p.requires_grad, m.parameters()) for m in models]
    parameters = [x for y in parameters for x in y]
    optimizer = torch.optim.SGD(parameters, lr=.00015)
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        for batch in data:
            optimizer.zero_grad()
            result = model(batch.to_tensors())
            lo = loss(result[predicted], result[label])
            lo.backward()
            optimizer.step()

    return model

def test_Model_init():

    # Confirm that the model detects that parameters are missing.
    # try:
    #     damura = DummyModel()
    #     assert False
    # except ParameterizationError as error:
    #     assert True

    # Confirm that the model can be initialized
    # damura = DummyModel({'m': [3.]})
    #
    # assert (damura.m == 3.).all()
    # assert (damura.b == 0).all() # Default for 'b'

    damura = DummyModel({'m': [4.], 'b': [5.]})
    assert (damura.m == 4.).all()
    assert (damura.b == 5.).all()

def test_Model_get_set_state():
    sabura = DummyModel({'m': [2.], 'b': [3.]})
    mabura = DummyModel({'m': (sabura, 'm')})
    state1 = sabura.get_state()
    assert state1['external'] == {}
    assert (state1['internal'].keys()) == set(['out_column', 'in_column', 'b', 'm'])
    assert state1['internal']['b'] == 3.
    state2 = mabura.get_state()
    assert state2['external']['m'] == (sabura, 'm')
    assert (state2['internal'].keys()) == set(['out_column', 'in_column', 'b'])
    assert state2['internal']['b'] == 0.
    cabura = DummyModel()
    cabura.set_state(state1)
    cstate1 = cabura.get_state()
    assert cstate1['external'] == {}
    assert (cstate1['internal'].keys()) == set(['out_column', 'in_column', 'b', 'm'])
    assert cstate1['internal']['b'] == 3.
    cabura.set_state(state2)
    cstate2 = cabura.get_state()
    assert cstate2['external']['m'] == (sabura, 'm')
    assert (cstate2['internal'].keys()) == set(['out_column', 'in_column', 'b'])
    assert cstate2['internal']['b'] == 0.
    neuralnet = ComplexModel()
    state = neuralnet.get_state()
    neuralnet2 = ComplexModel()
    neuralnet2.set_state(state, reset=False)

def test_Model_save_load():

    # Construct model with inputs and components.
    sabura = DummyModel({'m': [0.]})
    sabura.__name__ = 'sabura'
    damura = DummyModel({'m': [2.]})
    damura.__name__ = 'damura'
    lamura = DummyModel()
    lamura.__name__ = 'lamura'
    babura = DummyModel({'m': (damura, 'm'), 'b': [4.], 'c': sabura, 'd': torch.nn.Conv2d(4,5,4)}, input=lamura)
    babura.__name__ = 'babura'

    if not os.path.isdir('save_test'):
        os.mkdir('save_test')
    else:
        shutil.rmtree('save_test')
        os.mkdir('save_test')
    babura.save(method='json', path='save_test/test.json')
    files = os.walk('save_test/.').__next__()[2]
    assert len(files) == 3
    # Test different save methods.
    newone = DummyModel({'m': [5.], 'd': torch.nn.Conv2d(4,5,4), 'c': DummyModel()})
    assert (newone.m == 5.).all()
    # old = newone.state_dict()['d.bias'].clone().detach().numpy()
    newone.load_state('save_test/torch_babura-test.json')
    assert (newone.d.bias - babura.d.bias < .005).all()
    assert (newone.d.weight - babura.d.weight < .005).all()
    # new = newone.state_dict()['d.bias'].clone().detach().numpy()
    shutil.rmtree('save_test')

def test_Model_inferencing():

    damura = DummyModel({'m': [2.]})
    x = Message({'x': torch.Tensor([1,2,3])})
    y = damura(x)
    assert y == x
    assert (y['x'] == torch.Tensor([1,2,3])).all()
    assert (y['y'] == torch.Tensor([2.,4.,6.])).all()

def test_ModelFromModule():

    pop = LinearModule()
    mom = model_from_module(LinearModule)
    messi = Message({'x':torch.Tensor([1,2,3])})
    pom = mom()
    assert set(pom.components) == set(['conv1', 'm', 'b'])
    result = pom(messi)
    y = result['y']
    x = result['x']
    assert 'y' in result and 'x' in result and 'z' not in result
    # Test that Model calls are recursive on inputs
    hom = DummyModel({'m': [3.]}, out_column='z')
    pom.input = hom
    result = pom(messi)
    assert (x == result['x']).all()
    assert (y == result['y']).all()
    assert 'y' in result and 'x' in result and 'z' in result

def test_save_model():

    A = DummyModel({'m': [0.], 'ok': torch.nn.Conv2d(1,20,5), 'c': 'yes'})
    A.save(method='json')
    A.save(method='html')
    saved = A.save(method='dict')
    assert type(saved) is dict
    assert set(saved.keys()) == set(['out_column', 'm', 'in_column', 'ok', 'c', 'b'])

def test_freeze_and_unfreeze():

    A = DummyModel({'m': [0.]})
    assert A.m.requires_grad == True
    assert A.b.requires_grad == True
    A.freeze()
    assert A.m.requires_grad == False
    assert A.b.requires_grad == False
    A.unfreeze()
    assert A.m.requires_grad == True
    assert A.b.requires_grad == True
    A.freeze('m')
    assert A.m.requires_grad == False
    assert A.b.requires_grad == True
    A.unfreeze('m')
    assert A.m.requires_grad == True
    assert A.b.requires_grad == True
    A.freeze('b')
    assert A.m.requires_grad == True
    assert A.b.requires_grad == False
    A.unfreeze('b')
    assert A.m.requires_grad == True
    assert A.b.requires_grad == True

    B = model_from_module(LinearModule)() # This one has a PyTorch module Conv1 in it as well.
    assert B.m.requires_grad == True
    assert B.b.requires_grad == True
    assert B.conv1.weight.requires_grad == True
    assert B.conv1.weight.requires_grad == True
    B.freeze()
    assert B.m.requires_grad == False
    assert B.b.requires_grad == False
    assert B.conv1.weight.requires_grad == False
    assert B.conv1.weight.requires_grad == False
    B.unfreeze('conv1')
    assert B.m.requires_grad == False
    assert B.b.requires_grad == False
    assert B.conv1.weight.requires_grad == True
    assert B.conv1.weight.requires_grad == True
    B.freeze('conv1')
    assert B.m.requires_grad == False
    assert B.b.requires_grad == False
    assert B.conv1.weight.requires_grad == False
    assert B.conv1.weight.requires_grad == False
    B.unfreeze(['m','b'])
    assert B.m.requires_grad == True
    assert B.b.requires_grad == True
    assert B.conv1.weight.requires_grad == False
    assert B.conv1.weight.requires_grad == False

def get_minibatcher(training_data):

    repeater = RepeaterPipe(training_data)
    lol = LoopingPipe(repeater)
    shuffler = ShufflerPipe(lol)
    minibatcher = BatchingPipe(shuffler, batch_size=50)

    return minibatcher

def test_one_Model_training():

    A = DummyModel({'m': [0.]})
    B = model_from_module(LinearModule)()
    training_data = generate_linear_model_data()
    m = training_data[1]['m']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    train_model(A, minibatcher)
    # For some reason, this model struggles to learn the y-intercept.
    assert (m-A.m < .6).all()
    train_model(B, minibatcher)
    assert (m - B.m < .6).all()

    assert (A.m - B.m < .6).all() # Test precision between models

def test_multiple_Models_training():
    """
    Here, we compose a model multilinear = A + B + C, where A, B, and C represent
    different components of the overall model.
    """
    A = DummyModel({'m': [0.]})
    A.freeze('b')
    B = model_from_module(LinearModule)()
    B.freeze('b')
    C = DummyModel({'m': [0.]})
    C.freeze('m')
    multilinear = DummyMultilinearModel({'m1':A.m, 'm2': B.m, 'b': C.b})
    assert multilinear.m1 is A.m
    assert multilinear.m2 is B.m
    assert multilinear.b is C.b
    training_data = generate_multilinear_model_data()
    m1 = training_data[1]['m1']
    m2 = training_data[1]['m2']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    train_model(multilinear, minibatcher)
    assert (A.m - m1 < .6).all()
    assert (B.m - m2 < .6).all()
    assert (A.b == 0).all()
    assert (C.m == 0.).all()

def test_setattr_in_pipeline():
    """
    Setattribute should not trigger any recursive actions.
    """
    A = DummyModel({'m': [3.]}, out_column='y1')
    B = DummyModel({'m': [1.], 'b': [2.]}, input=A, in_column='y1', out_column='y')
    A.ok = 0
    B.ok = 1
    assert A.ok == 0
    assert B.ok == 1
    assert (A.m == 3.).all()
    assert (B.m == 1.).all()
    # Test that Parameters and subModules automatically get added as components.
    A.yes = Parameter(torch.Tensor([3.,4.]))
    assert 'yes' in A.components
    assert 'yes' not in B.components
    assert B.yes is A.yes

def test_all_parameters():

    # Pipe layout
    A = DummyModel({'m': [3.]}, out_column='y1')
    B = DummyModel({'m': [1.], 'b': [2.]}, input=A, in_column='y1', out_column='y')
    # assert len(unique(B.all_parameters())) == len(B.all_parameters())
    # matches = match(B.all_parameters(), [B.m, B.b, A.m, A.b])
    # assert len(matches) == 4

    # Component layout
    A = DummyModel({'m': [0.]})
    B = model_from_module(LinearModule)()
    C = DummyModel({'m': [1.]})
    multilinear = DummyMultilinearModel({'m1':(A, 'm'), 'm2': (B, 'm'), 'b': (C, 'm'), 'ok': [1.]})
    multilinear.state_dict()
    multilinear.hi = torch.nn.Conv2d(4,4,4)
    params = multilinear.all_parameters()
    assert len(params) == 6

    for param in params: # See if teh Conv2d from above had it's parameters registered.
        if param.shape == (4,4,4,4,):
            test = True
    assert test


    multilinear2 = DummyMultilinearModel()
    multilinear2.hi = torch.nn.Conv2d(4,4,4)
    state = multilinear.get_state()
    multilinear2.set_state(state, reset=False)
    matches = match(multilinear.all_parameters(), multilinear2.all_parameters(), equivalence = lambda a,b: (a==b).all())
    assert len(matches) == 6

    assert len(unique(multilinear.all_parameters())) == len(multilinear.all_parameters())
    matches = match(multilinear.all_parameters(), [A.m, B.m, C.m])
    assert len(matches) == 3

    # Component layout
    A = DummyModel({'m': [1.],'b':[3.]}, out_column='z')
    B = LinearJunctionModel(components={'b':[.5], 'f':A})
    assert len(unique(B.all_parameters())) == len(B.all_parameters())
    matches = match(B.all_parameters(), [A.m, A.b, B.b])
    assert len(matches) == 3

    # Pipes and components
    A = DummyModel({'m': [1.],'b':[3.]}, out_column='z')
    C = DummyModel({'m': [0.]})
    B = LinearJunctionModel(components={'b':[.5], 'f':A}, input=C)

    assert len(unique(B.all_parameters())) == len(B.all_parameters())
    matches = match(B.all_parameters(), [A.m, A.b, B.b, C.m, C.b])
    assert len(matches) == 5

def test_multiple_Models_training_in_pipeline():
    """
    Here, model A pipes its output into B
    """
    A = DummyModel({'m': [3.]}, out_column='y1')
    B = DummyModel({'m': [1.], 'b': [2.]}, input=A, in_column='y1', out_column='y')
    A.freeze('b')
    B.freeze('m')
    training_data = generate_linear_model_data()
    m = training_data[1]['m']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    assert (A.m == 3.).all()
    assert (B.m == 1).all()
    assert (A.b == 0).all()
    assert (B.b == 2.).all()
    batch = minibatcher.__next__()
    batch.to_tensors()
    A(batch)
    train_model(B, minibatcher, models = [A, B])
    assert (A.m - m < .6).all()
    assert (B.b != 2).all()
    assert (B.m == 1).all()
    assert (A.b == 0).all()

def test_multiple_Models_training_in_junction():
    """
    Here, model A is provided as a component of B
    """
    A = DummyModel({'m': [1.],'b':[3.]}, out_column='z')
    B = LinearJunctionModel(components={'b':[.5], 'f':A})
    training_data = generate_linear_model_data()
    m = training_data[1]['m']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    batch = minibatcher.__next__()
    batch.to_tensors()
    B(batch)
    assert (A.m == 1.).all()
    train_model(B, minibatcher, models=[A, B])
    assert (A.m != 1).all()

def test_multiple_Models_training_via_junction():
    """
    Here, model B takes a junction A as input that randomly calls one of C, D, or E
    Hence, B implements y = f(x) + b, where f is randomly C, D, or E
    """
    C = DummyModel({'m': [1.],'b':[0.]}, out_column='z')
    D = DummyModel({'m': [2.],'b':[0.]}, out_column='z')
    E = DummyModel({'m': [3.],'b':[0.]}, out_column='z')
    C.freeze('b')
    D.freeze('b')
    E.freeze('b')
    A = RandomJunction(components={'C':C, 'D': D, 'E':E})
    B = LinearJunctionModel(components={'b': [3.], 'f':A})
    training_data = generate_linear_model_data(n=750)
    m = training_data[1]['m']
    b = training_data[1]['b']
    errors = training_data[1]['errors']
    minibatcher = get_minibatcher(training_data[0])
    batch = minibatcher.__next__()
    batch.to_tensors()
    banana = B(batch)['y']
    rambo = False
    for i in range(20):
        bonana = B(batch)['y']
        if (banana != bonana).all():
            rambo = True
            break
    assert rambo
    train_model(B, minibatcher, models=[B, C, D, E])
    # Test that all Models trained
    assert (C.m - m < .6).all()
    assert (D.m - m < .6).all()
    assert (E.m - m < .6).all()
    assert (C.m != D.m).all()
    assert (D.m != E.m).all()
    assert (E.m != C.m).all()

def test_enable_disable():

    def get_dummys():
        """
        This Model increments it's internal _count variable by the length of the Message being accessed on update.
        On inference, it simply appends the input to itself and returns that.
        """
        dummy = DummyUpdateModel()
        tummy = DummyUpdateModel(input=dummy)
        assert dummy._count == 0
        assert tummy._count == 0
        return dummy, tummy

    data, meta = generate_linear_model_data()
    batch = data[2:10]
    l = len(batch)

    # Test updates and inference for one model
    dummy, tummy = get_dummys()
    out = dummy(batch)
    assert dummy._count == l
    assert tummy._count == 0
    assert out == batch.append(batch)

    # Test updates and inference for multiple Models
    dummy, tummy = get_dummys()
    out = tummy(batch)
    assert dummy._count == l
    assert tummy._count == 2*l
    assert out == batch.append(batch).append(batch).append(batch) # 4x

    # Test disable updates for one Model
    dummy, tummy = get_dummys()
    tummy.disable_updates()
    out = tummy(batch)
    assert dummy._count == l
    assert tummy._count == 0
    assert out == batch.append(batch).append(batch).append(batch) # 4x

    # Test disable updates for all Models recursively
    dummy, tummy = get_dummys()
    tummy.disable_updates_all()
    out = tummy(batch)
    assert dummy._count == 0
    assert tummy._count == 0
    assert out == batch.append(batch).append(batch).append(batch) # 4x
    # Test reenabling updates for one Model
    tummy.enable_updates()
    out = tummy(batch)
    assert dummy._count == 0
    assert tummy._count == 2*l
    assert out == batch.append(batch).append(batch).append(batch) # 4x
    # Test reenabling updates for all Models
    tummy.enable_updates_all()
    out = tummy(batch)
    assert dummy._count == l
    assert tummy._count == 4*l
    assert out == batch.append(batch).append(batch).append(batch) # 4x

    # Test disable inference for one Model
    dummy, tummy = get_dummys()
    tummy.disable_inference()
    out = tummy(batch)
    assert dummy._count == l
    assert tummy._count == 2*l
    assert out == batch.append(batch) # 2x from the first one

    # Test disable inference for all Models recursively
    dummy, tummy = get_dummys()
    tummy.disable_inference_all()
    out = tummy(batch)
    assert dummy._count == l
    assert tummy._count == l
    assert out == batch # Identity
    # Test renabling inference for one Model
    tummy.enable_inference()
    out = tummy(batch)
    assert dummy._count == 2*l
    assert tummy._count == 2*l
    assert out == batch.append(batch) # 2x from the first one
    # Test renabling inference for all Models
    tummy.enable_inference_all()
    out = tummy(batch)
    assert dummy._count == 3*l
    assert tummy._count == 4*l
    assert out == batch.append(batch).append(batch).append(batch) # 4x from the first one

def test_cuda_cpu():

    if not torch.cuda.is_available(): # This test will only work on a system with CUDA enabled.
        return
    A = DummyModel({'m': [3.]}, out_column='y1')
    B = DummyModel({'m': [1.], 'b': [2.]}, input=A, in_column='y1', out_column='y')
    devices = set([x.device for x in B.state_dict().values()])
    assert len(devices) == 1
    assert devices.pop().type == 'cpu'
    devices = set([x.device for x in A.state_dict().values()])
    assert len(devices) == 1
    assert devices.pop().type == 'cpu'
    B.cuda()
    devices = set([x.device for x in B.state_dict().values()])    
    assert len(devices) == 1
    assert devices.pop().type == 'cuda'
    devices = set([x.device for x in A.state_dict().values()])
    assert len(devices) == 1
    assert devices.pop().type == 'cuda'
    B.cpu()
    devices = set([x.device for x in B.state_dict().values()])    
    assert len(devices) == 1
    assert devices.pop().type == 'cpu'
    devices = set([x.device for x in A.state_dict().values()])
    assert len(devices) == 1
    assert devices.pop().type == 'cpu'
    A.cuda()
    devices = set([x.device for x in B.state_dict().values()])    
    assert len(devices) == 1
    assert devices.pop().type == 'cpu'
    devices = set([x.device for x in A.state_dict().values()])
    assert len(devices) == 1
    assert devices.pop().type == 'cuda'
