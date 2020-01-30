from abc import ABC, abstractmethod, abstractproperty
import torch
from torch.nn import Module, Parameter
from abc import abstractmethod
from fireworks.utils.exceptions import ParameterizationError
from .message import Message
from .junction import Junction, PyTorch_Junction
from .component_map import Component_Map, PyTorch_Component_Map
from .pipe import HookedPassThroughPipe, recursive
import os
import pandas as pd

class Model(HookedPassThroughPipe, Junction, ABC):
    """
    Represents a statistical model which has a set of components, and a
    means for converting inputs into outputs. The model functions like a Pipe
    with respect to the input/output stream, and it functions like a Junction
    with respect to the parameterization. components can be provided via multiple
    different sources in this way, providing flexibility in model configuration.
    Models can also provide components for other Models, enabling one to create
    complex graphs of Models that can be trained simultaneously or individually.
    """

    def __init__(self, components = {}, *args, input = None, **kwargs):
        """
        Args:
            components: A dict of components that the model can call on.
        """
        self._flags = {'recursive_get': 1, 'components_initialized': 0} # Used for controlling recursion. Don't mess with this.
        self.components = Component_Map(components, owner=self)
        HookedPassThroughPipe.__init__(self, input = input)

        self.init_default_components()
        self.enable_updates()
        self.enable_inference()
        self._flags['components_initialized'] = 1

    def init_default_components(self):
        """
        This method can optionally be implemented in order for the model to provide a default initialization for some or all of its
        required components.
        """
        pass

    @abstractmethod
    def forward(self, message):
        """
        Represents a forward pass application of the model to an input. Must be implemented by a subclass.
        This should return a Message.
        """
        pass

    def save(self, *args, method='json', **kwargs):
        """
        Aggregates this model's components into a single Message and saves them using the chosen method.
        You can use any method that Messages support writing to via the to_ method, and you can
        provide additional key-word arguments as needed to support this.
        If the save method involves a path, then the path will be modified for each component and the state_dict.
        For the state_dict, the path will be torch_{name}-{path}, and for each component it will be {key}_{path}
        where name is either the name of the class or self.__name__ if it is defined. key is the value of the key in
        the components dictionary.
        """
        # Get name of Model
        if hasattr(self, '__name__'):
            name = self.__name__
        else:
            name = type(self).__name__

        # Parse save path if provided
        if 'path' in kwargs:
            kwargs = kwargs.copy()
            path = kwargs['path']
            paths = path.split('/')
            paths[-1]="{0}-{1}".format(name, paths[-1])

        state = self.get_state()
        external = state['external']
        internal = state['internal']

        # Save internal state_dict
        if 'path' in kwargs:
            kwargs['path'] = os.path.join(*paths[:-1], 'torch_'+paths[-1])

        internal_state_as_message = Message.from_objects(internal)
        serialized = internal_state_as_message.to(method=method, **kwargs)

        external_components = set()
        external_dict = {}

        for key, value in external.items():
            if value[0] not in external_components:
                external_components.add(value[0])
                external_dict[key] = value[0]

        # Save other components. This should recursively trigger the same action on the components if necesssary.
        for key, component in (external_dict.items()):
            if hasattr(component, 'save'):
                if 'path' in kwargs:
                    kwargs['path'] = os.path.join(*paths[:-1], key+'_'+paths[-1])
                component.save(method=method, **kwargs)

        # Save inputs. This is done by default and can be disabled by providing 'recursive=False' as an argument.
        if ('recursive' in kwargs and kwargs['recursive']) or 'recursive' not in kwargs:

            if 'path' in kwargs:
                new_path = paths[-1].split('.')
                new_path[0] = '--' + new_path[0]
                new_path = '.'.join(new_path)
                kwargs['path'] = os.path.join(*paths[:-1], new_path)
            try:
                self.recursive_call('save',method=method, **kwargs)
            except AttributeError: #TODO: Make a custom RecursionEnd error to indicate that the recursion has ended in order to provide tighter error checking.
                pass

        return serialized

    def load_state(self, *args, method='json', **kwargs):
        """
        Loads the data in the given save file into the state dict.
        """
        if 'path' in kwargs:
            kwargs = kwargs.copy()
            kwargs['filepath'] = kwargs['path']
            del kwargs['path']

        methods = {
            'json': pd.read_json,
            'pickle': pd.read_pickle,
            'csv': pd.read_csv,
        }

        df = methods[method](*args, **kwargs) # Load parameters in
        state_dict = Message(df).to_dict()
        state_dict = {key: value[0] for key, value in state_dict.items()}
        self.set_state({'external':{}, 'internal': state_dict}, reset=False)

    def get_state(self):
        return self.components.get_state()

    def set_state(self, state, reset=True):

        if reset:
            self.components = Component_Map({}, owner=self)
            self.init_default_components()
        self.components.set_state(state)

    def update(self, message, **kwargs): pass

    def compile(self): pass

    # TODO: Implement description methods (__repr__, __str__)

    def _getitem_hook(self, message):

        self._update_hook(message, method='get')
        return self._forward_hook(message)

    def _next_hook(self, message):

        self._update_hook(message, method='next')
        return self._forward_hook(message)

    def _call_hook(self, message, *args, **kwargs):

        self._update_hook(message, method='call')
        return self._forward_hook(message, *args, **kwargs)

    def __call__(self, *args, **kwargs):

        return HookedPassThroughPipe.__call__(self, *args, **kwargs)

    def __getattr__(self, name):

        if self._flags['components_initialized'] and name in self.components:
            return self.components[name]

        if not name.startswith('_') and self._flags['recursive_get']:
            try:
                return HookedPassThroughPipe.__getattr__(self, name)
            except:
                pass

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    # def __setattr__(self, name, value):
    #     """
    #     Attribute modifications ignore the recursive aspect of Pipes.
    #     """
    #
    #     if name in ['_flags', 'input']:
    #         object.__setattr__(self, name, value)
    #     else:
    #         self._flags['recursive_get'] = 0
    #         Module.__setattr__(self, name, value)
    #         self._flags['recursive_get'] = 1
    #         if self._flags['components_initialized']:
    #             # self.update_components()
    #             self.components[name] = value

    def enable_inference(self):

        self._forward_hook = self.forward
        self._inference_enabled = True

    @recursive()
    def enable_inference_all(self):
        self.enable_inference()

    def disable_inference(self): #TODO: test this

        self._forward_hook = identity
        self._inference_enabled = False

    @recursive()
    def disable_inference_all(self):
        self.disable_inference()

    def enable_updates(self):

        self._updates_enabled = True
        self._update_hook = self.update

    @recursive()
    def enable_updates_all(self):
        self.enable_updates()

    def disable_updates(self):

        self._updates_enabled = True
        self._update_hook = identity

    @recursive()
    def disable_updates_all(self):
        self.disable_updates()

def identity(message, *args, **kwargs):

    return message

def freeze_module(module, parameters = None, submodules = None):
    """
    Recursively freezes the parameters in a PyTorch module.
    """
    _change_temperature(False, module, parameters, submodules)

def unfreeze_module(module, parameters = None, submodules = None):
    """
    Recursively unfreezes the parameters in a PyTorch module.
    """
    _change_temperature(True, module, parameters, submodules)

def _change_temperature(boo, module, parameters = None, submodules = None):
    """
    Changes the temperature of a PyTorch module.
    """
    parameters = parameters or module.parameters()
    submodules = submodules or module.modules()

    for parameter in parameters:
        parameter.requires_grad = boo
    for submodule in submodules:
        if submodule is not module:
            change_temperature(boo, submodule)

def model_from_module(module_class):
    """
    Given the class definition for a pytorch module, returns a model that
    encapsulates that module.
    """
    class ModelFromModule(module_class, PyTorch_Model):

        def __init__(self, components={}, *args, input=None, **kwargs):
            self._flags = {'recursive_get': 1, 'components_initialized': 0}
            module_class.__init__(self, *args, **kwargs)
            PyTorch_Model.__init__(self, components, input=input, skip_module_init=True)
            PyTorch_Model._sync_parameters(self)

        def __call__(self, message, *args, **kwargs):
            try: # This will trigger a recursive call if possible.
                message = self.input(message, *args, **kwargs)
            except:
                pass

            return self.forward(message, *args, **kwargs)

    return ModelFromModule

def to_parameter(component):
    """
    Attempts to convert a component to Pytorch Parameter if it is a tensor-like. This is required for using that component during model training.
    """
    if not isinstance(component, Parameter) and not isinstance(component, Module): # Convert to Parameter
        try:
            component = Parameter(torch.Tensor(component))
        except:
        # If the component is not a tensor-like, Parameter, or Module, then it is some other object that we simply attach to the model
        # For example, it could be a Pipe or Junction that the model can call upon.
            pass
    return component

class PyTorch_Model(Module, Model, PyTorch_Junction ):

    def __init__(self, components={}, *args, input=None, skip_module_init=False, **kwargs):

        self._flags = {'recursive_get': 1, 'components_initialized': 0} # Used for controlling recursion. Don't mess with this.

        if not skip_module_init: # This is so the ModelFromModule Class can work.
            Module.__init__(self)

        self.components = PyTorch_Component_Map({}, model=self)
        self.init_default_components()

        for key, value in components.items():
            self.components[key] = value

        HookedPassThroughPipe.__init__(self, input = input)
        self.enable_updates()
        self.enable_inference()
        self._flags['components_initialized'] = 1

    def __setattr__(self, name, value):
        """
        Attribute modifications ignore the recursive aspect of Pipes.
        """

        if name in ['_flags', 'input'] or name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            if self._flags['components_initialized']:
                # self.update_components()
                self.components[name] = value
            self._flags['recursive_get'] = 0
            Module.__setattr__(self, name, value)
            self._flags['recursive_get'] = 1

    def __getattr__(self, name):

        return Model.__getattr__(self, name)

    def __call__(self, *args, **kwargs):

        return Model.__call__(self, *args, **kwargs)

    def state_dict(self):

        state_dict = {
            key: value for key, value in Module.state_dict(self).items()
            if key not in self.components._external_attribute_names.keys()
            }

        return state_dict

    def all_parameters(self):
        """
        Returns a list of every PyTorch parameter that this Model depends on that is unfrozen.
        This is useful for providing a parameters list to an optimizer.
        """

        all_params = []
        for param in self.components.values():
            if type(param) is Parameter:
                all_params.append(param)
            elif isinstance(param,Module):
                all_params.extend(list(param.parameters()))

        if hasattr(self, 'input') and self.input is not None:
            try:
                all_params.extend(self.input.all_parameters())
            except AttributeError:
                pass

        return all_params

    def _sync_parameters(self): # TODO: Test this.
        """
        Syncs the Parameters and Modules associated with this object with the component map.
        """
        for key, value in self._parameters.items():
            self.components[key] = value
        for key, value in self._modules.items():
            self.components[key] = value

    def set_state(self, state, reset=True):

        if reset:
            self.components = PyTorch_Component_Map({}, model=self)
            self.init_default_components()
        self.components.set_state(state)

    def get_state(self):
        """
        Returns state after serializing internal state.
        """
        state = self.components.get_state()
        serialized_state = {'external': state['external'], 'internal': {}}

        for key, value in state['internal'].items():
            if isinstance(value, Model): # Serialize the Model recursively and use that as the value.
                serialized_state['internal'][key] = value.get_state()
            elif isinstance(value, Module): # Serialize the submodule and use that as the value.
                state_dict = value.state_dict()
                for k, v in state_dict.items():
                    state_dict[k] = v.cpu().detach().numpy()
                serialized_state['internal'][key] = state_dict

            else:
                if type(value) in [torch.Tensor, torch.nn.Parameter]:
                    value = value.cpu().detach().numpy()
                serialized_state['internal'][key] = value

        return serialized_state

    @recursive()
    def cuda(self, *args, **kwargs):
        """ Recursively moves this model and its inputs to GPU memory. The allowed arguments are the same as torch.nn.Module.cuda() """
        return Module.cuda(self, *args, **kwargs)

    @recursive()
    def cpu(self, *args, **kwargs):
        """ Recursively moves this model and its inputs to GPU memory. The allowed arguments are the same as torch.nn.Module.cuda() """
        return Module.cpu(self, *args, **kwargs)

    def freeze(self, components = None):
        """
        Freezes the given components of the model (or all of them if none are specified) in order to prevent gradient updates.
        This means setting requires_grad to false for specified components so that these components
        are not updated during training.
        """
        self._change_temperature(False, components)
        
    def unfreeze(self, components = None):
        """
        Unfreezes the given components of the model (or all of them if none are specified) in order to prevent gradient updates.
        This means setting requires_grad to true for specified components so that these components
        are updated during training.
        """
        self._change_temperature(True, components)

    def _change_temperature(self, boo, components = None):

        if isinstance(components, str): # eg. a single component is provided as a string
            components = [components]

        if components is None:
            components = self.components

        for component in components:
            if isinstance(getattr(self, component), Parameter):
                getattr(self, component).requires_grad = boo
            elif isinstance(getattr(self,component), Model):
                getattr(self,component)._change_temperature(boo) # Recursively freezes Models
            elif isinstance(getattr(self,component), Module): # Is a PyTorch module but not a model.
                _change_temperature(boo, getattr(self,component))
