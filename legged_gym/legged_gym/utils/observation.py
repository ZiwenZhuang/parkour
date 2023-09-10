from collections import OrderedDict
import numpy as np

def get_obs_slice(segments: OrderedDict, component_name: str):
    """ Get the slice from segments and name. Return the slice and component shape """
    obs_start = obs_end = 0
    component_shape = None
    for k, v in segments.items():
        obs_start = obs_end
        obs_end = obs_start + np.prod(v)
        if k == component_name:
            component_shape = v # tuple
            break
    assert component_shape is not None, "No component ({}) is found in the given components {}".format(component_name, [segments.keys()])
    return slice(obs_start, obs_end), component_shape


