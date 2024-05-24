import numpy as np

import torch
import torch.nn as nn

from rsl_rl.modules.mlp import MlpModel
from rsl_rl.modules.conv2d import Conv2dHeadModel
from rsl_rl.utils.utils import get_obs_slice

class EncoderActorCriticMixin:
    """ A general implementation where a seperate encoder is used to embed the obs/privileged_obs """
    def __init__(self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            obs_segments= None,
            privileged_obs_segments= None,
            encoder_component_names= [], # allow multiple encoders
            encoder_class_name= "MlpModel", # accept list of names (in the same order as encoder_component_names)
            encoder_kwargs= dict(), # accept list of kwargs (in the same order as encoder_component_names),
            encoder_output_size= None,
            critic_encoder_component_names= None, # None, "shared", or a list of names (in the same order as encoder_component_names)
            critic_encoder_class_name= None, # accept list of names (in the same order as encoder_component_names)
            critic_encoder_kwargs= None, # accept list of kwargs (in the same order as encoder_component_names),
            **kwargs,
        ):
        """ NOTE: recurrent encoder is not implemented and tested yet.
        """
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.obs_segments = obs_segments
        self.privileged_obs_segments = privileged_obs_segments
        self.encoder_component_names = encoder_component_names
        self.encoder_class_name = encoder_class_name
        self.encoder_kwargs = encoder_kwargs
        self.encoder_output_size = encoder_output_size
        self.critic_encoder_component_names = critic_encoder_component_names
        self.critic_encoder_class_name = critic_encoder_class_name if not critic_encoder_class_name is None else encoder_class_name
        self.critic_encoder_kwargs = critic_encoder_kwargs if not critic_encoder_kwargs is None else encoder_kwargs
        self.obs_segments = obs_segments
        self.privileged_obs_segments = privileged_obs_segments

        self.prepare_obs_slices()

        super().__init__(
            num_actor_obs - sum([s[0].stop - s[0].start for s in self.encoder_obs_slices]) + len(self.encoder_obs_slices) * self.encoder_output_size,
            num_critic_obs if self.critic_encoder_component_names is None else num_critic_obs - sum([s[0].stop - s[0].start for s in self.critic_encoder_obs_slices]) + len(self.critic_encoder_obs_slices) * self.encoder_output_size,
            num_actions,
            obs_segments= obs_segments,
            privileged_obs_segments= privileged_obs_segments,
            **kwargs,
        )

        self.encoders = self.build_encoders(
            self.encoder_component_names,
            self.encoder_class_name,
            self.encoder_obs_slices,
            self.encoder_kwargs,
            self.encoder_output_size,
        )
        if not (self.critic_encoder_component_names is None or self.critic_encoder_component_names == "shared"):
            self.critic_encoders = self.build_encoders(
                self.critic_encoder_component_names,
                self.critic_encoder_class_name,
                self.critic_encoder_obs_slices,
                self.critic_encoder_kwargs,
                self.encoder_output_size,
            )

    def prepare_obs_slices(self):
        # NOTE: encoders are stored in the order of obs_component_names respectively.
        #   latents_order stores the order of how each output latent should be concatenated with
        #   the rest of the obs vector.
        self.encoder_obs_slices = [get_obs_slice(self.obs_segments, name) for name in self.encoder_component_names]
        self.latents_order = [i for i in range(len(self.encoder_obs_slices))]
        self.latents_order.sort(key= lambda i: self.encoder_obs_slices[i][0].start)
        if self.critic_encoder_component_names is not None:
            if self.critic_encoder_component_names == "shared":
                self.critic_encoder_obs_slices = self.encoder_obs_slices
            else:
                critic_obs_segments = self.obs_segments if self.privileged_obs_segments is None else self.privileged_obs_segments
                self.critic_encoder_obs_slices = [get_obs_slice(critic_obs_segments, name) for name in self.critic_encoder_component_names]
                self.critic_latents_order = [i for i in range(len(self.critic_encoder_obs_slices))]
                self.critic_latents_order.sort(key= lambda i: self.critic_encoder_obs_slices[i][0].start)

    def build_encoders(self, component_names, class_name, obs_slices, kwargs, encoder_output_size):
        encoders = nn.ModuleList()
        for component_i, name in enumerate(component_names):
            model_class_name = class_name[component_i] if isinstance(class_name, (tuple, list)) else class_name
            obs_slice = obs_slices[component_i]
            model_kwargs = kwargs[component_i] if isinstance(kwargs, (tuple, list)) else kwargs
            model_kwargs = model_kwargs.copy() # 1-level shallow copy
            # This code is not clean enough, need to sort out later
            if model_class_name == "MlpModel":
                hidden_sizes = model_kwargs.pop("hidden_sizes") + [encoder_output_size,]
                encoders.append(MlpModel(
                    np.prod(obs_slice[1]),
                    hidden_sizes= hidden_sizes,
                    output_size= None,
                    **model_kwargs,
                ))
            elif model_class_name == "Conv2dHeadModel":
                hidden_sizes = model_kwargs.pop("hidden_sizes") + [encoder_output_size,]
                encoders.append(Conv2dHeadModel(
                    obs_slice[1],
                    hidden_sizes= hidden_sizes,
                    output_size= None,
                    **model_kwargs,
                ))
            else:
                raise NotImplementedError(f"Encoder for {model_class_name} on {name} not implemented")
        return encoders
    
    def embed_encoders_latent(self, observations, obs_slices, encoders, latents_order):
        leading_dims = observations.shape[:-1]
        latents = []
        for encoder_i, encoder in enumerate(encoders):
            # This code is not clean enough, need to sort out later
            if isinstance(encoder, MlpModel):
                latents.append(encoder(
                    observations[..., obs_slices[encoder_i][0]].reshape(-1, np.prod(obs_slices[encoder_i][1]))
                ).reshape(*leading_dims, -1))
            elif isinstance(encoder, Conv2dHeadModel):
                latents.append(encoder(
                    observations[..., obs_slices[encoder_i][0]].reshape(-1, *obs_slices[encoder_i][1])
                ).reshape(*leading_dims, -1))
            else:
                raise NotImplementedError(f"Encoder for {type(encoder)} not implemented")
        # replace the obs vector with the latent vector in eace obs_slice[0] (the slice of obs)
        embedded_obs = []
        embedded_obs.append(observations[..., :obs_slices[latents_order[0]][0].start])
        for order_i in range(len(latents)- 1):
            current_idx = latents_order[order_i]
            next_idx = latents_order[order_i + 1]
            embedded_obs.append(latents[current_idx])
            embedded_obs.append(observations[..., obs_slices[current_idx][0].stop: obs_slices[next_idx][0].start])
        current_idx = latents_order[-1]
        next_idx = None
        embedded_obs.append(latents[current_idx])
        embedded_obs.append(observations[..., obs_slices[current_idx][0].stop:])
        
        return torch.cat(embedded_obs, dim= -1)
    
    def get_encoder_latent(self, observations, obs_component, critic= False):
        """ Get the latent vector from the encoder of the specified obs_component
        """
        leading_dims = observations.shape[:-1]
        encoder_obs_components = self.critic_encoder_component_names if critic else self.encoder_component_names
        encoder_obs_slices = self.critic_encoder_obs_slices if critic else self.encoder_obs_slices
        encoders = self.critic_encoders if critic else self.encoders

        for i, name in enumerate(encoder_obs_components):
            if name == obs_component:
                obs_component_var = observations[..., encoder_obs_slices[i][0]]
                if isinstance(encoders[i], MlpModel):
                    obs_component_var = obs_component_var.reshape(-1, np.prod(encoder_obs_slices[i][1]))
                elif isinstance(encoders[i], Conv2dHeadModel):
                    obs_component_var = obs_component_var.reshape(-1, *encoder_obs_slices[i][1])
                latent = encoders[i](obs_component_var).reshape(*leading_dims, -1)
                return latent
        raise ValueError(f"obs_component {obs_component} not found in encoder_obs_components")
    
    def act(self, observations, **kwargs):
        obs = self.embed_encoders_latent(observations, self.encoder_obs_slices, self.encoders, self.latents_order)
        return super().act(obs, **kwargs)
    
    def act_inference(self, observations):
        obs = self.embed_encoders_latent(observations, self.encoder_obs_slices, self.encoders, self.latents_order)
        return super().act_inference(obs)
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        if self.critic_encoder_component_names == "shared":
            obs = self.embed_encoders_latent(critic_observations, self.encoder_obs_slices, self.encoders, self.latents_order)
        elif self.critic_encoder_component_names is None:
            obs = critic_observations
        else:
            obs = self.embed_encoders_latent(critic_observations, self.critic_encoder_obs_slices, self.critic_encoders, self.critic_latents_order)
        return super().evaluate(obs, masks, hidden_states)
    
from .actor_critic import ActorCritic
class EncoderActorCritic(EncoderActorCriticMixin, ActorCritic):
    pass

from .actor_critic_recurrent import ActorCriticRecurrent
class EncoderActorCriticRecurrent(EncoderActorCriticMixin, ActorCriticRecurrent):
    pass
    