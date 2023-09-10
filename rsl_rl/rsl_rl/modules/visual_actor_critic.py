import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from rsl_rl.modules.deterministic_policy import DeterministicPolicyMixin
from rsl_rl.modules.conv2d import Conv2dHeadModel
from rsl_rl.utils.utils import get_obs_slice

class VisualActorCriticMixin:
    def __init__(self,
            num_actor_obs,
            num_critic_obs,
            num_actions,
            obs_segments= None,
            privileged_obs_segments= None,
            visual_component_name= "forward_depth",
            visual_kwargs= dict(),
            visual_latent_size= 256,
            **kwargs,
        ):
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.obs_segments = obs_segments
        self.visual_latent_size = visual_latent_size
        self.visual_obs_slice = get_obs_slice(obs_segments, visual_component_name)
        self.visual_kwargs = dict(
            channels= [64, 64],
            kernel_sizes= [3, 3],
            strides= [1, 1],
            hidden_sizes= [256],
        ); self.visual_kwargs.update(visual_kwargs)
        if not privileged_obs_segments is None: # use the same visual encoder
            print("VisualActorCriticMixin.__init__ Warning: privileged_obs_segment is not processed, please make sure it is what you want")
            pass

        super().__init__(
            num_actor_obs - (self.visual_obs_slice[0].stop - self.visual_obs_slice[0].start) + self.visual_latent_size,
            num_critic_obs,
            num_actions,
            **kwargs,
        )

        self.visual_encoder = Conv2dHeadModel(
            image_shape= self.visual_obs_slice[1],
            output_size= self.visual_latent_size,
            **self.visual_kwargs,
        )

    def embed_visual_latent(self, observations):
        leading_dims = observations.shape[:-1]
        visual_latent = self.visual_encoder(
            observations[..., self.visual_obs_slice[0]].reshape(-1, *self.visual_obs_slice[1])
        ).reshape(*leading_dims, -1)
        obs = torch.cat([
            observations[..., :self.visual_obs_slice[0].start],
            visual_latent,
            observations[..., self.visual_obs_slice[0].stop:],
        ], dim= -1)
        return obs

    def act(self, observations, **kwargs):
        obs = self.embed_visual_latent(observations)
        return super().act(obs, **kwargs)

    def act_inference(self, observations):
        obs = self.embed_visual_latent(observations)
        return super().act_inference(obs)
    
    def act_with_embedded_latent(self, observations, **kwargs):
        """ Using this method to run the actor network that is already embedded with visual latent.
        """
        return super().act(observations, **kwargs)

class VisualDeterministicRecurrent(DeterministicPolicyMixin, VisualActorCriticMixin, ActorCriticRecurrent):
    pass

class VisualDeterministicAC(DeterministicPolicyMixin, VisualActorCriticMixin, ActorCritic):
    pass
