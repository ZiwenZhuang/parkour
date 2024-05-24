""" A file put all mixin class combinations """
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .encoder_actor_critic import EncoderActorCriticMixin
from .state_estimator import EstimatorMixin

class EncoderStateAc(EstimatorMixin, EncoderActorCriticMixin, ActorCritic):
    pass

class EncoderStateAcRecurrent(EstimatorMixin, EncoderActorCriticMixin, ActorCriticRecurrent):
    
    def load_misaligned_state_dict(self, module, obs_segments, privileged_obs_segments=None):
        pass