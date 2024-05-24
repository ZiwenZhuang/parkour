""" PPO with teacher network """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rsl_rl.modules as modules
from rsl_rl.utils import unpad_trajectories
from rsl_rl.storage.rollout_storage import ActionLabelRollout
from rsl_rl.algorithms.ppo import PPO
from legged_gym import LEGGED_GYM_ROOT_DIR

# assuming learning iteration is at an assumable iteration scale
def GET_PROB_FUNC(option, iteration_scale):
    PROB_options = {
        "linear": (lambda x: max(0, 1 - 1 / iteration_scale * x)),
        "exp": (lambda x: max(0, (1 - 1 / iteration_scale) ** x)),
        "tanh": (lambda x: max(0, 0.5 * (1 - torch.tanh(1 / iteration_scale * (x - iteration_scale))))),
    }
    return PROB_options[option]

class TPPO(PPO):
    def __init__(self,
            *args,
            teacher_ac_path= None, # running device will be handled
            teacher_policy_class_name= "ActorCritic",
            teacher_policy= dict(),
            label_action_with_critic_obs= True, # else, use actor obs
            teacher_act_prob= "exp", # a number or a callable to (0 ~ 1) to the selection of act using teacher policy
            update_times_scale= 100, # a rough estimation of how many times the update will be called
            using_ppo= True, # If False, compute_losses will skip ppo loss computation and returns to DAGGR
            distillation_loss_coef= 1., # can also be string to select a prob function to scale the distillation loss
            distill_target= "real",
            distill_latent_coef= 1.,
            distill_latent_target= "real",
            distill_latent_obs_component_mapping= None,
            buffer_dilation_ratio= 1.,
            lr_scheduler_class_name= None,
            lr_scheduler= dict(),
            hidden_state_resample_prob= 0.0, # if > 0, Some hidden state in the minibatch will be resampled
            action_labels_from_sample= False, # if True, the action labels from teacher policy will be from policy.act instead of policy.act_inference
            **kwargs,
        ):
        """
        Args:
        - distill_latent_obs_component_mapping: a dict of
            {student_obs_component_name: teacher_obs_component_name}
            only when both policy are the instance of EncoderActorCriticMixin
        """
        super().__init__(*args, **kwargs)
        self.label_action_with_critic_obs = label_action_with_critic_obs
        self.teacher_act_prob = teacher_act_prob
        self.update_times_scale = update_times_scale
        if isinstance(self.teacher_act_prob, str):
            self.teacher_act_prob = GET_PROB_FUNC(self.teacher_act_prob, update_times_scale)
        else:
            self.__teacher_act_prob = self.teacher_act_prob
            self.teacher_act_prob = lambda x: self.__teacher_act_prob
        self.using_ppo = using_ppo
        self.__distillation_loss_coef = distillation_loss_coef
        if isinstance(self.__distillation_loss_coef, str):
            self.distillation_loss_coef_func = GET_PROB_FUNC(self.__distillation_loss_coef, update_times_scale)
        self.distill_target = distill_target
        self.distill_latent_coef = distill_latent_coef
        self.distill_latent_target = distill_latent_target
        self.distill_latent_obs_component_mapping = distill_latent_obs_component_mapping
        self.buffer_dilation_ratio = buffer_dilation_ratio
        self.lr_scheduler_class_name = lr_scheduler_class_name
        self.lr_scheduler_kwargs = lr_scheduler
        self.hidden_state_resample_prob = hidden_state_resample_prob
        self.action_labels_from_sample = action_labels_from_sample
        self.transition = ActionLabelRollout.Transition()

        # build and load teacher network
        teacher_actor_critic = getattr(modules, teacher_policy_class_name)(**teacher_policy)
        if not teacher_ac_path is None:
            if "{LEGGED_GYM_ROOT_DIR}" in teacher_ac_path:
                teacher_ac_path = teacher_ac_path.format(LEGGED_GYM_ROOT_DIR= LEGGED_GYM_ROOT_DIR)
            state_dict = torch.load(teacher_ac_path, map_location= "cpu")
            teacher_actor_critic_state_dict = state_dict["model_state_dict"]
            teacher_actor_critic.load_state_dict(teacher_actor_critic_state_dict)
        else:
            print("TPPO Warning: No snapshot loaded for teacher policy. Make sure you have a pretrained teacher network")
        teacher_actor_critic.to(self.device)
        self.teacher_actor_critic = teacher_actor_critic
        self.teacher_actor_critic.eval()

        # initialize lr scheduler if needed
        if not self.lr_scheduler_class_name is None:
            self.lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_class_name)(
                self.optimizer,
                **self.lr_scheduler_kwargs,
            )

    def init_storage(self, *args, **kwargs):
        self.storage = ActionLabelRollout(
            *args,
            **kwargs,
            buffer_dilation_ratio= self.buffer_dilation_ratio,
            device= self.device,
        )

    def act(self, obs, critic_obs):
        # get actions
        return_ = super().act(obs, critic_obs)
        if self.label_action_with_critic_obs and self.action_labels_from_sample:
            self.transition.action_labels = self.teacher_actor_critic.act(critic_obs).detach()
        elif self.label_action_with_critic_obs:
            self.transition.action_labels = self.teacher_actor_critic.act_inference(critic_obs).detach()
        elif self.action_labels_from_sample:
            self.transition.action_labels = self.teacher_actor_critic.act(obs).detach()
        else:
            self.transition.action_labels = self.teacher_actor_critic.act_inference(obs).detach()

        # decide whose action to use
        if not hasattr(self, "use_teacher_act_mask"):
            self.use_teacher_act_mask = torch.ones(obs.shape[0], device= self.device, dtype= torch.bool)
        return_[self.use_teacher_act_mask] = self.transition.action_labels[self.use_teacher_act_mask]

        return return_
    
    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        return_ = super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        self.teacher_actor_critic.reset(dones)
        # resample teacher action mask for those dones env
        self.use_teacher_act_mask[dones] = torch.rand(dones.sum(), device= self.device) < self.teacher_act_prob(self.current_learning_iteration)
        return return_

    def collect_transition_from_dataset(self, transition, infos):
        """ The interface to collect transition from dataset rather than env """
        super().act(transition.observation, transition.privileged_observation)
        self.transition.action_labels = transition.action
        super().process_env_step(transition.reward, transition.done, infos, transition.next_observation, transition.next_privileged_observation)

    def compute_returns(self, last_critic_obs):
        if not self.using_ppo:
            return
        return super().compute_returns(last_critic_obs)
    
    def update(self, *args, **kwargs):
        return_ = super().update(*args, **kwargs)
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return return_
    
    def compute_losses(self, minibatch):
        if self.hidden_state_resample_prob > 0.0:
            # assuming the hidden states are from LSTM or GRU, which are always betwein -1 and 1
            hidden_state_example = minibatch.hidden_states[0][0] if isinstance(minibatch.hidden_states[0], tuple) else minibatch.hidden_states[0]
            resample_mask = torch.rand(hidden_state_example.shape[1], device= self.device) < self.hidden_state_resample_prob
            # for each hidden state, resample from -1 to 1
            if isinstance(minibatch.hidden_states[0], tuple):
                # for LSTM not tested
                # iterate through actor and critic hidden state
                minibatch = minibatch._replace(hidden_states= tuple(
                    tuple(
                        torch.where(
                            resample_mask.unsqueeze(-1).unsqueeze(-1),
                            torch.rand_like(minibatch.hidden_states[i][j], device= self.device) * 2 - 1,
                            minibatch.hidden_states[i][j],
                        ) for j in range(len(minibatch.hidden_states[i]))
                    ) for i in range(len(minibatch.hidden_states))
                ))
            else:
                # for GRU
                # iterate through actor and critic hidden state
                minibatch = minibatch._replace(hidden_states= tuple(
                    torch.where(
                        resample_mask.unsqueeze(-1),
                        torch.rand_like(minibatch.hidden_states[i], device= self.device) * 2 - 1,
                        minibatch.hidden_states[i],
                    ) for i in range(len(minibatch.hidden_states))
                ))

        if self.using_ppo:
            losses, inter_vars, stats = super().compute_losses(minibatch)
        else:
            losses = dict()
            inter_vars = dict()
            stats = dict()
            self.actor_critic.act(minibatch.obs, masks=minibatch.masks, hidden_states=minibatch.hidden_states.actor)

        # distillation loss (with teacher actor)
        if self.distill_target == "real":
            dist_loss = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim= -1
            )
        elif self.distill_target == "mse_sum":
            dist_loss = F.mse_loss(
                self.actor_critic.action_mean,
                minibatch.action_labels,
                reduction= "none",
            ).sum(-1)
        elif self.distill_target == "l1":
            dist_loss = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim= -1,
                p= 1,
            )
        elif self.distill_target == "tanh":
            # for tanh, similar to loss function for sigmoid, refer to https://stats.stackexchange.com/questions/12754/matching-loss-function-for-tanh-units-in-a-neural-net
            dist_loss = F.binary_cross_entropy(
                (self.actor_critic.action_mean + 1) * 0.5,
                (minibatch.action_labels + 1) * 0.5,
            ) * 2
        elif self.distill_target == "scaled_tanh":
            l1 = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim= -1,
                p= 1,
            )
            dist_loss = F.binary_cross_entropy(
                (self.actor_critic.action_mean + 1) * 0.5,
                (minibatch.action_labels + 1) * 0.5, # (n, t, d)
                reduction= "none",
            ).mean(-1) * 2 * l1 / self.actor_critic.action_mean.shape[-1] # (n, t)
        elif self.distill_target == "max_log_prob":
            action_labels_log_prob = self.actor_critic.get_actions_log_prob(minibatch.action_labels)
            dist_loss = -action_labels_log_prob
        elif self.distill_target == "kl":
            raise NotImplementedError()

        if "tanh" in self.distill_target:
            stats["l1distance"] = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim= -1,
                p= 1,
            ).mean().detach()
            stats["l1_before_tanh"] = torch.norm(
                torch.tan(self.actor_critic.action_mean) - torch.tan(minibatch.action_labels),
                dim= -1,
                p= 1
            ).mean().detach()
        # update distillation loss coef if applicable
        self.distillation_loss_coef = self.distillation_loss_coef_func(self.current_learning_iteration) if hasattr(self, "distillation_loss_coef_func") else self.__distillation_loss_coef
        losses["distillation_loss"] = dist_loss.mean()

        # distill latent embedding
        if self.distill_latent_obs_component_mapping is not None:
            for k, v in self.distill_latent_obs_component_mapping.items():
                # get the latent embedding
                latent = self.actor_critic.get_encoder_latent(
                    minibatch.obs,
                    k,
                )
                with torch.no_grad():
                    target_latent = self.teacher_actor_critic.get_encoder_latent(
                        minibatch.critic_obs,
                        v,
                    )
                if self.actor_critic.is_recurrent:
                    latent = unpad_trajectories(latent, minibatch.masks)
                    target_latent = unpad_trajectories(target_latent, minibatch.masks)
                if self.distill_latent_target == "real":
                    dist_loss = torch.norm(
                        latent - target_latent,
                        dim= -1,
                    )
                elif self.distill_latent_target == "l1":
                    dist_loss = torch.norm(
                        latent - target_latent,
                        dim= -1,
                        p= 1,
                    )
                elif self.distill_latent_target == "tanh":
                    dist_loss = F.binary_cross_entropy(
                        (latent + 1) * 0.5,
                        (target_latent + 1) * 0.5,
                    ) * 2
                elif self.distill_latent_target == "scaled_tanh":
                    l1 = torch.norm(
                        latent - target_latent,
                        dim= -1,
                        p= 1,
                    )
                    dist_loss = F.binary_cross_entropy(
                        (latent + 1) * 0.5,
                        (target_latent + 1) * 0.5, # (n, t, d)
                        reduction= "none",
                    ).mean(-1) * 2 * l1 / latent.shape[-1] # (n, t)
                setattr(self, f"distill_latent_{k}_coef", self.distill_latent_coef)
                losses[f"distill_latent_{k}"] = dist_loss.mean()

        return losses, inter_vars, stats
