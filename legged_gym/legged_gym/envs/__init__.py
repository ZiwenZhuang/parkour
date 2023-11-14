# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO, A1PlaneCfg, A1RoughCfgTPPO
from .base.legged_robot import LeggedRobot
from .base.legged_robot_field import LeggedRobotField
from .base.legged_robot_noisy import LeggedRobotNoisy
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from .a1.a1_field_distill_config import A1FieldDistillCfg, A1FieldDistillCfgPPO
from .go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from .go1.go1_field_distill_config import Go1FieldDistillCfg, Go1FieldDistillCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1_teacher", LeggedRobot, A1PlaneCfg(), A1RoughCfgTPPO() )
task_registry.register( "a1_field", LeggedRobotNoisy, A1FieldCfg(), A1FieldCfgPPO() )
task_registry.register( "a1_distill", LeggedRobotNoisy, A1FieldDistillCfg(), A1FieldDistillCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "go1_field", LeggedRobotNoisy, Go1FieldCfg(), Go1FieldCfgPPO())
task_registry.register( "go1_distill", LeggedRobotNoisy, Go1FieldDistillCfg(), Go1FieldDistillCfgPPO())

## The following tasks are for the convinience of opensource
from .a1.a1_remote_config import A1RemoteCfg, A1RemoteCfgPPO
task_registry.register( "a1_remote", LeggedRobotNoisy, A1RemoteCfg(), A1RemoteCfgPPO() )
from .a1.a1_jump_config import A1JumpCfg, A1JumpCfgPPO
task_registry.register( "a1_jump", LeggedRobotNoisy, A1JumpCfg(), A1JumpCfgPPO() )
from .a1.a1_down_config import A1DownCfg, A1DownCfgPPO
task_registry.register( "a1_down", LeggedRobotNoisy, A1DownCfg(), A1DownCfgPPO() )
from .a1.a1_leap_config import A1LeapCfg, A1LeapCfgPPO
task_registry.register( "a1_leap", LeggedRobotNoisy, A1LeapCfg(), A1LeapCfgPPO() )
from .a1.a1_crawl_config import A1CrawlCfg, A1CrawlCfgPPO
task_registry.register( "a1_crawl", LeggedRobotNoisy, A1CrawlCfg(), A1CrawlCfgPPO() )
from .a1.a1_tilt_config import A1TiltCfg, A1TiltCfgPPO
task_registry.register( "a1_tilt", LeggedRobotNoisy, A1TiltCfg(), A1TiltCfgPPO() )
from .go1.go1_remote_config import Go1RemoteCfg, Go1RemoteCfgPPO
task_registry.register( "go1_remote", LeggedRobotNoisy, Go1RemoteCfg(), Go1RemoteCfgPPO() )
from .go1.go1_jump_config import Go1JumpCfg, Go1JumpCfgPPO
task_registry.register( "go1_jump", LeggedRobotNoisy, Go1JumpCfg(), Go1JumpCfgPPO() )
from .go1.go1_down_config import Go1DownCfg, Go1DownCfgPPO
task_registry.register( "go1_down", LeggedRobotNoisy, Go1DownCfg(), Go1DownCfgPPO() )
from .go1.go1_leap_config import Go1LeapCfg, Go1LeapCfgPPO
task_registry.register( "go1_leap", LeggedRobotNoisy, Go1LeapCfg(), Go1LeapCfgPPO() )
from .go1.go1_crawl_config import Go1CrawlCfg, Go1CrawlCfgPPO
task_registry.register( "go1_crawl", LeggedRobotNoisy, Go1CrawlCfg(), Go1CrawlCfgPPO() )
from .go1.go1_tilt_config import Go1TiltCfg, Go1TiltCfgPPO
task_registry.register( "go1_tilt", LeggedRobotNoisy, Go1TiltCfg(), Go1TiltCfgPPO() )
