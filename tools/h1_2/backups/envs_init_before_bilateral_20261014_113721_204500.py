# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.legged_env import LeggedEnv
from legged_lab.envs.base.tt_env import TTEnv

from legged_lab.envs.t1_tt.t1_tt_config import (
    T1TableTennisEnvCfg,
    T1TableTennisAgentCfg,
    T1TT_EvalEnvCfg,
)


from legged_lab.utils.task_registry import task_registry
task_registry.register("t1_tt", TTEnv, T1TableTennisEnvCfg(), T1TableTennisAgentCfg()) #TTEnv
task_registry.register("t1_tt_eval", TTEnv, T1TT_EvalEnvCfg(), T1TableTennisAgentCfg()) 

# BEGIN PACE G1 TASKS (step 6)
from legged_lab.envs.g1_tt.g1_tt_env import G1TTEnv
from legged_lab.envs.g1_tt.g1_tt_config import (
    G1TableTennisEnvCfg,
    G1TT_EvalEnvCfg,
    G1TableTennisAgentCfg,
)

task_registry.register("g1_tt", G1TTEnv, G1TableTennisEnvCfg(), G1TableTennisAgentCfg())
task_registry.register("g1_tt_eval", G1TTEnv, G1TT_EvalEnvCfg(), G1TableTennisAgentCfg())
# END PACE G1 TASKS (step 6)

# BEGIN PACE G1 WRIST TASKS
from legged_lab.envs.g1_tt_wrist.g1_tt_wrist_env import G1WristTTEnv
from legged_lab.envs.g1_tt_wrist.g1_tt_wrist_config import (
    G1WristTableTennisEnvCfg, G1WristTT_EvalEnvCfg, G1WristTableTennisAgentCfg,
)
task_registry.register("g1_tt_wrist", G1WristTTEnv, G1WristTableTennisEnvCfg(), G1WristTableTennisAgentCfg())
task_registry.register("g1_tt_wrist_eval", G1WristTTEnv, G1WristTT_EvalEnvCfg(), G1WristTableTennisAgentCfg())
# END PACE G1 WRIST TASKS

# BEGIN PACE H1-2 TASKS
from legged_lab.envs.h1_2_tt.h1_2_tt_env import H1_2TTEnv
from legged_lab.envs.h1_2_tt.h1_2_tt_config import (
    H1_2TableTennisEnvCfg, H1_2TT_EvalEnvCfg, H1_2TableTennisAgentCfg,
)
task_registry.register("h1_2_tt", H1_2TTEnv, H1_2TableTennisEnvCfg(), H1_2TableTennisAgentCfg())
task_registry.register("h1_2_tt_eval", H1_2TTEnv, H1_2TT_EvalEnvCfg(), H1_2TableTennisAgentCfg())
# END PACE H1-2 TASKS
