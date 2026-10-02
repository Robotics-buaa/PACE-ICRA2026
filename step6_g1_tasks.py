"""Register G1 tasks, check runner interfaces, and optionally preview the GUI."""

import argparse
import ast
import copy
import faulthandler
import hashlib
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT = Path("/workspace/projects/PACE-ICRA2026")
START = "# BEGIN PACE G1 TASKS (step 6)"
END = "# END PACE G1 TASKS (step 6)"
REGISTRATION = '''# BEGIN PACE G1 TASKS (step 6)
from legged_lab.envs.g1_tt.g1_tt_env import G1TTEnv
from legged_lab.envs.g1_tt.g1_tt_config import (
    G1TableTennisEnvCfg,
    G1TT_EvalEnvCfg,
    G1TableTennisAgentCfg,
)

task_registry.register("g1_tt", G1TTEnv, G1TableTennisEnvCfg(), G1TableTennisAgentCfg())
task_registry.register("g1_tt_eval", G1TTEnv, G1TT_EvalEnvCfg(), G1TableTennisAgentCfg())
# END PACE G1 TASKS (step 6)
'''


def registration_names(source):
    names = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "task_registry" and node.func.attr == "register"
                and node.args and isinstance(node.args[0], ast.Constant)):
            names.append(node.args[0].value)
    return names


def install(project):
    entry = project / "legged_lab/envs/__init__.py"
    protected = [project / rel for rel in (
        "legged_lab/envs/base/tt_env.py",
        "legged_lab/envs/t1_tt/t1_tt_config.py",
        "legged_lab/envs/g1_tt/g1_tt_env.py",
        "legged_lab/envs/g1_tt/g1_tt_config.py",
        "legged_lab/physics/aerodynamics.py",
        "legged_lab/mdp/rewards.py",
    )]
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    g1_source = (project / "legged_lab/envs/g1_tt/g1_tt_env.py").read_text()
    assert "def _initialize_g1_geometry(self):" in g1_source, "Complete step 5 first"
    source = entry.read_text()
    before_names = registration_names(source)
    if START in source or END in source:
        assert source.count(START) == source.count(END) == 1, "Unexpected step 6 block markers"
        start = source.index(START)
        end = source.index(END) + len(END)
        suffix = source[end:]
        if suffix.startswith("\n"):
            suffix = suffix[1:]
        updated = source[:start] + REGISTRATION + suffix
    else:
        assert not {"g1_tt", "g1_tt_eval"}.intersection(before_names), (
            "Existing G1 registrations require review before editing", before_names
        )
        updated = source + ("" if source.endswith("\n") else "\n") + "\n" + REGISTRATION
    names = registration_names(updated)
    assert names.count("g1_tt") == names.count("g1_tt_eval") == 1
    assert [n for n in names if n not in ("g1_tt", "g1_tt_eval")] == [
        n for n in before_names if n not in ("g1_tt", "g1_tt_eval")
    ]
    compile(updated, str(entry), "exec")
    if updated != source:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = entry.with_name(entry.name + ".step6_" + stamp + ".bak")
        shutil.copy2(entry, backup)
        entry.write_text(updated)
        print("Backup:", backup, flush=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest
               for p, digest in digests.items())
    print("[PASS] g1_tt and g1_tt_eval registered; existing task registrations preserved", flush=True)


def check(project, gui):
    sys.path.insert(0, str(project))
    faulthandler.enable()
    faulthandler.dump_traceback_later(180 if gui else 60, repeat=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=not gui).app
    env = None
    try:
        import torch
        import legged_lab.envs
        from legged_lab.utils import task_registry
        from legged_lab.envs.base.tt_env import TTEnv
        from legged_lab.envs.g1_tt.g1_tt_env import G1TTEnv
        from legged_lab.envs.g1_tt.g1_tt_config import (
            G1TableTennisEnvCfg, G1TT_EvalEnvCfg, G1TableTennisAgentCfg,
        )
        from rsl_rl.runners import OnPolicyPredictorRegressionRunner

        for name, cfg_type in (("g1_tt", G1TableTennisEnvCfg), ("g1_tt_eval", G1TT_EvalEnvCfg)):
            registered_cfg, registered_agent = task_registry.get_cfgs(name)
            assert task_registry.get_task_class(name) is G1TTEnv
            assert type(registered_cfg) is cfg_type
            assert isinstance(registered_agent, G1TableTennisAgentCfg)
            assert registered_agent.experiment_name == "g1_table_tennis"
            assert registered_agent.resume is False
            assert registered_agent.predictor["history_len"] == 5
            assert list(registered_agent.predictor["hidden_sizes"]) == [64, 64]
        assert task_registry.get_task_class("t1_tt") is TTEnv
        assert task_registry.get_task_class("t1_tt_eval") is TTEnv
        t1_cfg, t1_agent = task_registry.get_cfgs("t1_tt")
        assert t1_agent.experiment_name == "t1_table_tennis"
        train_cfg, _ = task_registry.get_cfgs("g1_tt")
        eval_cfg, _ = task_registry.get_cfgs("g1_tt_eval")
        assert eval_cfg.scene.max_episode_length_s > train_cfg.scene.max_episode_length_s
        assert t1_cfg.scene.robot.spawn.usd_path != train_cfg.scene.robot.spawn.usd_path
        print("[PASS] Registered G1 train/eval configs, separate log name, T1 tasks intact", flush=True)

        # Only modify copies of the registered configurations for this check.
        task_name = "g1_tt_eval" if gui else "g1_tt"
        registered_cfg, registered_agent = task_registry.get_cfgs(task_name)
        cfg, agent = copy.deepcopy(registered_cfg), copy.deepcopy(registered_agent)
        cfg.scene.num_envs = 1
        cfg.scene.seed = agent.seed
        cfg.noise.add_noise = False
        cfg.domain_rand.action_delay.enable = False
        cfg.domain_rand.perception_delay.enable = False
        cfg.domain_rand.events.push_robot = None
        cfg.domain_rand.events.add_base_mass = None
        cfg.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        cfg.domain_rand.events.reset_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (1.0, 1.0)
        cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (0.0, 0.0)
        env_class = task_registry.get_task_class(task_name)
        env = env_class(cfg, headless=not gui)
        print("[PASS] Environment created through task registry:", task_name, flush=True)

        # Construct and run forward passes only. No checkpoint loading or optimization.
        runner = OnPolicyPredictorRegressionRunner(
            env, copy.deepcopy(agent.to_dict()), log_dir=None, device=agent.device,
        )
        obs, extras = env.get_observations()
        critic_obs = extras["observations"]["critic"]
        assert obs.shape == (1, 435) and critic_obs.shape == (1, 520)
        assert runner._pred_input_dim == 15
        with torch.no_grad():
            actions = runner.alg.policy.act_inference(obs)
            values = runner.alg.policy.evaluate(critic_obs)
            prediction = runner._predictor(torch.zeros((1, 15), device=runner.device))
        assert actions.shape == (1, 23) and values.shape == (1, 1) and prediction.shape == (1, 3)
        assert all(bool(torch.isfinite(x).all()) for x in (actions, values, prediction))
        assert runner.current_learning_iteration == 0 and runner.tot_timesteps == 0
        print("[PASS] Predictor runner forward interfaces: actor 435->23, critic 520->1, predictor 15->3", flush=True)

        # Default joint targets test physical stepping independently of random network outputs.
        zero_actions = torch.zeros((1, 23), device=env.device)
        for _ in range(24):
            obs, reward, done, extras = env.step(zero_actions)
            runner._record_ball_positions()
            assert bool(torch.isfinite(obs).all()) and bool(torch.isfinite(reward).all())
            assert bool(torch.isfinite(extras["observations"]["critic"]).all())
        assert runner._traj_len >= runner.pred_history_len
        assert runner.current_learning_iteration == 0 and runner.tot_timesteps == 0
        print("[PASS] Registered task: 24 control steps and predictor trajectory recording; no optimization", flush=True)
        print("STEP 6 TASK CHECK PASSED", flush=True)

        if gui:
            env.reset(torch.arange(1, device=env.device))
            env.sim.set_camera_view(eye=(-4.0, -3.0, 2.0), target=(-0.2, 0.0, 0.8))
            env.sim.render()
            print("G1 GUI PREVIEW READY: default pose, table and paddle; physics paused for inspection", flush=True)
            print("Close the GUI window or press Ctrl+C to finish", flush=True)
            faulthandler.cancel_dump_traceback_later()
            while app.is_running():
                env.sim.render()
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("G1 preview stopped", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open the existing direct GUI after task checks")
    parser.add_argument("--install-only", action="store_true")
    args = parser.parse_args()
    install(PROJECT)
    if not args.install_only:
        check(PROJECT, args.gui)
