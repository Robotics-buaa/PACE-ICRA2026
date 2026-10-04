"""Adapt G1 body targets to measured paddle geometry, then verify two environments."""

import argparse
import ast
import faulthandler
import hashlib
import json
import math
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT = Path("/workspace/projects/PACE-ICRA2026")
GEOMETRY_METHODS = r'''    def _initialize_g1_geometry(self):
        """Measure the default paddle offset before randomized episode resets."""
        ids, names = self.robot.find_bodies(G1_PADDLE_BODY_NAME)
        if names != [G1_PADDLE_BODY_NAME]:
            raise RuntimeError(f"Expected one G1 paddle body, found {names}")
        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos, self.robot.data.default_joint_vel
        )
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos)
        self.sim.forward()
        paddle_quat = self.robot.data.body_quat_w[:, ids[0]]
        offset = torch.tensor(
            G1_PADDLE_CENTER_OFFSET, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        center = self.robot.data.body_pos_w[:, ids[0]] + math_utils.quat_apply(
            paddle_quat, offset
        )
        measured = math_utils.quat_rotate_inverse(
            self.robot.data.root_link_quat_w,
            center - self.robot.data.root_link_pos_w,
        )
        if not bool(torch.isfinite(measured).all()):
            raise RuntimeError("Non-finite G1 reference paddle geometry")
        self._g1_measured_paddle_offset_b = measured.detach().clone()
        configured = self.cfg.reference_paddle_offset_b
        if configured is None:
            self._g1_reference_paddle_offset_b = measured.detach().clone()
        else:
            self._g1_reference_paddle_offset_b = torch.tensor(
                configured, device=self.device, dtype=torch.float32
            ).expand(self.num_envs, -1)

    def _g1_paddle_offset_xy(self):
        # Rotate the neutral-pose reference with the current heading.
        heading = self.robot.data.heading_w
        offset = self._g1_reference_paddle_offset_b
        c, s = torch.cos(heading), torch.sin(heading)
        return torch.stack(
            [c * offset[:, 0] - s * offset[:, 1],
             s * offset[:, 0] + c * offset[:, 1]], dim=-1,
        )

    def _g1_base_target(self, ball_target):
        """Convert an environment-frame ball target to a table-frame base target."""
        target = ball_target.clone()
        target[:, :2] -= self._g1_paddle_offset_xy()
        target[:, 2] = self.cfg.reference_base_height
        table_origin = self.table.data.root_link_pos_w - self.scene.env_origins
        return target - table_origin

    def _g1_relative_target(self, actor_obs):
        robot_pos = self.robot.data.root_link_pos_w - self.table.data.root_link_pos_w
        target = self._g1_base_target(self.ball_prediction)
        actor_obs[:, -3:-1] = (
            target[:, :2] - robot_pos[:, :2]
        ) * self.obs_scales.robot_pos
        return actor_obs

    def compute_current_observations(self):
        actor_obs, critic_obs = super().compute_current_observations()
        return self._g1_relative_target(actor_obs), critic_obs

    def compute_current_observations_perception(self):
        actor_obs, critic_obs = super().compute_current_observations_perception()
        return self._g1_relative_target(actor_obs), critic_obs

    def compute_intermediate_values(self):
        # Retain PACE ball dynamics and hit bookkeeping, then replace geometry targets.
        super().compute_intermediate_values()
        self.pos_pred_before_ro = self._g1_base_target(self.pos_pred_before)
        self.pos_pred_after_ro = self._g1_base_target(self.pos_pred_after)
        mask_invalid = self.mask_invalid.unsqueeze(-1)
        self.robot_future_pos = torch.where(
            self.mask_before.unsqueeze(-1),
            self.pos_pred_before_ro, self.pos_pred_after_ro,
        )
        ready = self.robot_future_pos.new_tensor([
            *self.cfg.ready_base_xy, self.cfg.reference_base_height,
        ]).expand_as(self.robot_future_pos)
        table_origin = self.table.data.root_link_pos_w - self.scene.env_origins
        self.robot_future_pos = torch.where(
            mask_invalid, ready - table_origin, self.robot_future_pos
        )
        self.vel_ro_before = torch.clamp(
            (self.pos_pred_before_ro - self.robot_pos) * 4.0, min=-7.0, max=7.0
        )
        self.vel_ro_after = torch.clamp(
            (self.pos_pred_after_ro - self.robot_pos) * 4.0, min=-7.0, max=7.0
        )
        self.robot_future_vel = torch.where(
            self.mask_before.unsqueeze(-1), self.vel_ro_before, self.vel_ro_after
        )
        self.robot_future_vel = torch.where(
            mask_invalid, torch.zeros_like(self.robot_future_vel), self.robot_future_vel
        )
        # Invalid predictions fall back to the actual G1 paddle center.
        self.ball_future_pose = torch.where(
            mask_invalid, self.paddle_pos, self.ball_future_pose
        )
        landing_vis = torch.stack([
            self.predict_x_land, self.predict_y_land,
            torch.full_like(self.predict_x_land, 0.78),
        ], dim=-1)
        self.ball_future_pose_vis = torch.where(
            self.touched_paddel_no_bounce_table.unsqueeze(-1),
            landing_vis, self.ball_future_pose,
        ) + self.scene.env_origins
'''
FIELDS = '''    # Measure the default G1 paddle offset unless explicitly overridden.
    reference_base_height: float = G1_TT_CFG.init_state.pos[2]
    reference_paddle_offset_b: tuple = None
    ready_base_xy: tuple = (-1.80, 0.30)

'''


def install(project):
    paths = [project / rel for rel in (
        "legged_lab/envs/base/tt_env.py",
        "legged_lab/envs/t1_tt/t1_tt_config.py",
        "legged_lab/physics/aerodynamics.py",
        "legged_lab/mdp/rewards.py",
    )]
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    env_path = project / "legged_lab/envs/g1_tt/g1_tt_env.py"
    cfg_path = env_path.with_name("g1_tt_config.py")
    env_source = env_path.read_text()
    assert "def init_obs_buffer(self):" in env_source, "Complete the step 4 noise fix first"
    if "def _initialize_g1_geometry(self):" not in env_source:
        anchor = "    def init_buffers(self):\n        super().init_buffers()\n"
        assert env_source.count(anchor) == 1, "Unexpected G1 init_buffers; stop before editing"
        env_source = env_source.replace(
            anchor, "    def init_buffers(self):\n"
            "        self._initialize_g1_geometry()\n        super().init_buffers()\n", 1,
        )
        env_source = env_source.rstrip() + "\n\n" + GEOMETRY_METHODS + "\n"
    cfg_source = cfg_path.read_text()
    if "reference_paddle_offset_b" not in cfg_source:
        tree = ast.parse(cfg_source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "G1TableTennisEnvCfg")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "__post_init__")
        lines = cfg_source.splitlines(keepends=True)
        lines[method.lineno - 1:method.lineno - 1] = [FIELDS]
        cfg_source = "".join(lines)
    updates = [(env_path, env_source), (cfg_path, cfg_source)]
    # Keep the completed step 4 check from overwriting the new G1 methods if rerun.
    old_check = project / "step4_g1_interface.py"
    if old_check.exists():
        source = old_check.read_text()
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "ENV_SOURCE"
                            for t in n.targets))
        assert "'''" not in env_source
        lines = source.splitlines(keepends=True)
        lines[node.lineno - 1:node.end_lineno] = ["ENV_SOURCE = r'''" + env_source + "'''\n"]
        updates.append((old_check, "".join(lines)))
    for path, source in updates:
        compile(source, str(path), "exec")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for path, source in updates:
        if path.read_text() == source:
            continue
        backup = path.with_name(path.name + ".step5_" + stamp + ".bak")
        shutil.copy2(path, backup)
        path.write_text(source)
        print("Backup:", backup, flush=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest
               for p, digest in digests.items())
    print("[PASS] G1 geometry targets installed; T1 and compatibility fixes unchanged", flush=True)


def check(project):
    sys.path.insert(0, str(project))
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch
        import isaaclab.utils.math as math_utils
        from legged_lab.envs.g1_tt.g1_tt_config import G1TableTennisEnvCfg, G1TT_EvalEnvCfg
        from legged_lab.envs.g1_tt.g1_tt_env import G1TTEnv

        cfg = G1TableTennisEnvCfg()
        eval_cfg = G1TT_EvalEnvCfg()
        assert cfg.reference_base_height == eval_cfg.reference_base_height == 0.74
        assert cfg.reference_paddle_offset_b is None
        cfg.scene.num_envs = 2
        cfg.scene.seed = 42
        cfg.noise.add_noise = True
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
        env = G1TTEnv(cfg, headless=True)
        reference = env._g1_reference_paddle_offset_b.clone()
        assert reference.shape == (2, 3) and bool(torch.isfinite(reference).all())
        assert torch.allclose(reference[0], reference[1], atol=1e-5)
        assert float(reference.norm(dim=-1).min()) > 0.05
        # Independently check the measured reference against the loaded default pose.
        actual = math_utils.quat_rotate_inverse(
            env.robot.data.root_link_quat_w,
            env.paddle_touch_point - env.robot.data.root_link_pos_w,
        )
        assert torch.allclose(actual, reference, atol=1e-5), (actual, reference)
        print("[PASS] Measured G1 default paddle offset (base frame):", reference[0].tolist(), flush=True)
        print("[PASS] Base target height = %.3f m; both environment origins calibrated consistently" %
              cfg.reference_base_height, flush=True)

        # Test heading rotation, both actor observation paths, and table-frame conversion.
        root = env.robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        yaw = torch.tensor([0.0, math.pi / 2], device=env.device)
        root[:, 3:7] = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw,
        )
        env.robot.write_root_pose_to_sim(root[:, :7])
        env.robot.write_root_velocity_to_sim(torch.zeros((2, 6), device=env.device))
        env.sim.forward()
        env.scene.update(env.physics_dt)
        env.compute_perception()
        env.compute_paddle_touch()
        prediction = torch.tensor([[-1.75, 0.2, 1.1], [-1.8, -0.1, 1.0]], device=env.device)
        env.ball_prediction = prediction.clone()
        yaw_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw,
        )
        world_offset = math_utils.quat_apply(yaw_quat, reference)
        robot_local = env.robot.data.root_link_pos_w - env.scene.env_origins
        expected_relative = (prediction[:, :2] - world_offset[:, :2]
                             - robot_local[:, :2]) * env.obs_scales.robot_pos
        for observe in (env.compute_current_observations, env.compute_current_observations_perception):
            actor, critic = observe()
            assert actor.shape == (2, 87) and critic.shape == (2, 104)
            assert torch.allclose(actor[:, -3:-1], expected_relative, atol=1e-5), actor[:, -3:-1]
        print("[PASS] Actor targets use measured G1 geometry at yaw 0 and 90 degrees", flush=True)

        def set_ball(local_positions, velocities):
            state = env.ball.data.default_root_state.clone()
            state[:, :3] = torch.tensor(local_positions, device=env.device) + env.scene.env_origins
            state[:, 7:10] = torch.tensor(velocities, device=env.device)
            state[:, 10:13] = 0.0
            env.ball.write_root_pose_to_sim(state[:, :7])
            env.ball.write_root_velocity_to_sim(state[:, 7:])
            env.has_touch_paddle.zero_()
            env.has_touch_paddle_rew.zero_()
            env.ball_contact_rew.zero_()
            env.sim.forward()
            env.scene.update(env.physics_dt)
            env.compute_perception()
            env.compute_paddle_touch()

        env.has_touch_own_table_prev[:] = torch.tensor([False, True], device=env.device)
        set_ball([[-1.7, 0.0, 0.95], [-1.7, 0.0, 0.95]], [[-2.0, 0.1, 1.0], [-2.0, 0.1, 1.0]])
        env.compute_intermediate_values()
        assert not bool(env.mask_invalid.any()), env.mask_invalid
        table_origin = env.table.data.root_link_pos_w - env.scene.env_origins
        expected_base = env.ball_future_pose.clone()
        expected_base[:, :2] -= world_offset[:, :2]
        expected_base[:, 2] = cfg.reference_base_height
        expected_base -= table_origin
        assert torch.allclose(env.robot_future_pos, expected_base, atol=1e-5)
        expected_vel = ((expected_base - env.robot_pos) * 4.0).clamp(-7.0, 7.0)
        assert torch.allclose(env.robot_future_vel, expected_vel, atol=1e-5)
        for name in ("reward_future_dis_ro", "reward_future_vel_base", "reward_future_dis_ee"):
            term = env.reward_manager.get_term_cfg(name)
            assert bool(torch.isfinite(term.func(env, **term.params)).all()), name
        print("[PASS] Pre/post-bounce body position and velocity rewards share G1 targets", flush=True)

        set_ball([[-3.0, 0.0, 0.6], [-3.0, 0.0, 0.6]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        env.compute_intermediate_values()
        assert bool(env.mask_invalid.all())
        assert torch.allclose(env.ball_future_pose, env.paddle_pos, atol=1e-5)
        assert bool((env.robot_future_vel == 0).all())
        ready = torch.tensor([*cfg.ready_base_xy, cfg.reference_base_height], device=env.device)
        assert torch.allclose(env.robot_future_pos, ready.unsqueeze(0) - table_origin, atol=1e-5)
        print("[PASS] Invalid ball targets fall back to the actual paddle and configured ready position", flush=True)

        env.reset(torch.arange(2, device=env.device))
        assert torch.allclose(env._g1_reference_paddle_offset_b, reference)
        for _ in range(10):
            obs, reward, done, extras = env.step(torch.zeros((2, 23), device=env.device))
            assert obs.shape == (2, 435) and extras["observations"]["critic"].shape == (2, 520)
            assert bool(torch.isfinite(obs).all()) and bool(torch.isfinite(reward).all())
            assert bool(torch.isfinite(extras["observations"]["critic"]).all())
        print("[PASS] Two environments, noise enabled: 10 control steps; actor=(2,435), critic=(2,520)", flush=True)
        report_path = project / "legged_lab/assets/unitree/G1_TT/g1_reference_geometry.json"
        report = {
            "reference_base_height": cfg.reference_base_height,
            "measured_paddle_offset_b": reference[0].tolist(),
            "ready_base_xy": list(cfg.ready_base_xy),
            "paddle_offset_source": "loaded G1 default joint pose, before randomized reset",
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("Reference report:", report_path, flush=True)
        print("STEP 5 GEOMETRY CHECK PASSED", flush=True)
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
    parser.add_argument("--install-only", action="store_true")
    args = parser.parse_args()
    install(PROJECT)
    if not args.install_only:
        check(PROJECT)
