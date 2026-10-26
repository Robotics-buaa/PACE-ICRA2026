"""Compare neutral stance at two positions, with and without nearby serves.

Diagnostic copies only: no training, asset edits, gain changes or fixed base.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path("/workspace/projects/PACE-ICRA2026")
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=6.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 < args.seconds <= 10:
    parser.error("--seconds must be in (0, 10]")
app = AppLauncher(args).app
env = None

try:
    import torch
    import isaaclab.utils.math as math_utils
    import legged_lab.envs
    from legged_lab.utils.task_registry import task_registry

    manifest = json.loads((ROOT / "tools/h1_2/bilateral_install_report.json").read_text())
    protected = {**manifest["protected_sha256"], **manifest["new_files_sha256"]}
    def verify():
        for name, sha in protected.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, "SOURCE CHANGED: " + name
    verify()
    cfg, _ = task_registry.get_cfgs("h1_2_tt_bilateral")
    cfg = deepcopy(cfg)
    cfg.scene.num_envs = 4
    cfg.scene.seed = 42
    cfg.device = args.device
    cfg.noise.add_noise = False
    cfg.domain_rand.action_delay.enable = False
    cfg.domain_rand.perception_delay.enable = False
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.add_base_mass = None
    # Uniform material for the comparison; only this in-memory fixture changes.
    cfg.domain_rand.events.physics_material = None
    cfg.domain_rand.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ("x", "y", "yaw")}
    cfg.domain_rand.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (1.0, 1.0)
    cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (0.0, 0.0)
    for event in (cfg.domain_rand.events.reset_locomotion_joints, cfg.domain_rand.events.reset_manipulation_joints):
        if "velocity_range" in event.params:
            event.params["velocity_range"] = (0.0, 0.0)

    names = ["x=-1.60, serves", "x=-1.95, serves", "x=-1.60, ball far away", "x=-1.95, ball far away"]
    observed = [{"case": name, "first_tilt_over_0_25_s": None, "first_non_timeout_reset_s": None,
                 "first_episode_min_base_z": float("inf"), "first_episode_max_tilt": 0.0,
                 "pose_at_first_tilt": None} for name in names]
    lower_names = [f"{side}_{part}_joint" for side in ("left", "right") for part in ("hip_pitch", "knee", "ankle_pitch")]
    Parent = task_registry.get_task_class("h1_2_tt_bilateral")

    class StanceFixture(Parent):
        def reset_ball(self, env_ids):
            super().reset_ball(env_ids)
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            ids = ids[ids >= 2]
            if len(ids):
                state = self.ball.data.default_root_state[ids].clone()
                # All normal gravity/contact settings stay on. During <=10 s the
                # ball stays far above the scene, never touching robot or table.
                state[:, :3] = self.scene.env_origins[ids] + state.new_tensor([100.0, 100.0, 1000.0])
                state[:, 7:] = 0.0
                self.reset_ball_state_buf[ids] = state
                self.ball.write_root_pose_to_sim(state[:, :7], ids)
                self.ball.write_root_velocity_to_sim(state[:, 7:], ids)

        def reset(self, env_ids):
            super().reset(env_ids)
            if len(env_ids) == 0:
                return
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            root = self.robot.data.default_root_state[ids].clone()
            root[:, :3] += self.scene.env_origins[ids]
            root[:, 0] = self.scene.env_origins[ids, 0] + torch.where(ids % 2 == 0, -1.6, -1.95)
            root[:, 1] = self.scene.env_origins[ids, 1]
            root[:, 7:] = 0.0
            self.robot.write_root_pose_to_sim(root[:, :7], ids)
            self.robot.write_root_velocity_to_sim(root[:, 7:], ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.compute_perception()
            self.compute_paddle_touch()

        def check_reset(self):
            reset, timeout = super().check_reset()
            heights = self.robot.data.root_link_pos_w[:, 2] - self.scene.env_origins[:, 2]
            tilt = torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
            raw_pitch = math_utils.euler_xyz_from_quat(self.robot.data.root_link_quat_w)[1]
            pitch = torch.remainder(raw_pitch + torch.pi, 2 * torch.pi) - torch.pi
            lower_ids = self.robot.find_joints(lower_names, preserve_order=True)[0]
            foot_ids = self.robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
            for i, record in enumerate(observed):
                if record["first_non_timeout_reset_s"] is not None:
                    continue
                elapsed = float(self.episode_length_buf[i].item()) * self.step_dt
                record["first_episode_min_base_z"] = min(record["first_episode_min_base_z"], heights[i].item())
                record["first_episode_max_tilt"] = max(record["first_episode_max_tilt"], tilt[i].item())
                if tilt[i] > 0.25 and record["first_tilt_over_0_25_s"] is None:
                    foot_mean = self.robot.data.body_link_pos_w[i, foot_ids].mean(dim=0)
                    joint_pos = self.robot.data.joint_pos[i, lower_ids]
                    joint_target = self.robot.data.joint_pos_target[i, lower_ids]
                    record["first_tilt_over_0_25_s"] = elapsed
                    record["pose_at_first_tilt"] = {
                        "base_z": heights[i].item(), "pitch_rad": pitch[i].item(),
                        "pelvis_minus_foot_link_mean_x": (self.robot.data.root_link_pos_w[i, 0] - foot_mean[0]).item(),
                        "lower_joint_names": lower_names, "lower_joint_positions": joint_pos.tolist(),
                        "lower_joint_targets": joint_target.tolist(),
                        "lower_joint_errors": (joint_pos - joint_target).tolist(),
                        "feet_forces_w": self.contact_sensor.data.net_forces_w[i, self.feet_cfg.body_ids].tolist(),
                    }
                if reset[i] and not timeout[i]:
                    record["first_non_timeout_reset_s"] = elapsed
            return reset, timeout

    with torch.no_grad():
        env = StanceFixture(cfg, headless=args.headless)
        assert not env.robot.is_fixed_base and not cfg.scene.robot.spawn.rigid_props.disable_gravity
        frames = max(1, round(args.seconds / env.step_dt))
        for i in range(frames):
            obs, reward, done, extras = env.step(torch.zeros((4, 27), device=env.device))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            if i % max(1, round(2 / env.step_dt)) == 0:
                print(f"[STANCE] Progress {(i+1)*env.step_dt:.2f}/{args.seconds:.2f}s", flush=True)
    verify()
    report = {"cases": observed, "requested_seconds": args.seconds, "neutral_actions": True,
              "model_loaded": False, "learning_iterations": 0, "gains_and_default_joint_pose_changed": False,
              "fixture_material_randomization_disabled": True, "baseline_source_verified": True,
              "scope": "neutral stance comparison with pre-reset observations; no trained balance performance or automatic parameter fix"}
    out = ROOT / "tools/h1_2/neutral_stance_comparison_report.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for record in observed:
        print(f"[STANCE] {record['case']}: tilt>0.25 at={record['first_tilt_over_0_25_s']}; non_timeout_reset_at={record['first_non_timeout_reset_s']}; min_z={record['first_episode_min_base_z']:.3f}; max_tilt={record['first_episode_max_tilt']:.3f}", flush=True)
        if record["pose_at_first_tilt"]:
            print("[STANCE] pitch/joint_errors:", record["pose_at_first_tilt"]["pitch_rad"], record["pose_at_first_tilt"]["lower_joint_errors"], flush=True)
    print("Report:", out, flush=True)
    print("H1-2 NEUTRAL STANCE COMPARISON FINISHED", flush=True)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    try:
        if env is not None:
            env.close()
    finally:
        app.close()
