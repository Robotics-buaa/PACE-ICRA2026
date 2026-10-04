"""Install the G1 robot/paddle interface and check it in a full PACE scene."""

import argparse
import ast
import faulthandler
import hashlib
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT = Path("/workspace/projects/PACE-ICRA2026")
ENV_SOURCE = r'''"""G1-specific robot and paddle interfaces for the existing PACE environment."""

import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import CircularBuffer

from legged_lab.assets.unitree.g1_tt import (
    G1_PADDLE_BODY_NAME,
    G1_PADDLE_CENTER_OFFSET,
    G1_PADDLE_NORMAL,
)
from legged_lab.envs.base.tt_env import TTEnv
from .g1_tt_config import G1_HAND_JOINTS


class G1TTEnv(TTEnv):
    def init_obs_buffer(self):
        if self.add_noise:
            return super().init_obs_buffer()
        # The base implementation only defines actor_obs when noise is enabled.
        actor_obs, _ = self.compute_current_observations()
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.delayed_perception = actor_obs[..., -self.num_perception:]

    def init_buffers(self):
        self._initialize_g1_geometry()
        super().init_buffers()
        body_ids, names = self.robot.find_bodies(G1_PADDLE_BODY_NAME)
        if names != [G1_PADDLE_BODY_NAME]:
            raise RuntimeError(f"Expected one G1 paddle body, found {names}")
        self.paddle_body_id = body_ids[0]
        action_ids = set(self.action_joint_ids)
        self.held_joint_ids = [
            i for i in range(self.robot.num_joints) if i not in action_ids
        ]
        held_names = {self.robot.joint_names[i] for i in self.held_joint_ids}
        if held_names != set(G1_HAND_JOINTS):
            raise RuntimeError(f"Unexpected joints outside G1 actions: {held_names}")
        self._paddle_center_offset = torch.tensor(
            G1_PADDLE_CENTER_OFFSET, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self._paddle_local_normal = torch.tensor(
            G1_PADDLE_NORMAL, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        self._ball_radius = float(self.cfg.scene.ball.spawn.radius)
        self._hold_hand_joints()

    def _hold_hand_joints(self):
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[:, self.held_joint_ids],
            joint_ids=self.held_joint_ids,
        )

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids][:, self.held_joint_ids],
            self.robot.data.default_joint_vel[env_ids][:, self.held_joint_ids],
            joint_ids=self.held_joint_ids,
            env_ids=env_ids,
        )
        self._hold_hand_joints()
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.compute_paddle_touch()

    def step(self, actions):
        self._hold_hand_joints()
        return super().step(actions)

    def compute_paddle_touch(self):
        self.ball_global_pos = self.ball.data.root_pos_w
        body_pos = self.robot.data.body_pos_w[:, self.paddle_body_id]
        body_quat = self.robot.data.body_quat_w[:, self.paddle_body_id]
        self.paddle_quat = body_quat / body_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.paddle_touch_point = body_pos + math_utils.quat_apply(
            self.paddle_quat, self._paddle_center_offset
        )
        self.paddle_normal = math_utils.quat_apply(
            self.paddle_quat, self._paddle_local_normal
        )
        self.paddle_pos = self.paddle_touch_point - self.scene.env_origins

        # Preserve PACE's distance score and peak/retreat hit flag semantics.
        distance = torch.norm(self.ball_global_pos - self.paddle_touch_point, dim=1)
        self.paddel_ball_distance = distance - self._ball_radius
        contact_score = (
            self.cfg.ball.contact_threshold - self.paddel_ball_distance
        ) / self.cfg.ball.contact_threshold
        self.ball_contact = torch.clamp(contact_score, min=0.0, max=1.0)
        self.ball_contact_rew = torch.maximum(self.ball_contact_rew, self.ball_contact)
        new_hits = (contact_score > 0) & (self.ball_contact < self.ball_contact_rew)
        still_false = ~self.has_touch_paddle
        self.has_touch_paddle[still_false] = new_hits[still_false]

    def _initialize_g1_geometry(self):
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


def install(project):
    cfg_path = project / "legged_lab/envs/g1_tt/g1_tt_config.py"
    env_path = cfg_path.with_name("g1_tt_env.py")
    t1_path = project / "legged_lab/envs/t1_tt/t1_tt_config.py"
    t1_digest = hashlib.sha256(t1_path.read_bytes()).hexdigest()
    source = cfg_path.read_text()
    tree = ast.parse(source)
    reward_func = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "_make_g1_rewards")
    edits = []
    for label, replacement in (
        ("undesired_contacts", '    reward.undesired_contacts.params["sensor_cfg"].body_names = "(?!(?:.*_ankle_roll_link|g1_paddle_link)$).*"\n'),
        ("paddel_head_too_near", '    reward.paddel_head_too_near.params["asset_cfg"].body_names = ["head_link"]\n'),
    ):
        matches = [n for n in ast.walk(reward_func) if isinstance(n, ast.Assign)
                   and label in ast.dump(n.targets[0])]
        assert len(matches) == 1, ("Cannot safely locate G1 reward assignment", label)
        node = matches[0]
        edits.append((node.lineno - 1, node.end_lineno, replacement))
    env_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "G1TableTennisEnvCfg")
    post_init = next(n for n in env_cls.body if isinstance(n, ast.FunctionDef)
                     and n.name == "__post_init__")
    material_marker = 'self.domain_rand.events.physics_material.params["asset_cfg"].body_names'
    if material_marker not in ast.get_source_segment(source, post_init):
        edits.append((post_init.end_lineno, post_init.end_lineno,
                      '\n        # Preserve the separately authored paddle bounce material.\n'
                      '        self.domain_rand.events.physics_material.params["asset_cfg"].body_names = "(?!g1_paddle_link$).*"\n'))
    lines = source.splitlines(keepends=True)
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    updated = "".join(lines)
    updated = updated.replace(
        "# Restore with a verified G1 reference body during paddle integration.",
        "# Use the head body verified in the loaded G1 articulation.",
    )
    compile(updated, str(cfg_path), "exec")
    compile(ENV_SOURCE, str(env_path), "exec")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for path, new_source in ((cfg_path, updated), (env_path, ENV_SOURCE)):
        if path.exists() and path.read_text() == new_source:
            continue
        if path.exists():
            backup = path.with_name(path.name + ".step4_" + stamp + ".bak")
            shutil.copy2(path, backup)
            print("Backup:", backup, flush=True)
        path.write_text(new_source)
    assert hashlib.sha256(t1_path.read_bytes()).hexdigest() == t1_digest
    print("[PASS] G1 interface installed; original T1 configuration unchanged", flush=True)


def check(project):
    sys.path.insert(0, str(project))
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch
        from isaaclab.managers import SceneEntityCfg
        from legged_lab.assets.unitree.g1_tt import G1_PADDLE_BODY_NAME
        from legged_lab.envs.g1_tt.g1_tt_config import (
            G1_CONTROL_JOINTS, G1_HAND_JOINTS, G1TableTennisEnvCfg,
        )
        from legged_lab.envs.g1_tt.g1_tt_env import G1TTEnv

        cfg = G1TableTennisEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.seed = 42
        cfg.noise.add_noise = False
        cfg.domain_rand.action_delay.enable = False
        cfg.domain_rand.perception_delay.enable = False
        # Deterministic short interface check. Keep startup material randomization.
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
        assert env.num_actions == 23 and env.robot.num_joints == 37
        actual_actions = [env.robot.joint_names[i] for i in env.action_joint_ids]
        actual_obs = [env.robot.joint_names[i] for i in env.obs_joint_ids]
        assert actual_actions == G1_CONTROL_JOINTS and actual_obs == G1_CONTROL_JOINTS
        assert {env.robot.joint_names[i] for i in env.held_joint_ids} == set(G1_HAND_JOINTS)
        assert env.robot.body_names[env.paddle_body_id] == G1_PADDLE_BODY_NAME
        print("[PASS] Full G1 scene: 23 ordered actions, 37 joints, named paddle body", flush=True)

        material_cfg = cfg.domain_rand.events.physics_material.params["asset_cfg"]
        material_check = SceneEntityCfg("robot", body_names=material_cfg.body_names)
        material_check.resolve(env.scene)
        assert isinstance(material_check.body_ids, list)
        assert env.paddle_body_id not in material_check.body_ids
        head_cfg = SceneEntityCfg("robot", body_names="head_link")
        head_cfg.resolve(env.scene)
        assert len(head_cfg.body_ids) == 1
        head_term = env.reward_manager.get_term_cfg("paddel_head_too_near")
        assert head_term.params["asset_cfg"].body_ids == head_cfg.body_ids
        contact_cfg = env.reward_manager.get_term_cfg("undesired_contacts").params["sensor_cfg"]
        sensor_names = env.contact_sensor.body_names
        penalized_names = [sensor_names[i] for i in contact_cfg.body_ids]
        assert G1_PADDLE_BODY_NAME not in penalized_names
        assert "left_ankle_roll_link" not in penalized_names
        assert "right_ankle_roll_link" not in penalized_names

        # Same per-link PhysX view API used by Isaac Lab's material event.
        paddle_path = env.robot.root_physx_view.link_paths[0][env.paddle_body_id]
        paddle_view = env.robot._physics_sim_view.create_rigid_body_view(paddle_path)
        materials = paddle_view.get_material_properties()
        assert bool(torch.allclose(materials[..., 2], torch.full_like(materials[..., 2], 0.8), atol=1e-5)), materials
        print("[PASS] Head reward restored; paddle excluded from contact penalty and material randomization", flush=True)
        print("[PASS] Paddle restitution remains 0.8 after startup events", flush=True)

        actions = torch.zeros((1, 23), device=env.device)
        for _ in range(10):
            obs, reward, done, extras = env.step(actions)
            assert bool(torch.isfinite(obs).all()), "Non-finite actor observations"
            assert bool(torch.isfinite(extras["observations"]["critic"]).all()), "Non-finite critic observations"
            assert bool(torch.isfinite(reward).all()), "Non-finite rewards"
        assert torch.allclose(
            env.robot.data.joint_pos_target[:, env.held_joint_ids],
            env.robot.data.default_joint_pos[:, env.held_joint_ids],
        )
        print("[PASS] 10 control steps: finite observations/rewards; actor=%s, critic=%s" % (
            tuple(obs.shape), tuple(extras["observations"]["critic"].shape)), flush=True)

        ids = torch.arange(1, device=env.device)
        held_default = env.robot.data.default_joint_pos[:, env.held_joint_ids].clone()
        # Disturb the fingers, then verify reset really restores their state.
        limits = env.robot.data.soft_joint_pos_limits[:, env.held_joint_ids]
        disturbed = (held_default + 0.05).clamp(limits[..., 0], limits[..., 1])
        assert bool((disturbed - held_default).abs().max() > 0.01)
        env.robot.write_joint_state_to_sim(
            disturbed, torch.zeros_like(disturbed), joint_ids=env.held_joint_ids, env_ids=ids,
        )
        env.reset(ids)
        assert torch.allclose(env.robot.data.joint_pos[:, env.held_joint_ids], held_default, atol=1e-6)
        assert torch.allclose(env.robot.data.joint_pos_target[:, env.held_joint_ids], held_default)
        print("[PASS] All 14 hand joints held at default targets and restored by reset", flush=True)

        env.compute_paddle_touch()
        paddle_error = (env.paddle_touch_point - env.robot.data.body_pos_w[:, env.paddle_body_id]).norm(dim=-1)
        assert float(paddle_error.max()) < 1e-6
        assert torch.allclose(env.paddle_normal.norm(dim=-1), torch.ones(1, device=env.device), atol=1e-5)
        assert torch.allclose(env.paddle_pos, env.paddle_touch_point - env.scene.env_origins)
        print("[PASS] Paddle center and normal follow the G1 paddle rigid body", flush=True)

        # Check the existing PACE proximity/retreat interface without advancing physics.
        env.has_touch_paddle.zero_()
        env.ball_contact_rew.zero_()
        scores = []
        for distance in (0.20, 0.022, 0.040):
            pose = env.ball.data.root_state_w[:, :7].clone()
            pose[:, :3] = env.paddle_touch_point + distance * env.paddle_normal
            env.ball.write_root_pose_to_sim(pose)
            env.ball.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
            env.sim.forward()
            env.scene.update(env.physics_dt)
            env.compute_paddle_touch()
            scores.append(float(env.ball_contact.item()))
        assert scores[0] == 0.0 and scores[1] > scores[2] > 0.0, scores
        assert bool(env.has_touch_paddle.item()), "PACE hit flag did not trigger after retreat"
        print("[PASS] PACE approach/retreat contact score and hit flag use the G1 paddle", flush=True)
        print("STEP 4 INTERFACE CHECK PASSED", flush=True)
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
