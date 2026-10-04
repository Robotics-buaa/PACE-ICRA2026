"""Observe an existing G1 policy at physics-step resolution; never train or patch assets.

Copy this file to the PACE project root and run with Isaac Lab's Python.
Velocity-change events are collision candidates, not pair-specific contact proofs.
"""
import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import MethodType


def face_metrics(relative, normal):
    length = math.sqrt(sum(v * v for v in normal))
    normal = [v / max(length, 1e-12) for v in normal]
    plane = sum(p * n for p, n in zip(relative, normal))
    radial = math.sqrt(max(0.0, sum(v * v for v in relative) - plane * plane))
    angle = math.degrees(math.acos(min(1.0, abs(normal[0]))))
    return plane, radial, angle


class Observer:
    def __init__(self, env, folder, radius, half_thickness):
        self.env = env
        self.radius = radius
        self.half_thickness = half_thickness
        self.ball_radius = float(env.cfg.scene.ball.spawn.radius)
        self.rows = []
        self.path = folder / "paddle_diagnostics.csv"
        self.arm_ids = [i for i, name in enumerate(env.robot.joint_names)
                        if name.startswith("right_") and ("shoulder" in name or "elbow" in name)]
        self.arm_names = [env.robot.joint_names[i] for i in self.arm_ids]
        self.start()

    def start(self):
        self.samples = 0
        self.nearest = None
        self.previous = None
        self.hit = False
        self.success = False
        self.face_samples = 0
        self.velocity_candidates = 0
        self.forward_candidates = 0
        self.peak_score = 0.0
        self.impulse = {
            "candidate_dv_mps": 0.0,
            **{"candidate_v%s_%s" % (axis, when): None
               for axis in ("x", "y", "z") for when in ("before", "after")},
        }

    def observe(self):
        env = self.env
        self.samples += 1
        # Read the actual ball state, avoiding perception delay/noise.
        position = env.ball.data.root_pos_w[0].detach().cpu().tolist()
        velocity = env.ball.data.root_lin_vel_w[0].detach().cpu().tolist()
        center = env.paddle_touch_point[0].detach().cpu().tolist()
        normal = env.paddle_normal[0].detach().cpu().tolist()
        relative = [p - c for p, c in zip(position, center)]
        distance = math.sqrt(sum(v * v for v in relative))
        plane, radial, angle = face_metrics(relative, normal)
        face_near = (abs(plane) <= self.ball_radius + self.half_thickness + 0.004
                     and radial <= self.radius + self.ball_radius)
        self.face_samples += int(face_near)
        self.hit |= bool(env.has_touch_paddle[0].item())
        self.peak_score = max(self.peak_score, float(env.ball_contact[0].item()))
        # Also retain a near-paddle *candidate* velocity impulse. Table/other
        # collisions can produce this too; the output deliberately makes no
        # pair-specific contact claim. The finite distance gate excludes serves.
        if self.previous is not None:
            prev_pos, prev_vel, prev_near = self.previous
            moved = math.sqrt(sum((a-b)**2 for a, b in zip(position, prev_pos)))
            dv = math.sqrt(sum((a-b)**2 for a, b in zip(velocity, prev_vel)))
            if moved < 0.10 and (face_near or prev_near) and dv > 1.0:
                self.velocity_candidates += 1
                self.forward_candidates += int(prev_vel[0] < -0.5 and velocity[0] > 0.5)
                if dv > self.impulse["candidate_dv_mps"]:
                    self.impulse = {
                        "candidate_dv_mps": dv,
                        **{"candidate_v%s_before" % axis: prev_vel[i]
                           for i, axis in enumerate(("x", "y", "z"))},
                        **{"candidate_v%s_after" % axis: velocity[i]
                           for i, axis in enumerate(("x", "y", "z"))},
                    }
        self.previous = (position, velocity, face_near)
        if self.nearest is None or distance < self.nearest["center_distance_m"]:
            q = env.robot.data.joint_pos[0, self.arm_ids].detach().cpu().tolist()
            limits = env.robot.data.soft_joint_pos_limits[0, self.arm_ids].detach().cpu().tolist()
            margin = min(min(value-lo, hi-value) for value, (lo, hi) in zip(q, limits))
            self.nearest = {
                "center_distance_m": distance,
                "plane_distance_m": plane,
                "radial_distance_m": radial,
                "face_gap_m": abs(plane) - self.half_thickness - self.ball_radius,
                "normal_x": normal[0], "normal_y": normal[1], "normal_z": normal[2],
                "normal_angle_to_x_deg": angle,
                "ball_vx": velocity[0], "ball_vy": velocity[1], "ball_vz": velocity[2],
                "arm_min_soft_limit_margin_rad": margin,
                **{"q_" + name: value for name, value in zip(self.arm_names, q)},
            }

    def finish(self, reason):
        if self.nearest is None:
            self.start()
            return
        row = {
            "segment": len(self.rows) + 1, "end_reason": reason,
            "physics_samples": self.samples, "pace_hit_flag": int(self.hit),
            "success_flag": int(self.success), "peak_distance_score": self.peak_score,
            "near_face_samples": self.face_samples,
            "velocity_change_candidates": self.velocity_candidates,
            "forward_reversal_candidates": self.forward_candidates,
            **self.nearest, **self.impulse,
        }
        self.rows.append(row)
        with self.path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        print("[G1-DIAG] segment=%d reason=%s hit=%d success=%d "
              "distance=%.3fm face_gap=%.3fm radial=%.3fm "
              "angle_to_x=%.1fdeg normal=(%.2f,%.2f,%.2f) "
              "vx_at_nearest=%.2f face_samples=%d dv_candidates=%d forward=%d arm_margin=%.3frad"
              % (row["segment"], reason, row["pace_hit_flag"], row["success_flag"],
                 row["center_distance_m"], row["face_gap_m"], row["radial_distance_m"],
                 row["normal_angle_to_x_deg"], row["normal_x"], row["normal_y"], row["normal_z"],
                 row["ball_vx"], row["near_face_samples"], row["velocity_change_candidates"],
                 row["forward_reversal_candidates"], row["arm_min_soft_limit_margin_rad"]), flush=True)
        self.start()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load_run", default=None)
    parser.add_argument("--checkpoint", default="model_9999.pt")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--segments", type=int, default=20)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps <= 0 or args.segments <= 0:
        parser.error("steps and segments must be positive")
    project = Path.cwd()
    if not (project / "legged_lab/envs/g1_tt/g1_tt_env.py").is_file():
        parser.error("Run from /workspace/projects/PACE-ICRA2026")
    sys.path.insert(0, str(project))
    log_root = project / "logs/g1_table_tennis"
    if args.load_run:
        run = log_root / args.load_run
    else:
        candidates = sorted(p for p in log_root.glob("*_g1_pred_4096_full")
                            if p.is_dir() and (p / args.checkpoint).is_file())
        if not candidates:
            parser.error("No completed G1 full-training run found; supply --load_run")
        run = candidates[-1]
    checkpoint = run / args.checkpoint
    if not checkpoint.is_file():
        parser.error("Checkpoint not found: " + str(checkpoint))
    app = AppLauncher(args).app
    env = observer = None
    try:
        import torch
        import legged_lab.envs  # registers the tasks after Kit starts
        from legged_lab.utils import task_registry
        from legged_lab.assets.unitree.g1_tt import G1_PADDLE_RADIUS, G1_PADDLE_HALF_THICKNESS
        from rsl_rl.runners import OnPolicyPredictorRegressionRunner

        # Match the existing eval entry's runtime overrides, leaving its source intact.
        cfg, agent = task_registry.get_cfgs("g1_tt_eval")
        cfg, agent = deepcopy(cfg), deepcopy(agent)
        cfg.scene.num_envs = 1
        cfg.scene.env_spacing = 5
        cfg.scene.seed = agent.seed
        cfg.noise.add_noise = True
        cfg.domain_rand.events.push_robot = None
        cfg.scene.height_scanner.drift_range = (0.0, 0.0)
        env = task_registry.get_task_class("g1_tt_eval")(cfg, headless=args.headless)
        runner = OnPolicyPredictorRegressionRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        folder = run / ("paddle_diagnostics_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        folder.mkdir()
        observer = Observer(env, folder, G1_PADDLE_RADIUS, G1_PADDLE_HALF_THICKNESS)
        paddle_path = env.robot.root_physx_view.link_paths[0][env.paddle_body_id]
        view = env.robot._physics_sim_view.create_rigid_body_view(paddle_path)
        materials = view.get_material_properties().detach().cpu()
        metadata = {
            "checkpoint": str(checkpoint), "actor_width": 435, "critic_width": 520,
            "local_paddle_radius": G1_PADDLE_RADIUS,
            "local_paddle_half_thickness": G1_PADDLE_HALF_THICKNESS,
            "paddle_materials_static_dynamic_restitution": materials.tolist(),
            "action_scale": float(env.action_scale), "action_clip": float(env.clip_actions),
            "pace_distance_contact_threshold_m": float(env.cfg.ball.contact_threshold),
            "arm_names": observer.arm_names,
            "arm_default_positions": env.robot.data.default_joint_pos[0, observer.arm_ids].cpu().tolist(),
            "arm_soft_joint_limits": env.robot.data.soft_joint_pos_limits[0, observer.arm_ids].cpu().tolist(),
            "near_face_tolerance_m": 0.004, "velocity_change_threshold_mps": 1.0,
            "note": "Distance hits and velocity-change candidates are not pair-specific physical contact proof. "
                    "Angle to X is a coarse axis diagnostic, not an optimal return angle. "
                    "Rows are segments closed by ball reset, robot reset, or program exit; initial warmup included.",
        }
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("[G1-DIAG] CHECKPOINT:", checkpoint, flush=True)
        print("[G1-DIAG] Paddle restitution:", materials[..., 2].tolist(), flush=True)
        print("[G1-DIAG] Right-arm joints:", observer.arm_names, flush=True)
        print("[G1-DIAG] CSV:", observer.path, flush=True)
        original_touch = env.compute_paddle_touch
        original_intermediate = env.compute_intermediate_values
        original_reset_ball = env.reset_ball
        original_reset = env.reset

        def touch(this):
            original_touch()
            observer.observe()

        def intermediate(this):
            original_intermediate()
            observer.success |= bool(this.has_touch_opponent_table_just_now[0].item()
                                     and this.has_touch_paddle[0].item())

        def reset_ball(this, ids):
            if ids.numel() and bool((ids == 0).any().item()):
                observer.finish("ball_reset")
            return original_reset_ball(ids)

        def reset(this, ids):
            if len(ids) and bool((ids == 0).any().item()):
                observer.finish("robot_reset")
            return original_reset(ids)

        env.compute_paddle_touch = MethodType(touch, env)
        env.compute_intermediate_values = MethodType(intermediate, env)
        env.reset_ball = MethodType(reset_ball, env)
        env.reset = MethodType(reset, env)
        obs, _ = env.get_observations()
        with torch.inference_mode():
            for _ in range(args.steps):
                if not app.is_running() or len(observer.rows) >= args.segments:
                    break
                obs, _, _, _ = env.step(policy(obs))
                if not bool(torch.isfinite(obs).all()):
                    raise RuntimeError("Non-finite observations")
                runner._record_ball_positions()
                runner._maybe_predict_and_update_env()
    except KeyboardInterrupt:
        print("[G1-DIAG] Interrupted; saving the final partial segment.", flush=True)
    finally:
        if observer is not None:
            observer.finish("program_exit")
            print("[G1-DIAG] Saved:", observer.path, flush=True)
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
