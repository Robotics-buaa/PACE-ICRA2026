"""Static G1 wrist pose calibration for Robotics-buaa/PACE-ICRA2026.

Place this file in the PACE repository root, then run:
  ../IsaacLab/isaaclab.sh -p debug_g1_wrist_pose.py

No policy/checkpoint, ball, training, or physics stepping is used. Sliders
directly set joint states for forward kinematics. This does NOT validate
collisions, actuator feasibility, balance, or dynamic reachability.

GUI: sliders in radians; Print pose; Save prepare; Save impact; Reset; Quit.
JSON: position and normal in full root-link (pelvis) coordinates, heading-only
coordinates, world coordinates, complete joint configuration and root pose.
Existing JSON snapshots are retained; saving the same label creates a new
numbered key. Use --output to select the local JSON destination.

Prepared against feat/pjh-run asset names and Isaac Lab 2.1 APIs. Syntax
checked only; execute in the user's existing Isaac Sim / PACE environment.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="debug_poses/g1_wrist_poses.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    parser.error("This tool requires a GUI; omit --headless.")
launcher = AppLauncher(args)
app = launcher.app

import torch
import omni.ui as ui
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from legged_lab.assets.unitree.g1_tt_wrist import G1_TT_WRIST_CFG
from legged_lab.assets.unitree.g1_tt import (
    G1_PADDLE_BODY_NAME, G1_PADDLE_CENTER_OFFSET, G1_PADDLE_NORMAL,
)
from legged_lab.assets.table_tennis.table import TABLE_CFG

JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "right_palm_joint",
    "torso_joint",
]


def main():
    output = Path(args.output).expanduser().resolve()
    records = {}
    if output.exists():
        records = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(records, dict):
            raise ValueError(f"Expected a JSON object in {output}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=0.002, device=args.device,
    ))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0)
    light.func("/World/Light", light)
    robot_cfg = G1_TT_WRIST_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot = Articulation(robot_cfg)
    table_cfg = TABLE_CFG.copy()
    table_cfg.prim_path = "/World/Table"
    table = RigidObject(table_cfg)
    sim.reset()
    sim.set_camera_view(eye=(0.5, 2.6, 1.8), target=(-1.5, 0.0, 0.9))

    ids, names = robot.find_joints(JOINTS, preserve_order=True)
    if names != JOINTS:
        raise RuntimeError(f"Expected joints {JOINTS}; got {names}")
    body_ids, body_names = robot.find_bodies(G1_PADDLE_BODY_NAME)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one paddle body, got {body_names}")
    paddle_id = body_ids[0]

    q = robot.data.default_joint_pos.clone()
    q_default = q.clone()
    root = robot.data.default_root_state.clone()
    # Keep the pelvis upright, facing +world X. No environment randomization.
    root[:, 3:7] = root.new_tensor([1.0, 0.0, 0.0, 0.0])
    root[:, 7:] = 0.0
    limits = robot.data.soft_joint_pos_limits[0]
    center_offset = q.new_tensor(G1_PADDLE_CENTER_OFFSET).reshape(1, 3)
    local_normal = q.new_tensor(G1_PADDLE_NORMAL).reshape(1, 3)
    pending = []
    slider_models = []
    running = True

    def snapshot():
        base_p = robot.data.root_link_pos_w
        base_q = robot.data.root_link_quat_w
        paddle_q = robot.data.body_quat_w[:, paddle_id]
        paddle_p = robot.data.body_pos_w[:, paddle_id] + math_utils.quat_apply(
            paddle_q, center_offset,
        )
        normal_w = math_utils.quat_apply(paddle_q, local_normal)
        rel_w = paddle_p - base_p
        pos_b = math_utils.quat_rotate_inverse(base_q, rel_w)
        normal_b = math_utils.quat_rotate_inverse(base_q, normal_w)
        heading_q = math_utils.yaw_quat(base_q)
        pos_heading = math_utils.quat_rotate_inverse(heading_q, rel_w)
        normal_heading = math_utils.quat_rotate_inverse(heading_q, normal_w)
        return {
            "mode": "static_kinematic_calibration",
            "length_unit": "m", "angle_unit": "rad",
            "paddle_position_b": pos_b[0].tolist(),
            "paddle_normal_b": normal_b[0].tolist(),
            "paddle_position_heading": pos_heading[0].tolist(),
            "paddle_normal_heading": normal_heading[0].tolist(),
            "paddle_position_w": paddle_p[0].tolist(),
            "root_position_w": base_p[0].tolist(),
            "root_quaternion_wxyz": base_q[0].tolist(),
            "adjustable_joints_rad": {name: q[0, idx].item()
                                      for idx, name in zip(ids, names)},
            "all_joints_rad": dict(zip(robot.joint_names, q[0].tolist())),
        }

    window = ui.Window("G1 wrist pose calibration", width=520, height=880)
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("Static pose only | radians | +X forward, +Y left, +Z up",
                     height=28)
            ui.Label("Pelvis locked. No policy or physics stepping.\n"
                     "Check shoulder, elbow, paddle face and table clearance.",
                     height=42)
            for idx, name in zip(ids, names):
                lo, hi = limits[idx].tolist()
                if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                    raise RuntimeError(f"Invalid slider limits for {name}: {lo}, {hi}")
                ui.Label(f"{name}  [{lo:.2f}, {hi:.2f}]", height=22)
                slider = ui.FloatSlider(min=lo, max=hi, height=22)
                slider.model.set_value(float(q[0, idx]))

                def update(model, joint_id=idx, lower=lo, upper=hi):
                    value = model.as_float
                    if math.isfinite(value):
                        q[0, joint_id] = max(lower, min(upper, value))

                slider.model.add_value_changed_fn(update)
                slider_models.append((idx, slider.model))
            with ui.HStack(height=30):
                ui.Button("Print pose", clicked_fn=lambda: pending.append("print"))
                ui.Button("Reset", clicked_fn=lambda: pending.append("reset"))
                ui.Button("Quit", clicked_fn=lambda: pending.append("quit"))
            with ui.HStack(height=30):
                ui.Button("Save prepare", clicked_fn=lambda: pending.append("prepare"))
                ui.Button("Save impact", clicked_fn=lambda: pending.append("impact"))
            info = ui.Label("Adjust sliders, then click Print pose.",
                            height=120, word_wrap=True)

    print("\n[POSE] GUI ready. Sliders are in radians, positions in metres.", flush=True)
    print(f"[POSE] JSON output: {output}", flush=True)
    print("[POSE] No checkpoint is loaded. Static geometry only.", flush=True)
    try:
        while app.is_running() and running:
            started = time.monotonic()
            # Refresh forward kinematics, without advancing the physical world.
            robot.write_root_pose_to_sim(root[:, :7])
            robot.write_root_velocity_to_sim(root[:, 7:])
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
            sim.forward()
            robot.update(1.0 / 30.0)

            commands = pending[:]
            pending.clear()
            for command in commands:
                if command == "quit":
                    running = False
                    break
                if command == "reset":
                    q.copy_(q_default)
                    for idx, model in slider_models:
                        model.set_value(float(q[0, idx]))
                    info.text = "Reset to asset default pose."
                    continue
                data = snapshot()
                print("\n[POSE] " + command, flush=True)
                print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
                info.text = (
                    f"position_b [m]: {data['paddle_position_b']}\n"
                    f"normal_b: {data['paddle_normal_b']}\n"
                    f"Operation: {command}"
                )
                if command in ("prepare", "impact"):
                    key = command
                    suffix = 2
                    while key in records:
                        key = f"{command}_{suffix}"
                        suffix += 1
                    records[key] = data
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = output.with_name(output.name + ".tmp")
                    temp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
                    os.replace(temp_path, output)
                    print(f"[POSE] Saved {key} -> {output}", flush=True)
                    info.text += f"\nSaved as: {key}"
            sim.render()
            time.sleep(max(0.0, 1.0 / 30.0 - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        # Retain object references through the end of the visualization loop.
        window.visible = False


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
