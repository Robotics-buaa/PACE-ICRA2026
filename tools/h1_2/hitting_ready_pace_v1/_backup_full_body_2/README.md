# H1-2 ready pose with full-body PACE — full_body_2

This is a new, independent task. Local source/math/deployment checks do not
constitute a PyTorch/Isaac runtime pass. Runtime checks run in the user's Docker.

The original `legged_lab.scripts.train` is imported unchanged and its `train()`
function is called while retaining the module and application globals, with a
process-local new task registration and a subclass of its original predictor
runner. Original PACE PPO, optimizer settings, actor/critic dimensions
(495/580), all 27 action outputs, analytic future targets, future body position
and velocity rewards, and original predictor architecture/training cutoff are
retained. Files, assets, compatibility patches and old checkpoints are not edited.

Revision 2 changes application lifetime/reporting only. User runs of revision 1
reached network construction, sparse reset PASS and actor-prior loading twice,
then exited with only initial reports. No iteration/completed training report
was received; the underlying first error is not established by those logs.
The old wrapper proceeded on exit code 0, but the train entry then rejected its
incomplete check report, so full training did not start. The new wrapper retains
the original module, records milestones/errors before app close, enables fatal
Python stack dumps and validates both process code and completed report. It
preserves the original task learning/physics/budget. Use the version-aware
`H1_2_PACE_full_body_2_fix_and_continue_command.txt` for an installed revision 1;
it backs up revision 1 and refuses unknown source edits.

## Changes in this new task

- Every robot reset uses exact ready joint positions, zero joint/root velocity,
  matching PD position/velocity targets, consistent action/history state and
  right-arm gravity support. Root XY/yaw spawn variation is retained. Right-arm
  joint actions are relative to ready; other channels retain their original
  defaults and all channels retain the original action scale. Original joint
  randomization is overwritten only to establish the requested exact reset.
- Body future targets use the measured ready-paddle offset rather than the old
  neutral-paddle offset. No initial-position anchor or `.5m` drift fence is used;
  original PACE table workspace bounds plus height/tilt health checks apply.
- Learned standing actor weights initialize the new actor after an explicit
  affine observation-reference conversion. The right-arm output head starts at
  zero for ready; the critic and predictor start fresh. Learned standing noise
  scales initialize exploration. All actor weights, including legs/torso and
  predictive input connections, are trainable. No separate controller overrides
  PACE joint actions. The standing checkpoint/config/normalizers stay untouched.
  Initialization provides a prior, not guaranteed balance while chasing.
- Reuse the validated 22 arm/paddle vs torso/pelvis collision pairs, wrist shape
  completion with original inertial properties, hard gaps `.01/.03/.02m`, inner
  buffer `.015/.05/.035m`, `.08s` forecast and `.025rad` path sampling. Chunking
  at 128 envs bounds temporary geometry memory without coarsening checks.
  Existing arm PD-target rates `.6rad/s` shoulder/elbow and `.9rad/s` wrist remain
  conservative initial limits, not a proven final hitting-speed envelope.
- Right-only gravity support, original PD/gains/drive limits and free root with
  gravity remain. Torque headroom checks are estimates, not measured total drive
  torque or a continuous-time collision guarantee.
- Real ball/paddle and tabletop impulses replace old proximity/AABB event flags.
  A legal hit needs a prior measured own-table bounce and incoming negative X
  velocity. First actual opposite-table landing with outgoing positive X counts
  as a return; double hits invalidate the serve. Original predicted landing/net
  rewards remain auxiliary shaping after a real valid hit, not success counts.
  The paddle contact includes the handle; no center-face contact claim is made.
- Original future EE/body/velocity reward functions and weights are retained.
  Right-arm deviation uses ready as its reference; physical invalid-contact,
  early-clearance and height terms are local additions. Shared rewards are not
  edited. All terminal rewards and predictor samples are captured before reset.
- Predictor history windows are filtered by per-env serve epochs and valid
  analytic labels. Inference is masked until a complete current-serve history
  exists. Predictor exceptions propagate rather than silently disabling it.
  Actor's latest learned-prediction fields refresh without appending duplicate
  frames or leaking analytic critic/label data. Original observation noise is
  preserved on refreshed fields.

## Execution and budget

Use `H1_2_PACE_ready_full_10000_install_and_train_command.txt` in Docker. It
verifies received upstream source hashes and existing ready-package protection,
adds this directory only, and runs foreground:

1. New wiring check: 8 environments x 2 PPO iterations. Checks exact/sparse
   resets, 27-action policy updates, predictor updates/inference and joint
   policy+predictor save. This is not a repeat of the completed 20s ready hold.
2. Only if the check succeeds: 4096 environments x 10000 iterations, 24 control
   steps per env per iteration, 983040000 total transitions. This is a fresh run
   with the standing actor prior, not a resumption of old TT or 7-action models.

Outputs: `/workspace/projects/PACE-ICRA2026/logs/h1_2_table_tennis_ready_pace/`.
Original PACE names checkpoints `model_<iteration>.pt`; an additional explicit
`model_final.pt` includes predictor weights/optimizer. Launch and progress reports
record physical counters, task manifest, initialization, and errors. One console
log records both stages. No automatic OOM retry or silent batch-size reduction.
The new collision checks add work and memory beyond the old PACE task; historical
4096-env throughput/memory results do not guarantee this task's performance.

Completed training is not a skill pass. Judge the saved policy by measured valid
hits/returns, failures, contacts and motion, using a separate evaluation. The old
7-action acceptance verifies only that prototype's initial hold/control behavior.
