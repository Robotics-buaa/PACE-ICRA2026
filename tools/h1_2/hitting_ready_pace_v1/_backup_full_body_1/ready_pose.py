"""Ready target computed from the received real URDF/limits/collider bounds.

This is a kinematic candidate, never a simulator joint-state assignment. Live FK,
limits, normals and the entire sampled setup path must pass again before use.
"""
RIGHT_READY_RAD = {
    'right_shoulder_pitch_joint': -0.5552144708175045,
    'right_shoulder_roll_joint': -0.16977567600415505,
    'right_shoulder_yaw_joint': -0.318247055241751,
    'right_elbow_joint': 0.06346191115229871,
    'right_wrist_roll_joint': -0.9872064718920872,
    'right_wrist_pitch_joint': 0.08925006937905065,
    'right_wrist_yaw_joint': -1.03,
}
REFERENCE_PADDLE_CENTER_TORSO_M = (.38, -.38, .40)
REFERENCE_PADDLE_NORMAL_TORSO = (1., 0., 0.)
SETUP_MOVE_S = 4.
SOURCE_URDF_SHA256 = 'e5dae04b4f16590d34a64d8bb9a0bcc2a24072f79b05ec4371d49866fb7228bd'
