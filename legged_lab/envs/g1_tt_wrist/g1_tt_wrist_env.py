"""Use the G1 task interfaces with one controlled right-wrist joint."""
from legged_lab.envs.g1_tt.g1_tt_env import G1TTEnv
from legged_lab.assets.unitree.g1_tt_wrist import G1_WRIST_JOINT_NAME


class G1WristTTEnv(G1TTEnv):
    def init_buffers(self):
        super().init_buffers()
        ids, names = self.robot.find_joints(G1_WRIST_JOINT_NAME)
        if names != [G1_WRIST_JOINT_NAME] or len(ids) != 1:
            raise RuntimeError("Expected one active right_palm_joint")
        self.wrist_joint_ids = ids
        if ids[0] not in self.action_joint_ids or ids[0] in self.held_joint_ids:
            raise RuntimeError("Wrist must be policy-controlled, not held with fingers")

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids][:, self.wrist_joint_ids],
            self.robot.data.default_joint_vel[env_ids][:, self.wrist_joint_ids],
            joint_ids=self.wrist_joint_ids, env_ids=env_ids,
        )
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[env_ids][:, self.wrist_joint_ids],
            joint_ids=self.wrist_joint_ids, env_ids=env_ids,
        )
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.compute_paddle_touch()
