"""Preserved explicit env-paired PhysX contact views from ready_reset_3."""
import torch

class ContactPairs:
    """One sensor per explicit env path, filter paths paired to that same env."""
    def __init__(self, env, watch):
        self.env = env
        sim = env.contact_sensor._physics_sim_view
        roots = [p+'/Robot/' for p in env.scene.env_prim_paths]
        sensors = [r+n for r in roots for n in watch]
        filters = [[r+'torso_link', r+'pelvis'] for r in roots for n in watch]
        self.body = sim.create_rigid_contact_view(sensors, filter_patterns=filters)
        self.order_body = self._order(self.body, sensors, 2)
        balls = [p+'/Ball' for p in env.scene.env_prim_paths]
        from pxr import Usd, UsdPhysics
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        table_paths = []
        for p in env.scene.env_prim_paths:
            bodies = [str(b.GetPath()) for b in Usd.PrimRange(stage.GetPrimAtPath(p+'/Table'))
                      if b.HasAPI(UsdPhysics.RigidBodyAPI)]
            assert len(bodies) == 1, 'EXPECTED ONE TABLE RIGID BODY PER ENV'
            table_paths.append(bodies[0])
        targets = [[r+'h1_2_paddle_link', table] for r, table in zip(roots, table_paths)]
        self.ball = sim.create_rigid_contact_view(balls, filter_patterns=targets)
        self.order_ball = self._order(self.ball, balls, 2)
        self.nwatch = len(watch)

    def _order(self, view, expected, nfilter):
        assert view.sensor_count == len(expected) and view.filter_count == nfilter, 'CONTACT VIEW SHAPE MISMATCH'
        # Binding order can be lexical (env_10 before env_2). Never assume it.
        paths = list(view.sensor_paths)
        assert len(set(paths)) == len(paths) and set(paths) == set(expected), 'CONTACT SENSOR PATH MISMATCH'
        return torch.tensor([paths.index(p) for p in expected], device=self.env.device)

    def read(self):
        dt = self.env.physics_dt
        b = self.body.get_contact_force_matrix(dt=dt)[self.order_body]
        f = self.ball.get_contact_force_matrix(dt=dt)[self.order_ball]
        assert b.shape == (self.env.num_envs*self.nwatch, 2, 3)
        assert f.shape == (self.env.num_envs, 2, 3)
        self.table_force_z = f[:, 1, 2].abs()
        return torch.linalg.vector_norm(b, dim=-1).reshape(self.env.num_envs, -1).amax(-1), torch.linalg.vector_norm(f, dim=-1)
