"""Repair no-noise initialization in G1 only and expose step 4 exceptions."""

import ast
import shutil
from datetime import datetime
from pathlib import Path

PROJECT = Path("/workspace/projects/PACE-ICRA2026")
METHOD = '''    def init_obs_buffer(self):
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

'''


def repair(project):
    env_path = project / "legged_lab/envs/g1_tt/g1_tt_env.py"
    check_path = project / "step4_g1_interface.py"
    updates = []
    for path in (env_path, check_path):
        source = path.read_text()
        anchor = "class G1TTEnv(TTEnv):\n"
        assert source.count(anchor) == 1, ("Unexpected G1 source", path)
        if "def init_obs_buffer(self):" not in source:
            source = source.replace(anchor, anchor + METHOD, 1)
            import_anchor = "import isaaclab.utils.math as math_utils\n"
            assert source.count(import_anchor) == 1
            source = source.replace(
                import_anchor,
                import_anchor + "from isaaclab.utils.buffers import CircularBuffer\n",
                1,
            )
        if path == check_path and "import faulthandler\n" not in source:
            source = source.replace("import ast\n", "import ast\nimport faulthandler\n", 1)
            source = source.replace("import sys\n", "import sys\nimport traceback\n", 1)
            anchor = "def check(project):\n    sys.path.insert(0, str(project))\n"
            assert source.count(anchor) == 1
            source = source.replace(
                anchor, anchor + "    faulthandler.enable()\n"
                "    faulthandler.dump_traceback_later(60, repeat=True)\n", 1,
            )
            anchor = "    finally:\n        if env is not None:\n"
            assert source.count(anchor) == 1
            source = source.replace(
                anchor, "    except BaseException:\n        traceback.print_exc()\n"
                "        raise\n" + anchor, 1,
            )
            source = source.replace(
                "        app.close()\n", "        app.close()\n"
                "        faulthandler.cancel_dump_traceback_later()\n", 1,
            )
        compile(source, str(path), "exec")
        if path == check_path:
            tree = ast.parse(source)
            embedded = next(n.value.value for n in tree.body
                            if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "ENV_SOURCE"
                                    for t in n.targets))
            compile(embedded, "embedded_g1_tt_env.py", "exec")
            assert "def init_obs_buffer(self):" in embedded
        updates.append((path, source))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for path, source in updates:
        if path.read_text() == source:
            continue
        backup = path.with_name(path.name + ".noise_fix_" + stamp + ".bak")
        shutil.copy2(path, backup)
        path.write_text(source)
        print("Backup:", backup)
    print("[PASS] G1 no-noise initialization repaired; step 4 diagnostics enabled")


if __name__ == "__main__":
    repair(PROJECT)
