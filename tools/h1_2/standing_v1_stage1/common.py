"""Shared paths/protection for a bounded, separate standing learning stage."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/workspace/projects/PACE-ICRA2026')
HERE = Path(__file__).resolve().parent
STAND = HERE.parent/'standing_v1'
EXPERIMENT = ROOT/'logs/h1_2_standing_v1'
INTERFACE_REPORT = STAND/'interface_report_20261016T124508_373805Z.json'
OLD_CHECKPOINT = ROOT/'logs/h1_2_table_tennis/2026-10-15_03-34-22_h1_2_pred_4096_full/model_9999.pt'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    package = json.loads((HERE/'manifest.json').read_text())
    baseline = json.loads((ROOT/'tools/h1_2/bilateral_install_report.json').read_text())
    protected = {}
    for group in (baseline['protected_sha256'], baseline['new_files_sha256'],
                  package['required_source_sha256']):
        for name, digest in group.items():
            assert name not in protected or protected[name] == digest, 'CONFLICTING PROTECTION DATA: '+name
            protected[name] = digest
    for name, digest in protected.items():
        assert sha(ROOT/name) == digest, 'PROTECTED SOURCE CHANGED: '+name
    for name, digest in package['files_sha256'].items():
        assert sha(HERE/name) == digest, 'STAGE1 FILE CHANGED: '+name
    passed = json.loads(INTERFACE_REPORT.read_text())
    assert passed['interface_passed'] and passed['learning_updates'] == 0
    assert passed['runtime_lower_kp'] == [200., 200., 60., 200., 200., 60.]
    assert OLD_CHECKPOINT.is_file(), 'ORIGINAL CONTROL CHECKPOINT MISSING'
    return package


def snapshot_originals():
    verify()
    return {str(OLD_CHECKPOINT): sha(OLD_CHECKPOINT),
            str(ROOT/'legged_lab/envs/__init__.py'): sha(ROOT/'legged_lab/envs/__init__.py')}


def verify_originals(snapshot):
    verify()
    for name, digest in snapshot.items():
        assert sha(name) == digest, 'ORIGINAL FILE CHANGED: '+name


def write_json(path, data):
    # Reports are confined to a new run/evaluation folder; replace only our own report.
    path = Path(path)
    path.resolve().relative_to(EXPERIMENT.resolve())
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def check_runtime(env):
    names = [f'{side}_{part}_joint' for side in ('left', 'right')
             for part in ('hip_pitch', 'knee', 'ankle_pitch')]
    ids = [env.robot.joint_names.index(n) for n in names]
    kp = env.robot.root_physx_view.get_dof_stiffnesses()[0, ids].tolist()
    assert kp == [200., 200., 60., 200., 200., 60.], 'FORMAL GAINS CHANGED'
    assert not env.robot.is_fixed_base and not env.cfg.scene.robot.spawn.rigid_props.disable_gravity
    assert not env.cfg.noise.add_noise and not env.cfg.domain_rand.action_delay.enable
    assert abs(env.max_episode_length*env.step_dt-20.) < env.step_dt/2
    assert env.num_actions == 27
    return kp
