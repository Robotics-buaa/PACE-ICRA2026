"""G1 table-tennis asset with a fixed right-hand paddle."""
from pathlib import Path
from isaaclab_assets import G1_CFG

G1_PADDLE_PARENT_BODY_NAME = "right_palm_link"
G1_PADDLE_BODY_NAME = "g1_paddle_link"
G1_PADDLE_PARENT_OFFSET = (0.20, 0.0, 0.0)
G1_PADDLE_CENTER_OFFSET = (0.0, 0.0, 0.0)
G1_PADDLE_NORMAL = (0.0, 1.0, 0.0)
G1_PADDLE_RADIUS = 0.075
G1_PADDLE_HALF_THICKNESS = 0.005

G1_TT_CFG = G1_CFG.copy()
G1_TT_CFG.spawn.usd_path = str(Path(__file__).resolve().parent / "G1_TT" / "G1_TT.usda")
G1_TT_CFG.init_state.pos = (-1.6, 0.0, 0.74)
G1_TT_CFG.spawn.activate_contact_sensors = True
