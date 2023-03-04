from pathlib import Path

PLATFORM_FEATURE_MAPPINGS = {
    0: "Agent Position",
    1: "Agent Velocity",
    2: "Enemy Position",
    3: "Enemy Velocity",
    4: "Platform Width 1",
    5: "Platform Width 2",
    6: "Gap",
    7: "Platform Identifier",
    8: "Platform Height",
}

CONFIGS_PATH = str(Path(__file__).parent.parent.parent / "configs")
