# __init__.py


from .src.pose_detector import (
    load_models,
    detect_persons,
    estimate_poses,
    LOCAL_SP_DIR,  # Default path for pose model
    LOCAL_DET_DIR)

from .src.process_synced_poses import (
    process_synced_frames,
    get_synced_poses,
    get_synced_poses_from_dir,
    get_synced_poses_from_video)