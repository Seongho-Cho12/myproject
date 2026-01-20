from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Paths
    model_task_path: str = "hand_landmarker.task"

    # Camera
    cam_index: int = 0
    cam_width: int = 640
    cam_height: int = 480
    cam_buffer_size: int = 1  # backend에 따라 무시될 수 있음

    # Runtime tuning
    open_hold_frames: int = 8
    missing_break_frames: int = 10
    jump_thresh_px: int = 80
    detect_every_n: int = 2

    # Render
    canvas_size: int = 100
    canvas_margin: int = 30
    stroke_thickness: int = 3

    # Result strip
    strip_img_h: int = 100
    strip_gap: int = 10
    strip_pad: int = 20

    # UI
    window_cam: str = "AirWrite (SPACE/ESC)"
    window_result: str = "Result"


CFG = Config()
