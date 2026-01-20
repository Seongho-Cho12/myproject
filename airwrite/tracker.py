import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

def _is_finger_extended(lm, finger: str) -> bool:
    if finger == "thumb":
        return False
    tip = lm[TIP_IDS[finger]]
    pip = lm[PIP_IDS[finger]]
    return tip.y < pip.y

def count_extended_fingers(lm) -> int:
    cnt = 0
    for f in ["index", "middle", "ring", "pinky"]:
        if _is_finger_extended(lm, f):
            cnt += 1
    return cnt

def is_open_hand(lm, thresh=3) -> bool:
    return count_extended_fingers(lm) >= thresh

def is_index_only(lm) -> bool:
    if not _is_finger_extended(lm, "index"):
        return False
    for f in ["middle", "ring", "pinky"]:
        if _is_finger_extended(lm, f):
            return False
    return True

def create_landmarker(model_path: str):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)

def warmup(cap: cv2.VideoCapture, landmarker, session_t0: float, n_grab: int = 10, n_warm: int = 5):
    for _ in range(n_grab):
        cap.grab()
    for _ in range(n_warm):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - session_t0) * 1000)
        try:
            landmarker.detect_for_video(mp_image, ts_ms)
        except Exception:
            pass

class IndexTracker:
    """
    - MediaPipe 결과를 받아 글자/획/점 분절만 담당
    - 랜더링/결과표시는 외부에서 처리
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.char_strokes = []        # completed chars: list[strokes]
        self.current_strokes = []     # strokes for current char
        self.current_segment = []     # points for current segment
        self.open_count = 0
        self.missing_tip_frames = 0

    def _commit_segment_if_any(self):
        if len(self.current_segment) >= 2:
            self.current_strokes.append(self.current_segment)
        self.current_segment = []

    def _finalize_char(self):
        self._commit_segment_if_any()
        if len(self.current_strokes) > 0:
            self.char_strokes.append(self.current_strokes)
        self.current_strokes = []

    def process(self, frame_bgr, landmarker, session_t0: float, frame_idx: int, last_result):
        """
        returns: (new_last_result, did_add_point, did_finalize_char)
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - session_t0) * 1000)

        # detect every N frames
        if frame_idx % self.cfg.detect_every_n == 0 or last_result is None:
            last_result = landmarker.detect_for_video(mp_image, ts_ms)
        result = last_result

        did_add_point = False
        did_finalize_char = False

        have_hand = bool(result and result.hand_landmarks and len(result.hand_landmarks) > 0)
        if have_hand:
            lm = result.hand_landmarks[0]

            # open-hand -> finalize char
            if is_open_hand(lm, thresh=3):
                self.open_count += 1
            else:
                self.open_count = 0

            if self.open_count >= self.cfg.open_hold_frames:
                self._finalize_char()
                self.open_count = 0
                self.missing_tip_frames = 0
                did_finalize_char = True

            # index-only -> add point
            if is_index_only(lm):
                tip = lm[TIP_IDS["index"]]
                x = int(tip.x * w)
                y = int(tip.y * h)

                if len(self.current_segment) > 0:
                    px, py = self.current_segment[-1]
                    if (x - px) * (x - px) + (y - py) * (y - py) <= (self.cfg.jump_thresh_px * self.cfg.jump_thresh_px):
                        self.current_segment.append((x, y))
                        self.missing_tip_frames = 0
                        did_add_point = True
                    else:
                        self.missing_tip_frames += 1
                else:
                    self.current_segment.append((x, y))
                    self.missing_tip_frames = 0
                    did_add_point = True
            else:
                self.missing_tip_frames += 1
        else:
            self.missing_tip_frames += 1

        # missing -> break stroke
        if self.missing_tip_frames >= self.cfg.missing_break_frames:
            self._commit_segment_if_any()
            self.missing_tip_frames = 0

        return last_result, did_add_point, did_finalize_char

    def finish_session(self):
        """
        녹화 종료 시 호출: 마지막 글자 확정하고, 최종 char_strokes 반환
        """
        self._finalize_char()
        return self.char_strokes
