import time
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# ---------------------------
# Config
# ---------------------------
MODEL_TASK_PATH = "hand_landmarker.task"

# TrOCR (Korean). 필요시 다른 모델로 교체 가능.
TROCR_MODEL_NAME = "ddobokki/ko-trocr"

# Runtime tuning
OPEN_HOLD_FRAMES = 8
MISSING_BREAK_FRAMES = 10
JUMP_THRESH_PX = 80
DETECT_EVERY_N = 2

CANVAS_SIZE = 640
CANVAS_MARGIN = 30
STROKE_THICKNESS = 11

TEXT_FONT_SIZE = 50
IMG_TARGET_H = 60  # 글자 높이와 비슷하게

# ---------------------------
# Finger state helpers
# ---------------------------
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

# ---------------------------
# MediaPipe landmarker factory
# ---------------------------
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

# ---------------------------
# Render strokes -> grayscale image
# ---------------------------
def render_strokes(strokes, canvas_size=CANVAS_SIZE, margin=CANVAS_MARGIN, thickness=STROKE_THICKNESS):
    all_pts = [p for seg in strokes for p in seg]
    if len(all_pts) < 2:
        return None

    pts = np.array(all_pts, dtype=np.float32)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = max_xy - min_xy
    span[span == 0] = 1.0

    usable = canvas_size - 2 * margin
    scale = float(min(usable / span[0], usable / span[1]))

    img = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

    for seg in strokes:
        if len(seg) < 2:
            continue
        seg_pts = ((np.array(seg, dtype=np.float32) - min_xy) * scale + margin).astype(np.int32)
        for i in range(1, len(seg_pts)):
            cv2.line(img, tuple(seg_pts[i - 1]), tuple(seg_pts[i]), 0, thickness, cv2.LINE_AA)

    return img

# ---------------------------
# TrOCR OCR
# ---------------------------
def load_trocr(model_name: str):
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()
    return processor, model

@torch.inference_mode()
def trocr_single_gray(gray_img, processor, model) -> str:
    # gray_img uint8 (H,W) -> RGB PIL
    pil = Image.fromarray(gray_img).convert("RGB")
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, max_new_tokens=16)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

# ---------------------------
# Result view composer
# ---------------------------
def _try_load_font(size: int):
    # Windows에서 대체로 되는 폰트들 순서대로 시도
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",      # 맑은 고딕
        r"C:\Windows\Fonts\Malgun.ttf",
        r"C:\Windows\Fonts\gulim.ttc",       # 굴림
        r"C:\Windows\Fonts\batang.ttc",      # 바탕
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()

def compose_result_image(char_imgs, pred_chars, final_text, font_size=TEXT_FONT_SIZE, img_target_h=IMG_TARGET_H):
    """
    - 상단: 예측 문자열(폰트 크기 50)
    - 하단: 각 글자 이미지(높이를 글자 높이와 비슷하게) 가로로 나열
    """
    font = _try_load_font(font_size)

    # PIL로 텍스트 렌더링 준비
    pad = 20
    line_gap = 12

    # 각 글자 이미지 리사이즈(높이를 img_target_h로 맞춤)
    resized = []
    for g in char_imgs:
        if g is None:
            continue
        h, w = g.shape[:2]
        if h <= 0 or w <= 0:
            continue
        scale = img_target_h / float(h)
        nw = max(1, int(w * scale))
        rg = cv2.resize(g, (nw, img_target_h), interpolation=cv2.INTER_AREA)
        resized.append(rg)

    # 하단 strip 생성
    if len(resized) == 0:
        strip_h = img_target_h
        strip_w = 300
        strip = np.full((strip_h, strip_w), 255, dtype=np.uint8)
    else:
        gap = 10
        strip_h = img_target_h
        strip_w = sum(im.shape[1] for im in resized) + gap * (len(resized) - 1)
        strip = np.full((strip_h, strip_w), 255, dtype=np.uint8)
        x = 0
        for im in resized:
            w = im.shape[1]
            strip[:, x:x+w] = im
            x += w + gap

    # 텍스트 영역 크기 계산 (대략)
    txt = f"{final_text}"
    # dummy image to measure
    tmp = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), txt, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 예측 문자 라벨(선택): pred_chars를 작은 글씨로도 보여줌
    small_font = _try_load_font(max(18, font_size // 2))
    pred_line = " ".join([c if c else "□" for c in pred_chars])
    bbox2 = draw.textbbox((0, 0), pred_line, font=small_font)
    pred_h = bbox2[3] - bbox2[1]

    width = max(text_w, strip_w) + pad * 2
    height = pad + text_h + line_gap + pred_h + line_gap + strip_h + pad

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 상단 텍스트
    draw.text((pad, pad), txt, font=font, fill=(0, 0, 0))
    # 예측 글자별
    draw.text((pad, pad + text_h + line_gap), pred_line, font=small_font, fill=(0, 0, 0))

    # 하단 이미지 strip 붙이기
    strip_rgb = cv2.cvtColor(strip, cv2.COLOR_GRAY2RGB)
    strip_pil = Image.fromarray(strip_rgb)
    canvas.paste(strip_pil, (pad, pad + text_h + line_gap + pred_h + line_gap))

    # OpenCV용 BGR로 변환
    out = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    return out

# ---------------------------
# Main
# ---------------------------
def main():
    # torch CPU 속도 튜닝(환경에 따라)
    try:
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    except Exception:
        pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    # 끊김 완화: 해상도 낮추기
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    processor, model = load_trocr(TROCR_MODEL_NAME)

    recording = False
    showing_result = False

    landmarker = None
    session_t0 = None
    frame_idx = 0
    last_result = None

    # buffers
    char_imgs = []
    pred_chars = []
    final_text = ""
    result_bgr = None

    # per-character strokes
    current_strokes = []
    current_segment = []
    open_count = 0
    missing_tip_frames = 0

    def reset_session_buffers():
        nonlocal char_imgs, pred_chars, final_text, result_bgr
        nonlocal current_strokes, current_segment, open_count, missing_tip_frames
        nonlocal frame_idx, last_result
        char_imgs = []
        pred_chars = []
        final_text = ""
        result_bgr = None
        current_strokes = []
        current_segment = []
        open_count = 0
        missing_tip_frames = 0
        frame_idx = 0
        last_result = None

    def commit_segment_if_any():
        nonlocal current_strokes, current_segment
        if len(current_segment) >= 2:
            current_strokes.append(current_segment)
        current_segment = []

    def finalize_char():
        nonlocal current_strokes, current_segment, char_imgs
        commit_segment_if_any()
        img = render_strokes(current_strokes)
        current_strokes = []
        if img is not None:
            char_imgs.append(img)

    def warmup():
        # 카메라 버퍼 비우기
        for _ in range(10):
            cap.grab()
        # landmarker 워밍업
        for _ in range(5):
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

    def run_trocr_and_build_result():
        nonlocal pred_chars, final_text, result_bgr
        pred_chars = []
        for g in char_imgs:
            t = trocr_single_gray(g, processor, model)
            pred_chars.append(t)
        final_text = "".join(pred_chars)
        result_bgr = compose_result_image(char_imgs, pred_chars, final_text)

    print("조작: SPACE=녹화 시작/종료/결과 확인 후 재시작, ESC=종료")

    while True:
        if recording:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # UI
            cv2.putText(frame, "MODE: RECORDING", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
            cv2.putText(frame, f"Chars: {len(char_imgs)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - session_t0) * 1000)

            # detect every N frames (reuse last)
            if frame_idx % DETECT_EVERY_N == 0 or last_result is None:
                last_result = landmarker.detect_for_video(mp_image, ts_ms)
            result = last_result
            frame_idx += 1

            have_hand = bool(result and result.hand_landmarks and len(result.hand_landmarks) > 0)
            if have_hand:
                lm = result.hand_landmarks[0]

                # open-hand => finalize char
                if is_open_hand(lm, thresh=3):
                    open_count += 1
                else:
                    open_count = 0

                if open_count >= OPEN_HOLD_FRAMES:
                    finalize_char()
                    open_count = 0
                    missing_tip_frames = 0

                # index-only => add point
                if is_index_only(lm):
                    tip = lm[TIP_IDS["index"]]
                    x = int(tip.x * w)
                    y = int(tip.y * h)

                    # jump suppression
                    if len(current_segment) > 0:
                        px, py = current_segment[-1]
                        if (x - px) * (x - px) + (y - py) * (y - py) <= (JUMP_THRESH_PX * JUMP_THRESH_PX):
                            current_segment.append((x, y))
                            missing_tip_frames = 0
                        else:
                            missing_tip_frames += 1
                    else:
                        current_segment.append((x, y))
                        missing_tip_frames = 0

                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                else:
                    missing_tip_frames += 1
            else:
                missing_tip_frames += 1

            # missing => break stroke
            if missing_tip_frames >= MISSING_BREAK_FRAMES:
                commit_segment_if_any()
                missing_tip_frames = 0

            # show camera
            cv2.imshow("AirWrite (SPACE/ESC)", frame)

        elif showing_result:
            # 결과창만 표시
            if result_bgr is None:
                # 안전장치
                blank = np.full((400, 800, 3), 255, dtype=np.uint8)
                cv2.putText(blank, "No result", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
                cv2.imshow("Result", blank)
            else:
                cv2.imshow("Result", result_bgr)
            # 카메라창은 닫아도 됨
            try:
                cv2.destroyWindow("AirWrite (SPACE/ESC)")
            except Exception:
                pass
        else:
            # 대기 상태(시작 전): 카메라 한 프레임 띄우기
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "MODE: IDLE (SPACE to start)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
            cv2.imshow("AirWrite (SPACE/ESC)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        if key == 32:  # SPACE
            if recording:
                # STOP -> OCR -> show result
                recording = False

                finalize_char()  # 마지막 글자 확정
                run_trocr_and_build_result()

                showing_result = True

                if landmarker is not None:
                    landmarker.close()
                    landmarker = None

            elif showing_result:
                # RESULT -> START recording again
                try:
                    cv2.destroyWindow("Result")
                except Exception:
                    pass

                reset_session_buffers()
                landmarker = create_landmarker(MODEL_TASK_PATH)
                session_t0 = time.time()
                warmup()

                recording = True
                showing_result = False

            else:
                # IDLE -> START recording
                reset_session_buffers()
                landmarker = create_landmarker(MODEL_TASK_PATH)
                session_t0 = time.time()
                warmup()

                recording = True
                showing_result = False

    # cleanup
    try:
        cv2.destroyWindow("Result")
    except Exception:
        pass
    try:
        cv2.destroyWindow("AirWrite (SPACE/ESC)")
    except Exception:
        pass

    if landmarker is not None:
        landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
