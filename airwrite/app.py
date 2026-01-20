import cv2
import numpy as np
import time

from .config import CFG
from .tracker import create_landmarker, warmup, IndexTracker
from .render import render_strokes
from .compose import compose_strip_image

def run_app():
    cap = cv2.VideoCapture(CFG.cam_index)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.cam_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CFG.cam_buffer_size)

    recording = False
    showing_result = False

    landmarker = None
    session_t0 = None
    frame_idx = 0
    last_result = None

    tracker = IndexTracker(CFG)

    char_imgs = []
    result_bgr = None

    print("조작: SPACE=녹화 시작/종료/결과 확인 후 재시작, ESC=종료")

    while True:
        if recording:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            cv2.putText(frame, "MODE: RECORDING", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
            cv2.putText(frame, f"Chars: {len(tracker.char_strokes)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            last_result, did_add, _ = tracker.process(
                frame_bgr=frame,
                landmarker=landmarker,
                session_t0=session_t0,
                frame_idx=frame_idx,
                last_result=last_result,
            )
            frame_idx += 1

            # 시각화: point 추가했으면 마지막 점 표시
            if did_add:
                seg = tracker.current_segment
                if len(seg) > 0:
                    x, y = seg[-1]
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            cv2.imshow(CFG.window_cam, frame)

        elif showing_result:
            if result_bgr is None:
                result_bgr = compose_strip_image(char_imgs, CFG.strip_img_h, CFG.strip_gap, CFG.strip_pad)
            cv2.imshow(CFG.window_result, result_bgr)
            try:
                cv2.destroyWindow(CFG.window_cam)
            except Exception:
                pass

        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "MODE: IDLE (SPACE to start)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
            cv2.imshow(CFG.window_cam, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        if key == 32:  # SPACE
            if recording:
                # STOP -> render -> show result
                recording = False

                char_strokes = tracker.finish_session()
                char_imgs = []
                for strokes in char_strokes:
                    img = render_strokes(strokes, CFG.canvas_size, CFG.canvas_margin, CFG.stroke_thickness)
                    if img is not None:
                        char_imgs.append(img)

                result_bgr = compose_strip_image(char_imgs, CFG.strip_img_h, CFG.strip_gap, CFG.strip_pad)
                showing_result = True

                if landmarker is not None:
                    landmarker.close()
                    landmarker = None

            elif showing_result:
                # RESULT -> START
                try:
                    cv2.destroyWindow(CFG.window_result)
                except Exception:
                    pass

                tracker.reset()
                char_imgs = []
                result_bgr = None

                landmarker = create_landmarker(CFG.model_task_path)
                session_t0 = time.time()
                frame_idx = 0
                last_result = None
                warmup(cap, landmarker, session_t0)

                recording = True
                showing_result = False

            else:
                # IDLE -> START
                tracker.reset()
                char_imgs = []
                result_bgr = None

                landmarker = create_landmarker(CFG.model_task_path)
                session_t0 = time.time()
                frame_idx = 0
                last_result = None
                warmup(cap, landmarker, session_t0)

                recording = True
                showing_result = False

    # cleanup
    try:
        cv2.destroyWindow(CFG.window_result)
    except Exception:
        pass
    try:
        cv2.destroyWindow(CFG.window_cam)
    except Exception:
        pass

    if landmarker is not None:
        landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
