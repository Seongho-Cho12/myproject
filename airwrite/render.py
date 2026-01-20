import numpy as np
import cv2

def render_strokes(strokes, canvas_size: int, margin: int, thickness: int):
    """
    strokes: list[list[(x,y)]]
    returns: grayscale uint8 image (white bg, black ink) or None
    """
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
