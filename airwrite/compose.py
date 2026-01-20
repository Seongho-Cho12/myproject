import numpy as np
import cv2

def compose_strip_image(char_imgs, target_h: int, gap: int, pad: int):
    """
    입력: grayscale 이미지 리스트 (uint8)
    출력: BGR 이미지. 모든 글자 높이를 target_h로 맞춰 가로로 이어붙임.
    """
    if not char_imgs:
        out = np.full((300, 800, 3), 255, dtype=np.uint8)
        cv2.putText(out, "No chars captured", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        return out

    resized = []
    for g in char_imgs:
        if g is None:
            continue
        h, w = g.shape[:2]
        if h <= 0 or w <= 0:
            continue
        scale = target_h / float(h)
        new_w = max(1, int(w * scale))
        rg = cv2.resize(g, (new_w, target_h), interpolation=cv2.INTER_AREA)
        resized.append(rg)

    if not resized:
        out = np.full((300, 800, 3), 255, dtype=np.uint8)
        cv2.putText(out, "No valid images", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        return out

    strip_w = sum(im.shape[1] for im in resized) + gap * (len(resized) - 1)
    strip_h = target_h

    out_h = strip_h + pad * 2 + 50
    out_w = strip_w + pad * 2
    out = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

    cv2.putText(out, f"Captured chars: {len(resized)}  (SPACE=restart, ESC=quit)",
                (pad, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    x = pad
    y = pad + 50
    for im in resized:
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        w = bgr.shape[1]
        out[y:y+strip_h, x:x+w] = bgr
        x += w + gap

    return out
