# AirWrite TrOCR (손가락 공중필기 → 글자 이미지 → TrOCR 인식) 사용법

이 프로젝트는 웹캠에서 **검지 끝 좌표를 추적**해 공중에 쓴 글자를 **글자 단위 이미지로 렌더링**하고, 각 글자 이미지를 **TrOCR로 인식**한 뒤 결과(예측 문자열 + 사용한 이미지 strip)를 **새 창으로 표시**합니다.

---

## 0. 준비물

- Windows 10/11
- Python 3.10 ~ 3.12
- 웹캠
- `hand_landmarker.task` 모델 파일 (MediaPipe 손 랜드마커)
- 인터넷(최초 1회 모델 다운로드용)
  - MediaPipe 모델 1회 다운로드
  - TrOCR 모델 1회 다운로드(캐시됨)

---

## 1. 프로젝트 파일 구성

아래처럼 두 파일이 같은 폴더에 있어야 합니다.

```

D:\BCS_winter_2026
├─ proto.py
└─ hand_landmarker.task

````

---

## 2. hand_landmarker.task 다운로드 (이미 레포지토리에 존재)

아래 주소에서 파일을 다운로드하고 **파일명을 정확히 `hand_landmarker.task`로 저장**한 다음 `proto.py`와 같은 폴더에 두세요.
이미 레포지토리에 존재하므로 생략하셔도 됩니다.

- 다운로드 링크:
  - https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

---

## 3. 파이썬 패키지 설치

### 3-1) (권장) 가상환경 생성
```bat
cd D:\BCS_winter_2026
python -m venv .venv
.venv\Scripts\activate
````

### 3-2) 필수 라이브러리 설치

```bat
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe pillow numpy transformers torch
```

> 참고
>
> * `opencv-python-headless`가 설치되어 있으면 `cv2.imshow`가 동작하지 않습니다.
>   아래처럼 제거 후 `opencv-python`만 남기세요.
>
> ```bat
> python -m pip uninstall opencv-python-headless -y
> python -m pip install --force-reinstall --no-cache-dir opencv-python
> ```

---

## 4. 최초 실행(모델 다운로드)

TrOCR 모델은 최초 1회 다운로드됩니다(캐시됨).
처음 실행 시 인터넷 연결이 필요할 수 있습니다.

```bat
python proto.py
```

---

## 5. 조작법(핵심)

### 키 입력

* **SPACE**

  * (대기) → **녹화 시작**
  * (녹화 중) → **녹화 종료 + OCR + 결과창 표시**
  * (결과창 표시 중) → **다시 녹화 시작**
* **ESC**

  * 즉시 종료

### 글자 쓰기 규칙

* **검지만 펼친 상태(index-only)** 에서만 “그리기”로 인식됩니다.
* 획이 끊겨야 하는 구간은 아래 중 하나로 만들면 됩니다.

  * 주먹을 쥐거나(검지 아닌 상태)
  * 손이 카메라에서 잠시 사라지게 하기
* **검지 좌표가 10프레임 이상 안 잡히면** 그 지점에서 **획을 자동으로 끊습니다.**
* **글자 하나 끝났을 때**

  * 손가락을 **3개 이상 펼치고** 약 0.3초 정도(연속 8프레임) 유지
  * → “글자 종료”로 인식되어 다음 글자 입력으로 넘어갑니다.

---

## 6. 결과 화면 설명

녹화를 종료하면 `Result` 창이 뜹니다.

* 상단: 최종 예측 문자열(폰트 크기 약 50)
* 중단: 글자별 예측 결과(빈 글자는 □로 표시)
* 하단: 각 글자 인식에 사용된 이미지들을 가로로 나열한 strip

결과 확인 후 **SPACE**를 누르면 다시 녹화 모드로 돌아갑니다.

---

## 7. 성능/끊김 관련 기본 설정(코드 내 포함)

아래 튜닝이 기본으로 적용되어 있습니다.

* 카메라 해상도: 640x480
* 손 추론: 2프레임에 1번 수행(중간 프레임은 이전 결과 재사용)
* 시작 시 버퍼 비우기 + 워밍업 수행
* 포인터 튐 방지(이전 점과 너무 멀면 해당 프레임 점 무시)

---

## 8. 자주 발생하는 문제 해결

### 8-1) `hand_landmarker.task` 관련 에러

* `hand_landmarker.task`가 같은 폴더에 있는지 확인
* 파일명이 `hand_landmarker.task`인지 확인(확장자 `.txt` 붙는 실수 주의)

### 8-2) 창이 안 뜨거나 imshow 에러

* `opencv-python-headless` 설치되어 있으면 발생
* 해결:

```bat
python -m pip uninstall opencv-python-headless -y
python -m pip install --force-reinstall --no-cache-dir opencv-python
```

### 8-3) 첫 시작이 버벅거림

* 첫 시작 시 MediaPipe/모델 워밍업이 있어 잠깐 느릴 수 있음
* 코드는 시작 시 워밍업을 수행하도록 되어 있음

### 8-4) OCR이 이상하게 나옴

* TrOCR은 모델 특성상 손글씨/공중필기 스타일에서 오인식이 날 수 있음
* 결과창 하단의 이미지 strip을 보고,

  * 글자가 너무 작으면 `CANVAS_SIZE`, `STROKE_THICKNESS`를 키우는 방식으로 튜닝합니다.
  * 글자가 너무 굵거나 뭉개지면 thickness를 낮춥니다.

---

## 9. 실행 요약(최단 루트)

1. `hand_landmarker.task` 다운로드해서 `proto.py` 옆에 둠
2. 설치:

```bat
python -m pip install opencv-python mediapipe pillow numpy transformers torch
```

3. 실행:

```bat
python proto.py
```

4. SPACE로 시작 → 글자 쓰기 → 손가락 펼쳐 글자 구분 → SPACE로 결과 확인 → SPACE로 재시작


