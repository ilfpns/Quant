# Quant

## 사람 인식 (YOLOv8 + 웹캠)

로컬 웹캠으로 실시간 사람 인식을 수행하는 프로젝트. 현재 코드(OpenCV 캡처, GStreamer/DirectShow 백엔드, YOLOv8 추론)는 바이브코딩으로 빠르게 구현한 기본 버전이다.

### 목표 하드웨어

최종 배포 타겟은 **NVIDIA Jetson**이며, 추론 런타임으로 **TensorRT**를 사용할 예정이다. 지금 노트북에서 돌아가는 PyTorch(ultralytics) 기반 코드는 개발/실험용이고, Jetson에서는 모델을 ONNX를 거쳐 TensorRT 엔진으로 변환해 INT8/FP16 추론으로 전환하는 것이 목표다.

### 설치

```bash
pip install -r requirements.txt
```

처음 실행 시 `yolov8n.pt` 가중치가 자동으로 다운로드된다.

### 실행

```bash
python main.py
```

옵션:

- `--camera-index` (기본값 `0`)
- `--model` (기본값 `yolov8n.pt`)
- `--confidence` (기본값 `0.5`)
- `--width` / `--height` (기본값 `1280x720`)
- `--no-cpu` (CPU 사용률 측정 끄기)

`q` 종료, `c`는 지금까지 누적된 벤치마크 리포트(FPS, 평균 추론 지연시간, 모델 크기, 평균 CPU 사용률)를 콘솔에 출력한다.

### 벤치마크 전용 실행

```bash
python computer_info.py
```

카메라 + 디텍터를 정해진 시간(기본 15초, `--duration`)만큼 돌리고 FPS, 프레임당 평균 추론 지연시간(ms), 모델 파일 크기(MB), 평균 CPU 사용률(`psutil`, `--no-cpu`로 끄기 가능)을 리포트로 출력한다.

추가 옵션:

- `--quantize` — PyTorch dynamic quantization을 적용한 모델로 측정
- `--label` — 결과를 구분할 라벨 (생략 시 `--quantize` 여부로 `fp32_baseline` / `ptq_dynamic` 자동 지정)
- `--log-file` — 결과를 누적 기록할 txt 파일 경로 (기본 `benchmark_log.txt`)

실행할 때마다 `benchmark_log.txt`에 한 줄씩(탭 구분) 누적되므로, 여러 번 실행해서 파일을 열어보면 raw 모델과 양자화 모델의 성능을 나란히 비교할 수 있다:

```bash
python computer_info.py --label fp32_baseline
python computer_info.py --quantize
```

주의: `model_size_mb`는 디스크에 있는 `--model` 파일의 크기를 그대로 읽는 값이라, dynamic quantization은 모델을 새 파일로 저장하지 않고 메모리에서만 변환하므로 두 행의 모델 크기는 동일하게 나온다. 양자화된 모델의 실제 크기 차이를 보려면 양자화 모델을 별도 파일로 저장하는 단계가 추가로 필요하다.

**베이스라인 (양자화 전, YOLOv8n, CPU)**

```
===== Benchmark Report =====
Frames measured     : 19
Duration            : 5.68 s
FPS                 : 3.34
Avg inference time  : 241.92 ms
Model size          : 6.25 MB
Avg CPU usage       : 78.7 %
=============================
```

### 구조

```
person_detector/
  camera.py       # 웹캠 캡처 (Camera)
  detector.py     # YOLOv8 추론, "person" 클래스만 필터링 (PersonDetector, Detection)
  visualizer.py   # 프레임에 박스/라벨 그리기 (draw_detections)
  metrics.py      # FPS / 지연시간 / 모델 크기 / CPU 측정, txt 로그 기록 (PerformanceMonitor, print_report, append_report_to_log)
  quantization.py # 모델을 받아 양자화된 모델을 반환하는 순수 변환 함수 (dynamic_quantize)
  app.py          # camera + detector + visualizer + metrics를 묶어 실행 루프 구성 (PersonDetectionApp)
main.py           # 실시간 뷰어 CLI 진입점
computer_info.py  # 헤드리스 벤치마크 CLI 진입점
```

### 컴포넌트 분리 원칙

각 컴포넌트가 서로 몰라도 되도록 책임을 분리해서 유지보수와 교체가 쉽게 유지한다:

- `Camera`는 영상 소스를 열고/읽고/닫는 것만 안다. 인식이나 렌더링은 모른다.
- `PersonDetector`는 프레임을 `Detection` 리스트로 바꾸는 것만 안다. 카메라나 화면 출력은 모른다.
- `draw_detections`는 `Detection`을 프레임에 그리는 것만 안다. 부작용 없이 새 프레임을 반환한다.
- `PerformanceMonitor` / `print_report`는 측정과 리포트 출력만 안다. 카메라나 모델 내부를 모른다.
- `PersonDetectionApp`만이 위 컴포넌트들을 묶어 실행 루프를 구성한다.

기능을 추가할 때(다중 카메라, 다른 모델, 트래킹, 영상 저장 등) `app.py`나 `main.py`에 로직을 섞지 말고, 해당 역할을 담당하는 컴포넌트 하나만 추가/수정한다.

---

## 양자화(Quantization) 로드맵

Jetson + TensorRT 전환을 위해 모델 경량화를 단계적으로 학습/적용한다.

**경량화 3가지 방식**: Knowledge Distillation(지식 증류), Pruning(가지치기), Quantization(양자화). 이 프로젝트는 Quantization에 집중한다.

**양자화 대상**: 가중치(weight, 학습 후 고정) / 활성화(activation, 입력에 따라 달라지는 중간 출력값). 활성화 범위는 대표 샘플 데이터로 Calibration(scale, zero-point 추정)해서 정한다.

**적용 시점별 분류**:
- **PTQ (Post Training Quantization)**: 학습이 끝난 모델을 양자화. 하위에 Dynamic(런타임에 weight만 양자화, 구현 쉬움, 주로 CPU/Linear 계열에 효과적) / Static(weight+activation 모두 calibration data로 고정, 정확도 더 좋음, 사전 calibration 필요) 두 방식이 있다.
- **QAT (Quantization Aware Training)**: 학습 중 fake-quant를 삽입해 양자화 오차를 모델이 학습하도록 함. 정확도가 가장 좋지만 재학습이 필요하다.

**진행 순서**: PTQ (Dynamic) → PTQ (Static) → QAT → Jetson TensorRT INT8/FP16 변환

### 1단계: PTQ (Dynamic) — 진행 중

PyTorch 내장 `torch.quantization.quantize_dynamic`을 사용해 실험한다. 참고로 미리 알아둘 점:

- 별도 pip 패키지 설치는 필요 없다 (PyTorch에 내장). `torch`/`ultralytics` 버전만 최신인지 확인.
- Dynamic Quantization은 `nn.Linear`(및 RNN/LSTM) 위주로 효과가 있고, YOLOv8처럼 `Conv2d`가 대부분인 구조에는 속도 개선 효과가 제한적이다. 실험 설계 시 이 점을 감안해야 한다.
- ultralytics의 `YOLO` 객체에서 실제 `nn.Module`은 `model.model`로 꺼낼 수 있다.
- 양자화 backend 엔진은 `person_detector/quantization.py`의 `default_quantized_engine()`이 `torch.backends.quantized.supported_engines` 중에서 자동으로 고른다 (ARM이면 `qnnpack` 우선, 아니면 `fbgemm` 우선, 둘 다 없으면 지원되는 첫 엔진 — 이 노트북의 Windows CPU 빌드는 `onednn`만 지원해서 자동으로 그걸 쓴다).
- `PersonDetector(..., quantize=True)`로 dynamic quantization 적용 여부를 켤 수 있고, `main.py` / `computer_info.py` 둘 다 `--quantize` 플래그로 노출돼 있다.
- `computer_info.py --quantize`로 측정한 결과는 `benchmark_log.txt`에 fp32 결과와 함께 누적 기록되어 비교 가능하다 (위 "벤치마크 전용 실행" 참고). 단, dynamic quantization은 모델을 새 파일로 저장하지 않으므로 `model_size_mb`는 원본 파일 크기와 동일하게 나온다.

이후 단계(Static PTQ, QAT, TensorRT 변환)는 각 단계 진행 시 README에 갱신한다.
