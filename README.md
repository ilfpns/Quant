# Quant

## 사람 인식

로컬 웹캠으로 실시간 사람 인식을 수행하는 프로젝트. 현재 코드(OpenCV 캡처, GStreamer/DirectShow 백엔드, YOLOv8 추론)는 바이브코딩으로 빠르게 구현한 기본 버전이다.

### 목표 하드웨어

최종 배포 타겟은 **NVIDIA Jetson**이며, 추론 런타임으로 **TensorRT**를 사용할 예정이다. 지금 노트북에서 돌아가는 PyTorch(ultralytics) 기반 코드는 개발/실험용이고, Jetson에서는 모델을 ONNX를 거쳐 TensorRT 엔진으로 변환해 INT8/FP16 추론으로 전환하는 것이 목표다.

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

### 학습 TIL 
학습 TIL은 다음 링크에서 확인 가능하다. <br>
[TIL 보러가기 🫡](https://github.com/ilfpns/IL/tree/main/Projects/Quant)

### 1단계: PTQ (Dynamic) — 진행 중

PyTorch 내장 `torch.quantization.quantize_dynamic`을 사용해 실험한다. 참고로 미리 알아둘 점:

- 별도 pip 패키지 설치는 필요 없다 (PyTorch에 내장). `torch`/`ultralytics` 버전만 최신인지 확인.
- Dynamic Quantization은 `nn.Linear`(및 RNN/LSTM) 위주로 효과가 있고, YOLOv8처럼 `Conv2d`가 대부분인 구조에는 속도 개선 효과가 제한적이다. 실험 설계 시 이 점을 감안해야 한다.
- ultralytics의 `YOLO` 객체에서 실제 `nn.Module`은 `model.model`로 꺼낼 수 있다.
- 양자화 backend 엔진 설정이 필요: 노트북(x86)은 `fbgemm`, 추후 Jetson(ARM)에서는 `qnnpack`을 써야 한다 (`torch.backends.quantized.engine`).
- 양자화 전/후 비교를 위해 `computer_info.py`를 확장하거나, fp32 vs quantized를 나란히 비교하는 스크립트가 필요하다 (정확도 확인용 샘플 이미지도 몇 장 준비).

이후 단계(Static PTQ, QAT, TensorRT 변환)는 각 단계 진행 시 README에 갱신한다.
