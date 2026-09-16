# Quant

## 양자화(Quantization) 학습 프로젝트

PyTorch로 모델 경량화(양자화)를 단계적으로 학습/실험하는 프로젝트. 최종 목표는 **NVIDIA Jetson + TensorRT** 배포이며, 지금은 그 전 단계로 PyTorch 자체 양자화 API를 이용해 개념과 효과를 직접 확인하는 중이다.

> 이전에는 웹캠 + YOLOv8 사람 인식 예제였지만, 양자화 학습에 YOLO(Conv 위주 구조)는 적합하지 않아 정리하고 LSTM 기반 기초 예제로 전환했다.

### 목표 하드웨어

최종 배포 타겟은 Jetson이며, 추론 런타임으로 TensorRT를 사용할 예정이다. 지금 코드는 PyTorch 기반 개발/실험용이고, Jetson에서는 모델을 ONNX를 거쳐 TensorRT 엔진(INT8/FP16)으로 변환하는 것이 목표다.

---

## 양자화 로드맵

**경량화 3가지 방식**: Knowledge Distillation(지식 증류), Pruning(가지치기), Quantization(양자화). 이 프로젝트는 Quantization에 집중한다.

**양자화 대상**: 가중치(weight, 학습 후 고정) / 활성화(activation, 입력에 따라 달라지는 중간 출력값). 활성화 범위는 대표 샘플 데이터로 Calibration(scale, zero-point 추정)해서 정한다.

**적용 시점별 분류**:
- **PTQ (Post Training Quantization)**: 학습이 끝난 모델을 양자화. 하위에 Dynamic(런타임에 weight만 양자화, 구현 쉬움, 주로 CPU/Linear·RNN 계열에 효과적) / Static(weight+activation 모두 calibration data로 고정, 정확도 더 좋음, 사전 calibration 필요) 두 방식이 있다.
- **QAT (Quantization Aware Training)**: 학습 중 fake-quant를 삽입해 양자화 오차를 모델이 학습하도록 함. 정확도가 가장 좋지만 재학습이 필요하다.

**진행 순서**: PTQ (Dynamic) → PTQ (Static) → QAT → Jetson TensorRT INT8/FP16 변환
=======


### 학습 TIL 
학습 TIL은 다음 링크에서 확인 가능하다. <br>
[TIL 보러가기 🫡](https://github.com/ilfpns/IL/tree/main/Projects/Quant)

### 1단계: PTQ (Dynamic) — 완료

`torch.quantization.quantize_dynamic`을 `nn.Linear` + `nn.LSTM` 대상으로 적용해 `SimpleLSTM` 예제로 검증함.
