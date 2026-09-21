# Quant

## 양자화(Quantization) 학습 프로젝트

PyTorch로 모델 경량화(양자화)를 단계적으로 학습하고 실험하는 프로젝트. 최종 목표는 **NVIDIA Jetson + TensorRT** 배포이며, 현재는 그 전 단계로 PyTorch 자체 양자화 API를 이용해 개념과 효과를 직접 확인하고 있다.

> 이전에는 웹캠 + YOLOv8 사람 인식 예제였지만, 양자화 학습에 YOLO(Conv 위주 구조)는 적합하지 않아 정리하고 LSTM 기반 기초 예제로 전환했다.

### 목표 하드웨어

최종 배포 타겟은 Jetson이며, 추론 런타임으로 TensorRT를 사용할 예정이다. 현재 코드는 PyTorch 기반 개발/실험용이며, Jetson에서는 모델을 ONNX를 거쳐 TensorRT 엔진(INT8/FP16)으로 변환하는 것이 목표다.

---

## 양자화 로드맵

**경량화 3가지 방식**: Knowledge Distillation(지식 증류), Pruning(가지치기), Quantization(양자화)

이 프로젝트는 **Quantization**에 집중한다.

**양자화 대상**
- **Weight**: 학습 후 고정되는 가중치
- **Activation**: 입력에 따라 달라지는 중간 출력값

활성화 범위는 대표 샘플 데이터로 Calibration하여 scale, zero-point를 추정한다.

### 적용 시점별 분류

- **PTQ (Post Training Quantization)**  
  학습이 끝난 모델을 양자화한다.
  - **Dynamic**: 런타임에 weight를 양자화하며, 구현이 쉽고 주로 CPU/Linear·RNN 계열에 효과적이다.
  - **Static**: weight + activation을 calibration data를 통해 고정하며, 사전 calibration이 필요하다.

- **QAT (Quantization Aware Training)**  
  학습 중 fake-quant를 삽입해 양자화 오차를 모델이 학습하도록 한다. 정확도가 가장 좋지만 재학습이 필요하다.

---

## 진행 순서

**PTQ (Dynamic) → PTQ (Static) → QAT → Jetson TensorRT INT8/FP16 변환**

### 학습 TIL

학습 TIL은 다음 링크에서 확인할 수 있다.

[TIL 보러가기 🫡](https://github.com/ilfpns/IL/tree/main/Projects/Quant)

---

## 1단계: PTQ (Dynamic) — 완료

`torch.quantization.quantize_dynamic`을 `nn.Linear` + `nn.LSTM` 대상으로 적용해 `SimpleLSTM` 예제로 검증했다.

---

## 2단계: PTQ (Static) — 완료

`torch.quantization.prepare` / `convert`로 weight + activation을 calibration 데이터(임의 정규분포 샘플)로 고정해 `SimpleMLP`에 적용했다.

Dynamic PTQ와 state dict 크기는 비슷했지만, calibration 표본이 실제 데이터 분포와 다르면 MaxDiff가 더 커질 수 있다는 점을 확인했다.

---

## 3단계: QAT (Quantization Aware Training) — 완료

`torch.quantization.prepare_qat`으로 fake-quant를 삽입한 뒤, FP32 모델을 teacher로 삼아 **(임의 입력 → FP32 출력)** 쌍으로 self-distillation 학습을 진행하고 `convert`로 변환했다.  
(`quanti/qat.py`의 `qat_quantize`, `build_self_distillation_data`)

다만 이번 실험 조건에서는 QAT의 MaxDiff가 Static PTQ보다 오히려 크게 측정됐다.

- FP32 teacher가 사전 학습되지 않은 무작위 초기화 모델
- 학습 데이터도 임의 정규분포

의 조건에서는 의미 있는 task 신호가 없어, 학습이 진행될수록 가중치가 fake-quant 노이즈에 맞춰 흔들리며 원래 FP32 출력에서 더 멀어진 것으로 보인다.

**자세한 설명은 TIL에서 확인하는 것을 추천합니다.**
