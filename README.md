# Quant

## 양자화(Quantization) 학습 프로젝트

PyTorch로 모델 경량화(양자화)를 단계적으로 학습/실험하는 프로젝트. 최종 목표는 **NVIDIA Jetson + TensorRT** 배포이며, 지금은 그 전 단계로 PyTorch 자체 양자화 API를 이용해 개념과 효과를 직접 확인하는 중이다.

> 이전에는 웹캠 + YOLOv8 사람 인식 예제였지만, 양자화 학습에 YOLO(Conv 위주 구조)는 적합하지 않아 정리하고 LSTM 기반 기초 예제로 전환했다.

### 목표 하드웨어

최종 배포 타겟은 Jetson이며, 추론 런타임으로 TensorRT를 사용할 예정이다. 지금 코드는 PyTorch 기반 개발/실험용이고, Jetson에서는 모델을 ONNX를 거쳐 TensorRT 엔진(INT8/FP16)으로 변환하는 것이 목표다.

### 설치

```bash
pip install -r requirements.txt
```

### 실행

```bash
python main.py
```

`SimpleLSTM`(LSTM + Linear) 모델을 만들어 양자화 전(FP32)과 dynamic quantization 후(INT8)를 비교해서 출력한다: 레이어 타입 변화, state dict 크기, 크기 감소율, 출력값 최대 오차.

### 구조

```
ptq/
  model.py         # 실험용 모델 정의 (SimpleLSTM)
  quantization.py  # 모델을 받아 양자화된 모델을 반환하는 순수 변환 함수 (dynamic_quantize)
  report.py        # 양자화 전/후 측정 및 콘솔 리포트 출력 (measure, print_comparison)
main.py             # 위 컴포넌트를 묶어 before/after 비교를 실행하는 진입점
```

### 컴포넌트 분리 원칙

- `ptq/model.py`는 모델 구조만 안다. 양자화나 측정 방법은 모른다.
- `ptq/quantization.py`는 "모델을 넣으면 양자화된 모델이 나온다"는 순수 변환만 안다. 어떤 모델인지, 어떻게 측정할지는 모른다.
- `ptq/report.py`는 측정과 출력만 안다. 모델 구조나 양자화 방법은 모른다.
- `main.py`만이 위 컴포넌트들을 묶어 실행 흐름을 구성한다.

새로운 양자화 기법(Static PTQ, QAT)이나 다른 모델을 추가할 때도 해당 역할을 담당하는 파일 하나만 추가/수정하고, `main.py`에 로직을 섞지 않는다.

---

## 양자화 로드맵

**경량화 3가지 방식**: Knowledge Distillation(지식 증류), Pruning(가지치기), Quantization(양자화). 이 프로젝트는 Quantization에 집중한다.

**양자화 대상**: 가중치(weight, 학습 후 고정) / 활성화(activation, 입력에 따라 달라지는 중간 출력값). 활성화 범위는 대표 샘플 데이터로 Calibration(scale, zero-point 추정)해서 정한다.

**적용 시점별 분류**:
- **PTQ (Post Training Quantization)**: 학습이 끝난 모델을 양자화. 하위에 Dynamic(런타임에 weight만 양자화, 구현 쉬움, 주로 CPU/Linear·RNN 계열에 효과적) / Static(weight+activation 모두 calibration data로 고정, 정확도 더 좋음, 사전 calibration 필요) 두 방식이 있다.
- **QAT (Quantization Aware Training)**: 학습 중 fake-quant를 삽입해 양자화 오차를 모델이 학습하도록 함. 정확도가 가장 좋지만 재학습이 필요하다.

**진행 순서**: PTQ (Dynamic) → PTQ (Static) → QAT → Jetson TensorRT INT8/FP16 변환

### 1단계: PTQ (Dynamic) — 완료

`torch.quantization.quantize_dynamic`을 `nn.Linear` + `nn.LSTM` 대상으로 적용해 `SimpleLSTM` 예제로 검증함.

**결과 (`python main.py`)**

```
[Before] FP32 (before)
SimpleLSTM(
  (lstm): LSTM(128, 256, batch_first=True)
  (fc): Linear(in_features=256, out_features=10, bias=True)
)
State dict size: 1.521 MB

[After] INT8 dynamic (after)
SimpleLSTM(
  (lstm): DynamicQuantizedLSTM(128, 256, batch_first=True)
  (fc): DynamicQuantizedLinear(in_features=256, out_features=10, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
)
State dict size: 0.389 MB

Size reduction     : 74.4 %
Max output diff     : 0.002772
```

**알아둘 점**:
- Dynamic Quantization은 `nn.Linear`/`nn.LSTM`(RNN 계열) 위주로 효과가 있다. YOLOv8 같은 Conv 위주 모델에는 대상 레이어가 없어서 효과가 전혀 없었다 (실제로 확인함 — 그래서 이 예제로 교체함).
- 양자화 backend 엔진은 `ptq/quantization.py`의 `default_quantized_engine()`이 `torch.backends.quantized.supported_engines` 중에서 자동으로 고른다 (ARM이면 `qnnpack` 우선, 아니면 `fbgemm` 우선, 둘 다 없으면 지원되는 첫 엔진 — 이 노트북의 Windows CPU 빌드는 `onednn`만 지원해서 자동으로 그걸 쓴다. Jetson에서는 `qnnpack`이 선택될 것).
- Dynamic quantization은 모델을 새 파일로 저장하지 않고 메모리에서만 변환한다. `ptq/report.py`의 크기 비교는 `state_dict`를 임시로 저장해서 비교한 값이다.

이후 단계(Static PTQ, QAT, TensorRT 변환)는 각 단계 진행 시 README에 갱신한다.
