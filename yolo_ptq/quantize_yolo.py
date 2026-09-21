"""YOLOv8n(ONNX)을 ONNX Runtime static PTQ(INT8)로 양자화한다.

전제: yolov8n_sim.onnx (onnxsim으로 정리된 그래프)가 이미 존재.
Calibration 데이터: coco128 학습 이미지 중 일부를 실제 이미지 분포로 사용.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

INPUT_ONNX = "yolov8n_sim.onnx"
PREPROCESSED_ONNX = "yolov8n_preprocessed.onnx"
OUTPUT_ONNX = "yolov8n_int8.onnx"
CALIB_DIR = Path("datasets/coco128/images/train2017")
CALIB_IMAGE_COUNT = 50
IMG_SIZE = 640


def preprocess(image_path: Path) -> np.ndarray:
    """YOLO 입력 전처리: BGR->RGB, letterbox 없이 단순 리사이즈, [0,1] 정규화, NCHW."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img, axis=0)  # -> NCHW


class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir: Path, count: int, input_name: str) -> None:
        self.image_paths = sorted(image_dir.glob("*.jpg"))[:count]
        self.input_name = input_name
        self._iter = iter(self.image_paths)

    def get_next(self) -> dict | None:
        path = next(self._iter, None)
        if path is None:
            return None
        return {self.input_name: preprocess(path)}


def get_input_name(onnx_path: str) -> str:
    import onnx

    model = onnx.load(onnx_path)
    return model.graph.input[0].name


def get_detect_head_node_names(onnx_path: str, head_prefix: str = "/model.22") -> list[str]:
    """검출 헤드(Detect, YOLOv8n 기준 model.22) 노드 이름을 모두 찾는다.

    검출 헤드의 마지막 conv들(특히 클래스 점수 브랜치, cv3.*)은 양자화에 매우 민감해서
    naive INT8 PTQ를 걸면 클래스 점수가 통째로 붕괴하는 경우가 있다(직접 확인함:
    양자화 전 top score 0.7996 -> 양자화 후 0.0, mAP도 0으로 붕괴). 그래서 헤드는
    양자화 대상에서 제외하고 backbone/neck만 INT8로 양자화한다.
    """
    import onnx

    model = onnx.load(onnx_path)
    return [n.name for n in model.graph.node if n.name.startswith(head_prefix)]


def main() -> None:
    print(f"[INFO] pre-processing (shape inference): {INPUT_ONNX} -> {PREPROCESSED_ONNX}")
    quant_pre_process(INPUT_ONNX, PREPROCESSED_ONNX)

    input_name = get_input_name(PREPROCESSED_ONNX)
    exclude_nodes = get_detect_head_node_names(PREPROCESSED_ONNX)
    print(f"[INFO] model input name: {input_name}")
    print(f"[INFO] calibration images: {CALIB_IMAGE_COUNT} from {CALIB_DIR}")
    print(f"[INFO] excluding {len(exclude_nodes)} Detect head nodes from quantization")

    reader = YoloCalibrationDataReader(CALIB_DIR, CALIB_IMAGE_COUNT, input_name)

    quantize_static(
        model_input=PREPROCESSED_ONNX,
        model_output=OUTPUT_ONNX,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        nodes_to_exclude=exclude_nodes,
    )

    before = Path(PREPROCESSED_ONNX).stat().st_size / (1024 * 1024)
    after = Path(OUTPUT_ONNX).stat().st_size / (1024 * 1024)
    print(f"[OK] {INPUT_ONNX}: {before:.2f} MB -> {OUTPUT_ONNX}: {after:.2f} MB")


if __name__ == "__main__":
    main()
