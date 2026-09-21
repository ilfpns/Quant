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


def main() -> None:
    print(f"[INFO] pre-processing (shape inference): {INPUT_ONNX} -> {PREPROCESSED_ONNX}")
    quant_pre_process(INPUT_ONNX, PREPROCESSED_ONNX)

    input_name = get_input_name(PREPROCESSED_ONNX)
    print(f"[INFO] model input name: {input_name}")
    print(f"[INFO] calibration images: {CALIB_IMAGE_COUNT} from {CALIB_DIR}")

    reader = YoloCalibrationDataReader(CALIB_DIR, CALIB_IMAGE_COUNT, input_name)

    quantize_static(
        model_input=PREPROCESSED_ONNX,
        model_output=OUTPUT_ONNX,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
    )

    before = Path(PREPROCESSED_ONNX).stat().st_size / (1024 * 1024)
    after = Path(OUTPUT_ONNX).stat().st_size / (1024 * 1024)
    print(f"[OK] {INPUT_ONNX}: {before:.2f} MB -> {OUTPUT_ONNX}: {after:.2f} MB")


if __name__ == "__main__":
    main()
