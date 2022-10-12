import onnxruntime as ort
from transformers import AutoTokenizer


# ===== 載入Onnx =====
ort_sess = ort.InferenceSession("mdeberta-v3-base_epoch16_0.9160.onnx", providers=["CUDAExecutionProvider"])


# ===== Inference =====
tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", truncation_side='left')
outputs = ort_sess.run(
    None,
    dict(
        tokenizer(
            "This is a sample.",
            truncation=True,
            max_length=512,
            return_tensors="np",
            return_token_type_ids=False,
        )
    ),
)
