# Convert Pytorch to ONNX
## 環境設置
| 套件 | Version |
| -------- | :--------: |
| pytorch | 1.12.1 |
| transformers | 4.23.1 |
| onnxruntime-gpu | 1.12.1 |
## 範例程式碼
+ 導入套件
```python
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
```
+ 定義模型
```python
class CustomModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        classifier_dropout,
        num_labels,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        self.fc_dropout = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
    ):
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs[0]
        output = self.fc(self.fc_dropout(last_hidden_state))
        return output
```
+ 載入模型
    + 模型要切換成 **.eval()**
```python
pretrained_model_name_or_path = "microsoft/mdeberta-v3-base"
classifier_dropout = 0.2
num_labels = 4
ckpt_path = "mdeberta-v3-base_epoch16_0.9160.ckpt"
model = CustomModel(
    pretrained_model_name_or_path,
    classifier_dropout,
    num_labels,
)
model.load_state_dict(torch.load(ckpt_path))
model.eval()
```
+ 定義模型輸入(**dict**或**tuple**)
    + **tokenizer(return_tensors="pt")** 回傳的dtype是 **torch.int64**，但**tokenizer(return_tensors="np")** 回傳的dtype是 **np.int32**，兩個儲存的位元數不一致，會導致錯誤，因此要轉換成相同儲存類型
```python
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, truncation_side="left")
dummy_input = {
    k: v.to(torch.int32)
    for k, v in tokenizer(
        "This is a sample.",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).items()
}
```
+ 轉ONNX
    + 輸入為**dict**版本
```python
torch.onnx.export(
    model,  # model being run
    dummy_input,  # model input (or a tuple for multiple inputs)
    "mdeberta-v3-base_epoch16_0.9160.onnx",  # where to save the model
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=13,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input_ids", "token_type_ids", "attention_mask"],  # the model"s input names
    output_names=["logits"],  # the model"s output names
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "token_type_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},  # variable length axes
    },
)
```
+ 轉ONNX
    + 輸入為**tuple**版本
```python
torch.onnx.export(
    model,
    tuple(dummy_input.values()),  # tuple形式
    "mdeberta-v3-base_epoch16_0.9160.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input_ids", "token_type_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "token_type_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    },
)
```
# Inference by ONNX
+ 導入套件
```python
import onnxruntime as ort
from transformers import AutoTokenizer
```
+  載入ONNX模型
```python
ort_sess = ort.InferenceSession("mdeberta-v3-base_epoch16_0.9160.onnx", providers=["CUDAExecutionProvider"]) 
```
+ Inference
    + 輸入參數須為 **dict**，key對應 **input_names**
        + input_names=["input_ids", "token_type_ids", "attention_mask"]
    + 這邊因為是 **deberta** 模型比較特殊，才會有 **return_token_type_ids=False**
```python
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
```
