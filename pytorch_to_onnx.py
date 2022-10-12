import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ===== 定義模型 =====
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


# ===== 載入模型 =====
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


# ===== 定義模型輸入 =====
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


# ===== 轉Onnx (輸入為dict版本) =====
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


# ===== 轉Onnx (輸入為tuple版本) =====
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
