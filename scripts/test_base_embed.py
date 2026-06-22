import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

base = "/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B"
items = json.load(open("data/vqav2_pool_v2.json"))[:1]
img = Image.open(items[0]["image"]).convert("RGB")
q = items[0]["question"]

proc = AutoProcessor.from_pretrained(base, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    base, torch_dtype=torch.bfloat16, device_map="cuda"
)

# Base model: chat_template is empty; use vision placeholders + question
text = "<|vision_start|><|image_pad|><|vision_end|>" + q
print("text:", repr(text[:120]))
inputs = proc(text=[text], images=[img], return_tensors="pt")
print("input_ids dtype:", inputs["input_ids"].dtype, "shape:", inputs["input_ids"].shape)
for k, v in inputs.items():
    t = v.to("cuda")
    if k == "input_ids":
        t = t.long()
    inputs[k] = t
out = model(**inputs, output_hidden_states=True)
print("ok hidden", out.hidden_states[-1].shape)
