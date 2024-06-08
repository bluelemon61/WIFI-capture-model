import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import string
import numpy as np

import onnx
import onnxruntime

from utils import AttnLabelConverter

converter = AttnLabelConverter(string.printable[:-6])
num_classes = len(converter.character)
batch_max_length = 25
text = torch.LongTensor(1, batch_max_length + 1).fill_(0) # torch.randint(0, num_classes, (1, batch_max_length+1), dtype=torch.int64)

imgH, imgW = 32, 100

file_name = "./demo_image/demo_0.png"
file_path = os.path.join(os.path.dirname(__file__), file_name)

img = Image.open(file_path).convert('L')
resize = transforms.Resize([imgH, imgW])
img = resize(img)

to_tensor = transforms.ToTensor()
img = to_tensor(img)
img = img.unsqueeze(0)

print(img.shape)


model_name = "model.onnx"
model_path = os.path.join(os.path.dirname(__file__), model_name)
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(model_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ONNX 모델의 입력 차원 확인
for input in ort_session.get_inputs():
    print(f"Input name: {input.name}, shape: {input.shape}, type: {input.type}")

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img),
              ort_session.get_inputs()[1].name: to_numpy(text)}
ort_outputs = ort_session.run(None, ort_inputs)

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
# np.testing.assert_allclose(to_numpy(torch_out), preds[0], rtol=1e-03, atol=1e-05)

preds = torch.tensor(ort_outputs[0])
for i in preds[0]:
    print(f'preds: {i}')
_, preds_index = preds.max(2)
print(f'index: {preds_index}')
length_for_pred = torch.IntTensor([batch_max_length] * 1)
print(length_for_pred)
preds_str = converter.decode(preds_index, length_for_pred)

print(preds_str)