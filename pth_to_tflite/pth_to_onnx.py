import os
import numpy as np
import string

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

from utils import AttnLabelConverter
import inspect

batch_max_length = 25
input_channel = 1
output_channel = 512
hidden_size = 256
character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()[]_-"
converter = AttnLabelConverter(string.printable[:-6])
num_classes = len(converter.character)

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    self.stages = {'Trans': 'TPS', 'Feat': 'ResNet',
                          'Seq': 'BiLSTM', 'Pred': 'Attn'}

    self.Transformation = TPS_SpatialTransformerNetwork(
      F=20, I_size=(32, 100), I_r_size=(32, 100), I_channel_num=input_channel)
    
    self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
    self.FeatureExtraction_output = output_channel
    self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((32, 1)) # warning, origin is (None, 1)

    
    self.SequenceModeling = nn.Sequential(
      BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
      BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
    self.SequenceModeling_output = hidden_size

    self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_classes)
  
  def forward(self, input, text, is_train=True):
    """ Transformation stage """
    if not self.stages['Trans'] == "None":
      input = self.Transformation(input)

    """ Feature extraction stage """
    visual_feature = self.FeatureExtraction(input)
    visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
    visual_feature = visual_feature.squeeze(3)

    """ Sequence modeling stage """
    if self.stages['Seq'] == 'BiLSTM':
        contextual_feature = self.SequenceModeling(visual_feature)
    else:
        contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

    """ Prediction stage """
    if self.stages['Pred'] == 'CTC':
        prediction = self.Prediction(contextual_feature.contiguous())
    else:
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=batch_max_length)

    return prediction


# 파일 경로 확인
file_name = 'best_accuracy_24052602.pth'
file_path = os.path.join(os.path.dirname(__file__), file_name)

# 파일이 존재하는지 확인
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_name} does not exist in the script's directory.")

model = Model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.nn.DataParallel(model).to(device)

model.load_state_dict(torch.load(file_path, map_location=device))
print(model)

# 모델 클래스의 forward 메서드 확인
forward_args = inspect.signature(Model.forward).parameters

# forward 메서드의 입력 형태 출력
print("모델의 forward 메서드 입력 형태:", forward_args['text'])

model.eval()

# forward->  input: [batch_size x I_channel_num x I_height x I_width], text='test'
example_input = torch.randn(1, input_channel, 32, 100)
# [batch_size x (max_length+1)]
example_text = torch.randint(0, num_classes, (1, batch_max_length+1), dtype=torch.int64)

# DataParallel로 래핑된 모델에서 원래의 모델을 추출
if isinstance(model, nn.DataParallel):
    model = model.module

onnx_file_path = "model.onnx"
torch.onnx.export(
    model,                                     # PyTorch 모델
    (example_input, example_text),             # 예제 입력 튜플
    onnx_file_path,                            # 저장할 ONNX 파일 경로
    export_params=True,                        # 모델 파라미터도 함께 저장
    opset_version=16,                          # ONNX 버전 (최신 버전을 사용)
    do_constant_folding=True,                  # 상수 폴딩 최적화 적용
    input_names=['input', 'text'],             # 입력 레이어 이름
    output_names=['output'],                   # 출력 레이어 이름
    dynamic_axes={
        'input': {0: 'batch_size'},            # 배치 크기 가변
        'text': {0: 'batch_size'},             # 배치 크기 가변
        'output': {0: 'batch_size'}            # 배치 크기 가변
    }
)

print(f"모델이 성공적으로 {onnx_file_path} 파일로 변환되었습니다.")