# Wifi-Password-Regconization-model
WIFI Connector Android 앱에 들어갈 OCR 모델

[WIFI Connector android app repository](https://github.com/kyoutae1234/wifiConnector)

## Preprocessing
``` shell
git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
pip3 install lmdb pillow torchvision nltk natsort
pip3 install fire
```

## Data Set

``` shell
git clone https://github.com/Belval/TextRecognitionDataGenerator.git
pip install -r ./TextRecognitionDataGenerator/requirements.txt
```

handwriten text 필요시
` pip3 install -r requirements-hw.txt ` <br/>

``` shell
cd .\TextRecognitionDataGenerator\
python trdg/run.py -c 300000 -d 3 -rs -f 64 --length 4
```

만들어진 데이터 셋을 train, validation, evaluation 데이터로 분리.
LMDB 형식으로 변환

``` shell
python deep-text-recognition-benchmark/create_lmdb_dataset.py --inputPath dataset/training/ --outputPath ./data/train --gtFile dataset/train_gt.txt
```

## Base-Model (Pretrained)

[TPS-ResNet-BiLSTM-Attn-case-sensitive](https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?e=1&dl=0)

## Training (Fine Tuning)
``` shell
python deep-text-recognition-benchmark/train.py --train_data data/training --valid_data data/validation/ --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --sensitive --workers 0 --batch_size 96 --saved_model ./saved_models/target.pth
```

## Test
``` shell
python deep-text-recognition-benchmark/test.py --eval_data data/evaluation/ --benchmark_all_eval --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --sensitive --workers 0 --batch_size 96 --saved_model ./saved_models/target.pth
```

## pth to onnx
1. pth -> onnx 변환 - [pth_to_onnx/pth_to_onnx.py](./pth_to_onnx/pth_to_onnx.py)
2. onnx 테스트 - [pth_to_onnx/onnx_test.py](./pth_to_onnx/onnx_test.py)

## Final Model
[Google Drive](https://drive.google.com/file/d/1UhUPa-_4R8eeKn1Z0iWjuMo5YEQdmjB9/view?usp=sharing)