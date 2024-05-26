``` shell
git clone https://github.com/Belval/TextRecognitionDataGenerator.git
pip install -r .\TextRecognitionDataGenerator\requirements.txt
```

handwriten text 필요시
` pip3 install -r requirements-hw.txt ` <br/>

``` shell
cd .\TextRecognitionDataGenerator\
python trdg/run.py -c 300000 -d 3 -rs -f 64 --length 4
```