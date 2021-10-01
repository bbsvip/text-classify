Phân loại nội dung trong câu bằng SVM - Support Vector Machine

## Prerequisites
**pyvi**

**scikit-learn**

**pickle**

**numpy**

## Train
```bash
python train.py
```

## Inference
```py
from text_classify import TextClassify

text = '305 Yên Thế, Đà Nẵng'
classify = TextClassify()
result = classify.classify(text)
```
