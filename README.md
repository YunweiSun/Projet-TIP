# Projet-TIP

## Description
This is our proposition for https://www.kaggle.com/competitions/siim-isic-melanoma-classification/

## Usage
To install PyTorch
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- loadData.py
- network.py
- resnet18.py
- resnet18_g.py
- train_test.py

These three files need to be in the same directory as the isic-2020-resized folder, to run the code, first run img-csv.py to put all names of test images in test.csv, then run train_test.py.
