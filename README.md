# Getting Started

## 1. Environment setting
```Shell
conda create -n spectranet python=3.8 -y
conda activate spectranet
conda install pytorch
```

## 2. Download the sorce code
```Shell
git clone https://github.com/zacz08/spectranet.git
cd spectranet
```

## 3. Train the model
```Shell
python model_train.py
```

## 4. Test the model
```Shell
python model_test.py
```