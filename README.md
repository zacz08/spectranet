# Getting Started

## 1. Environment setting
```Shell
conda create -n spectralnet python=3.8 -y
conda activate spectralnet
conda install pytorch
```

## 2. Download the sorce code
```Shell
git clone https://github.com/zacz08/spectralnet.git
cd spectralnet
```

## 3. Train the model
```Shell
python ANet_train.py
```

## 4. Test the model
```Shell
python ANet_test.py
```