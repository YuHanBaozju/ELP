
# ELP-One-step-Event-driven-AF

This repository contains the code and resources for **One-Step Event-Driven High-Speed Autofocus (CVPR2025)**, which evaluates the ELP method on three datasets.

## Dataset and Project Download

- Download the datasets from the following link:  
  [Datasets on Google Drive](https://drive.google.com/drive/folders/1YGF8_uVfmoGnSHJ6QYEECnYdhSWOQl5b?usp=sharing)

- Clone this project repository:
  ```
  git clone https://github.com/YuHanBaozju/ELP.git
  cd ELP-One-step-Event-driven-AF
  ```


## Testing ELP on Three Datasets

### 1. DAVIS Dataset
Navigate to the `DAVIS_ELP` directory and run:
```
cd DAVIS_ELP
python eval_davis.py
```
**Note**:  
Before running, make sure to modify the dataset root directory path inside `eval_davis.py` to point to your downloaded DAVIS dataset.

### 2. EVK4 Dataset
Navigate to the `EVK4_ELP` directory and run:
```
cd EVK4_ELP
python eval_evk4.py
```
**Note**:  
Modify the dataset root directory path inside `eval_evk4.py` accordingly.

### 3. Synthetic Dataset
Navigate to the `synthetic_ELP` directory and run:
```
cd synthetic_ELP
python eval_sim.py
```
**Note**:  
Modify the dataset root directory path inside `eval_sim.py` accordingly.


## Visualization of Results

To visualize the ELP method’s performance on different datasets:

- Set `vis=True` in each of the corresponding `eval_xxx.py` files:

## Requirements
opencv-python
numpy
matplotlib

## Citation
#TBD
