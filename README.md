# Recognizing Dominant Patterns for Long-term Time Series Forecasting

## Datasets
This repository only contains the code of PRNet, with seven datasets.
`ETTh`, `ETTm`, `Exchange`, `QPS`, and `Weather` are available in the `dataset` folder, `Solar` and `Traffic` should be firstly unzipped.

`ECL` dataset is too large to upload, but can be downloaded from [here](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014). After downloaded and unzipped, the `LD2011_2014.txt` file should be copied into the `dataset` folder. **Attention!** The `,` in `LD2011_2014.txt` should be replaced with `.`, or else the values cannot be successfully identified.

| Dataset  | Lengths | Dims | Files |
|:--------:|:-------:|:----:|:-----:|
|   ECL    | 105217  | 370  |   1   |
|   ETTh   |  17420  |  7   |   2   |
|   ETTm   |  69680  |  7   |   2   |
| Exchange |  7588   |  8   |   1   |
|   QPS    |  30240  |  10  |   1   |
|  Solar   |  52560  | 137  |   1   |
| Traffic  |  17544  | 862  |   1   |
| Weather  |  26064  |  1   |   1   |

## Run
If you want to reproduce the experiment, please run `main.py` and the model will begin training and testing. It will output the amount of parameters, training time and accuracy of PRNet for you to evaluate the generality, efficiency and stability. 
The `data_loader` loads the datasets, `model` contains the code of PRNet, and `files` saves some cases trained by ourselves.

## Visualization
There is also a "Visualize.py" file, which is used for visualizing the forecasting result. After pasting "DataCSV" into the folder, you can run the code and obtain forecasting results. Of course you can input different index to see the forecasting results with different input sequences.

The package "Figure" saves nine forecasting results, you can also run the trained models by yourself.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.

