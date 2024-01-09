# Recognizing Dominant Patterns for Long-term Time Series Forecasting

## Datasets
This repository only contains the code of PRNet. Eight datasets are available [here](https://github.com/Hanwen-Hu/Time-Series-Datasets), including `ECL`, `ETTh`, `ETTm`, `Exchange`, `QPS`, `Solar`, `Traffic` and `Weather`. They should be firstly unzipped and moved into the `dataset` folder.

Dataset Lists: 
* ETTh.csv
* ETTm.csv
* exchange_rate.csv
* LD2011_2014.txt
* PeMS.csv
* QPS.csv
* solar_alabama.csv
* mpi_roof.csv


## Run
You can run `main.py` to reproduce the experiment. It will output the amount of parameters, training time and accuracy of PRNet for you to evaluate the generality, efficiency and stability. 
The `data_loader` loads the datasets, `model` contains the code of PRNet, and `files` saves some cases trained by ourselves.

## Visualization
There is also a "Visualize.py" file, which is used for visualizing the forecasting result.
The package "files/figures" saves nine forecasting results, you can also run the trained models by yourself.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.

