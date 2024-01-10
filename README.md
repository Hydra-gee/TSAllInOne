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

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th colspan="2">1</th>
<th colspan="2">2</th>
<th colspan="2">3.5</th>
<th colspan="2">7.5</th>
</tr>
<tr>
<th>MSE</th><th>MAE</th>
<th>MSE</th><th>MAE</th>
<th>MSE</th><th>MAE</th>
<th>MSE</th><th>MAE</th>
</tr>
</thead>
<tbody>
<tr>
<td>Electricity</td>
</tr>
<tr>
<td>ETTh</td>
<td>0.2139</td><td>0.3078</td>
<td>0.2463</td><td>0.3333</td>
<td>0.2834</td><td>0.3643</td>
<td>0.2463</td><td>0.3333</td>
</tr>
<tr>
<td>ETTm</td>
</tr>
<tr>
<td>Exchange</td>
</tr>
<tr>
<td>QPS</td>
</tr>
<tr>
<td>Solar</td>
</tr>
<tr>
<td>Traffic</td>
</tr>
<tr>
<td>Weather</td>
</tr>
</tbody>
</table>


## Visualization
There is also a "Visualize.py" file, which is used for visualizing the forecasting result.
The package "files/figures" saves nine forecasting results, you can also run the trained models by yourself.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.

