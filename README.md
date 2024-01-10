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
<th rowspan="2">L</th>
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
<td>Electricity</td><td>96</td>
</tr>
<tr>
<td>ETTh</td><td>24</td>
<td>0.2139</td><td>0.3078</td>
<td>0.2463</td><td>0.3333</td>
<td>0.2834</td><td>0.3643</td>
<td>0.2463</td><td>0.3333</td>
</tr>
<tr>
<td>ETTm</td><td>96</td>
<td>0.2012</td><td>0.2978</td>
<td>0.2401</td><td>0.3293</td>
<td>0.2772</td><td>0.3632</td>
<td>0.3354</td><td>0.4057</td>
</tr>
<tr>
<td>Exchange</td><td>30</td>
<td>0.0204</td><td>0.0996</td>
<td>0.0386</td><td>0.1360</td>
<td>0.0635</td><td>0.1776</td>
<td>0.1332</td><td>0.2622</td>
</tr>
<tr>
<td>QPS</td><td>60</td>
</tr>
<tr>
<td>Solar</td><td>288</td>
</tr>
<tr>
<td>Traffic</td><td>24</td>
</tr>
<tr>
<td>Weather</td><td>144</td>
</tr>
</tbody>
</table>


## Visualization
There is also a "Visualize.py" file, which is used for visualizing the forecasting result.
The package "files/figures" saves nine forecasting results, you can also run the trained models by yourself.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.

