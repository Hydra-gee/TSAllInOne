# Recognizing Dominant Patterns for Long-term Time Series Forecasting

## Datasets
This repository only contains the code of PRNet. Seven datasets are available [here](https://www.kaggle.com/datasets/limpidcloud/datasets-for-multivariate-time-series-forecasting), including `Electricity`, `ETT`, `Exchange`, `QPS`, `Solar`, `Traffic` and `Weather`. They should be firstly unzipped and moved into the `dataset` folder.


Dataset Lists: 
* Electricity
  * LD2011_2014.csv
  * LD2011_2014_h.csv
* ETT
  * ETTh.csv
  * ETTm.csv
* Exchange
  * exchange_rate.csv
* QPS
  * HQPS.csv
  * MQPS.csv
* Solar
  * solar_Alabama.csv
  * solar_Alabama_h.csv
* Traffic
  * PeMS.csv
* Weather
  * mpi_roof.csv
  * mpi_roof_h.csv


## Run
You can run `main.py` to reproduce the experiment. Below is an example of running the `Traffic` dataset with `pred_len = 24`.
```
python3 main.py -cuda_id 0 -dataset Traffic -pred_len 24
```
There are seven dataset names: 
```
Electricity ETT Exchange QPS Solar Traffic Weather
```
and their hyperparameters are listed in `files/configs.json`.

The `data_loader` loads the datasets, `model` contains the code of PRNet, and `files/networks` saves the cases trained by ourselves. 
Below is the experiment result.

We have also unified the **sampling interval** of datasets except for Exchange to **1 hour** for better comparison. You can set the parameter `hour_sampling` to `True` to evaluate these unified datasets.

## Visualization
There is also a `visualize` function, which is used for visualizing the forecasting result.
The package `files/figures` saves eight forecasting results, you can also run the trained models by yourself.

## Forecasting Accuracy
<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th rowspan="2">L</th>
<th colspan="2">L</th>
<th colspan="2">2L</th>
<th colspan="2">3.5L</th>
<th colspan="2">7.5L</th>
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
<td>0.1333</td><td>0.2263</td>
<td>0.1572</td><td>0.2453</td>
<td>0.1604</td><td>0.2453</td>
<td>0.1638</td><td>0.2513</td>
</tr>
<tr>
<td>ETT</td><td>96</td>
<td>0.2052</td><td>0.2988</td>
<td>0.2437</td><td>0.3291</td>
<td>0.2813</td><td>0.3571</td>
<td>0.3366</td><td>0.3987</td>
</tr>
<tr>
<td>Exchange</td><td>30</td>
<td>0.0204</td><td>0.0961</td>
<td>0.0371</td><td>0.1326</td>
<td>0.0630</td><td>0.1765</td>
<td>0.1357</td><td>0.2635</td>
</tr>
<tr>
<td>QPS</td><td>60</td>
<td>0.0279</td><td>0.0928</td>
<td>0.0583</td><td>0.1474</td>
<td>0.1294</td><td>0.2297</td>
<td>0.3233</td><td>0.3853</td>
</tr>
<tr>
<td>Solar</td><td>288</td>
<td>0.1836</td><td>0.2421</td>
<td>0.1980</td><td>0.2558</td>
<td>0.2066</td><td>0.2594</td>
<td>0.2069</td><td>0.2557</td>
</tr>
<tr>
<td>Traffic</td><td>24</td>
<td>0.3239</td><td>0.3007</td>
<td>0.3541</td><td>0.3226</td>
<td>0.3781</td><td>0.3380</td>
<td>0.3794</td><td>0.3365</td>
</tr>
<tr>
<td>Weather</td><td>144</td>
<td>0.3405</td><td>0.3328</td>
<td>0.4138</td><td>0.3923</td>
<td>0.4755</td><td>0.4386</td>
<td>0.5455</td><td>0.4807</td>
</tr>
<tr>
<td>ETTh</td><td>24</td>
<td>0.2134</td><td>0.3046</td>
<td>0.2458</td><td>0.3301</td>
<td>0.2834</td><td>0.3576</td>
<td>0.3328</td><td>0.3969</td>
</tr>
</tbody>
</table>

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.

