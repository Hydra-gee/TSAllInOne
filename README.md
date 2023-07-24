# Recognizing Dominant Patterns for General and Efficient Long-term Time Series Forecasting

This repository only contains the code of PRNet, with eight datasets.
ETTm, ETTh, Exchange, QPS, Stock and Weather are available in "DataCSV", Traffic and Solar should be firstly unzipped.

ECL dataset is too large to upload, but can be downloaded from https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014. After downloaded and unzipped, the "LD2011_2014.txt" file should be renamed as "LD2011_2014.csv" and copied into the "Electricity" folder.

## Run
If you want to reproduce the experiment, please download the datasets from the above link, rename the folder as "DataCSV", unzip all files in "DataCSV" and paste the whole folder into this project.
Consequently, this project will contain four folders "Loader", "PRNet", "Model" and "DataCSV" and a file "main.py".
The "Loader" loads the datasets, "PRNet" is the code of our model, and "Model" saves some cases trained by ourselves.

Run "main.py" and the model will begin training and testing. It will output the amount of parameters, training time and accuracy of PRNet for you to evaluate the generality, efficiency and stability. 

## Visualization
There is also a "Visualize.py" file, which is used for visualizing the forecasting result. After pasting "DataCSV" into the folder, you can run the code and obtain forecasting results. Of course you can input different index to see the forecasting results with different input sequences.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.
