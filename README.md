# Recognizing Dominant Patterns for General and Efficient Long-term Time Series Forecasting

This repository only contains the code of PRNet. Time series datasets are available at https://github.com/Hanwen-Hu/Time-Series-Datasets.

## Run
If you want to reproduce the experiment, please download the datasets from the above link, rename the folder as "DataCSV", and paste it into the folder of this project.
Consequently, this project will contain four folders "Loader", "PRNet", "Model" and "DataCSV" and a file "main.py".
The "Loader" loads the datasets, "PRNet" is the code of our model, and "Model" saves some cases trained by ourselves.

Run "main.py" and the model will begin training and testing. It will output the amount of parameters, training time and accuracy of PRNet for you to evaluate the generality, efficiency and stability. 

## Visualization
There is also a "Visual.ipynb" notebook, which is used for visualizing the forecasting result. After pasting "DataCSV" into the folder, you can run each block in the notebook and obtain results in Figure 17 in our paper.
Of course you can input different index to see the forecasting results with different input sequences.

## Contact
If you have any questions or suggestions for our paper or codes, please contact us. Email: hanwen_hu@sjtu.edu.cn.
