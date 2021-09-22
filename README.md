# A MACHINE LEARNING APPROACH FOR LEARNING TEMPORAL POINT PROCESS
## *Abstract:*
_Despite a vast application of temporal point processes in infectious disease diffusion forecasting, ecommerce, traffic prediction, preventive maintenance and many others there is no significant development in improving the simulation and prediction of temporal point processes in real world environments. With this problem at hand we propose a novel methodology for learning temporal point processes based on one-dimensional numerical integration techniques. These techniques are used for linearising the negative maximum likelihood (neML) function and to enable backpropagation of the neML derivatives. Our approach is tested on two real-life datasets. Firstly, on high frequency point process data, prediction of highway traffic, and secondly, on a very low frequency point processes dataset, prediction of ski injuries in ski resorts. Four different point process baseline models were compared: Fully connected neural network, LSTM model, RNN model and GRU model. The results show the ability of the proposed methodology to generalize on different datasets and illustrate how different numerical integration techniques and mathematical models influence the quality of the obtained models. The presented methodology is not limited to these datasets and can be further used to optimize and predict other processes that are based on temporal point processes._

To start training models run the `run.sh` via terminal:
```
sh run.sh
```
