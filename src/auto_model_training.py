
import pickle

import pandas as pd
import torch
import os
import matplotlib.pyplot as plt

from Pytorch.model import FCN_point_process_all, GRU_point_process_all, LSTM_point_process_all, RNN_point_process_all

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # project_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.path.dirname(os.path.realpath(__file__)))

    #CUDA setup
    print(f'Cuda is available: {torch.cuda.is_available()}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device_id = torch.device(device)
        torch.cuda.set_device(device_id)
        print(f"Current active cuda: {torch.cuda.current_device()}")
        
    os.environ['DATA_FOLDER'] = './data/'
    os.environ["TRAINING_DATASET"] = 'stan1_traka1_01012017.csv'
    
    data_folder = os.environ['DATA_FOLDER']
    train_df = pd.read_csv(data_folder+os.environ["TRAINING_DATASET"])
    
    #scaling data 
    scaling_coficient = 60 #this will scale data from seconds to minutes
    
    train_time = torch.tensor(train_df.event_time.values[:int(len(train_df.event_time.values)*0.8)]).type('torch.FloatTensor').reshape(1, -1, 1).to(device)
    test_time = torch.tensor(train_df.event_time.values)[int(len(train_df.event_time.values)*0.8):].type('torch.FloatTensor').reshape(1, -1, 1).to(device)
    
    train_time = train_time/scaling_coficient/scaling_coficient
    test_time = test_time/scaling_coficient/scaling_coficient
    
    #data_folder = project_dir+'/data/'
    # train_df = pd.read_csv('stan1_traka1_01012017.csv')
    # test_df = pd.read_csv('stan1_traka1_01012017.csv')
    # data.date1 = pd.to_datetime(data.date1)
    # train_data = data[data.date1.dt.hour < 21]
    # test_data = data[data.date1.dt.hour >= 21]
    # train_time = torch.tensor(train_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    # test_time = torch.tensor(test_data.date1_ts.values).type('torch.FloatTensor').reshape(1, -1, 1)
    
    in_size = 5
    out_size = 1

    learning_param_map = [
        {'rule': 'Euler', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Implicit Euler', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Trapezoid', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Simpsons', 'no_step': 10, 'learning_rate': 0.001},
        {'rule': 'Gaussian_Q', 'no_step': 10, 'learning_rate': 0.001}
    ]
    analytical_definition = [{'rule': 'Analytical', 'no_step': 2, 'learning_rate': 0.001}]
    models_to_evaluate = [
        # {'model': FCN_point_process_all(in_size+1, out_size, drop=0.1), 'learning_param_map': learning_param_map},
        {'model': GRU_point_process_all(in_size+1, out_size, drop=0.0), 'learning_param_map': learning_param_map},
        {'model': LSTM_point_process_all(in_size+1, out_size, drop=0.0), 'learning_param_map': learning_param_map},
        {'model': RNN_point_process_all(in_size+1, out_size, drop=0.0), 'learning_param_map': learning_param_map}
    ]

    print(f'Train size: {str(train_time.shape[1])}, test size: {str(test_time.shape[1])}.')

    in_size = 5
    out_size = 1
    no_epochs = 3
    evaluation_df = pd.DataFrame(columns=['model_name', 'rule', 'no_step', 'learning_rate', 'loss_on_train', 'loss_on_test'])

    for model_definition in models_to_evaluate:
        for params in model_definition['learning_param_map']:
            model = model_definition['model']
            epochs, train_losses, test_losses = model.fit(train_time, test_time, in_size, no_epoch=no_epochs,
                      no_steps=params['no_step'], method=params['rule'], log_epoch=10)

            model_name = f"autoput-01012017-{type(model).__name__}-{params['rule']}"
            train_losses = [loss.detach().numpy().flatten()[0] for loss in train_losses]
            test_losses = [loss.detach().numpy().flatten()[0] for loss in test_losses]
            print(train_losses, test_losses)

            plt.plot(epochs, train_losses, color='skyblue', linewidth=2, label='train')
            plt.plot(epochs, test_losses, color='darkgreen', linewidth=2, linestyle='dashed', label="test")
            plt.legend()
            plt.savefig(f'img/{model_name}.png')

            loss_on_train = model.evaluate(train_time, in_size, method='Trapezoid')
            loss_on_test = model.evaluate(test_time, in_size, method='Trapezoid')
            print(f"Model: {model_name}. Loss on train: {str(loss_on_train.data.numpy())},  "
                  f"loss on test: {str(loss_on_test.data.numpy())}")
            evaluation_df.loc[len(evaluation_df)] = [type(model).__name__,
                                                     params['rule'],
                                                     params['no_step'],
                                                     params['learning_rate'],
                                                     loss_on_train.data.numpy().flatten()[0],
                                                     loss_on_test.data.numpy().flatten()[0]]
            model_filepath = f"models/autoput-012017/{model_name}.torch"
            pickle.dump(model, open(model_filepath, 'wb'))

    print(evaluation_df)
    evaluation_df.to_csv('results/jan_autoput_scores_0.1.csv', index=False)




