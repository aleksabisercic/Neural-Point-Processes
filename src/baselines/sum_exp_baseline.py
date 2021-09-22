
import numpy as np
import pandas as pd
from tick.hawkes import SimuHawkesSumExpKernels, HawkesSumExpKern
from tick.dataset import fetch_hawkes_bund_data


def fit_exp_hawkes_and_simulate(train_times, decays, end_time):
    learner = HawkesSumExpKern(decays, penalty='elasticnet', elastic_net_ratio=0.8)
    learner.fit(train_times)
    score = learner.score()
    print(f'obtained {score}\n\n')

    # simulation
    simulation = SimuHawkesSumExpKernels(
        adjacency=learner.adjacency, decays=decays, baseline=learner.baseline, end_time=end_time,
        verbose=False, seed=42)
    simulation.simulate()

    return learner, simulation


def perform_ny_exp_test():
    # load ny taxi data
    ny_train_set = pd.read_csv('../data/nytaxi/taxi_feature_matrix_train.csv')
    ny_test_set = pd.read_csv('../data/nytaxi/taxi_feature_matrix_test.csv')
    ny_train_times = np.array(ny_train_set.query('target == 1').time)
    ny_test_times = np.array(ny_test_set.query('target == 1').time)
    decays = [0.1, 0.5, 1.]

    model, exp_simulation = fit_exp_hawkes_and_simulate([[ny_train_times]], decays, round(np.max(ny_test_times))+10)
    print("predicted: " + str(exp_simulation.timestamps))
    print("real: " + str(ny_train_times))

    predicted_in_test_period = exp_simulation.timestamps[0][exp_simulation.timestamps[0] > ny_test_times[0]]

    real_zero_extends = np.full((len(predicted_in_test_period) - len(ny_test_times)), 0)
    test_df = pd.DataFrame({'real': np.append(np.around(ny_test_times), real_zero_extends, 0),
                            'predict': np.around(predicted_in_test_period)}, columns=['real', 'predict'])
    test_df.to_csv('../results/ny_obtained_times_sumexp.csv', index=False)

    # evaluation_df = pd.DataFrame(columns=['name', 'dataset', "score_on_train",  "score_on_test"])
    evaluation_df = pd.read_csv('../results/baseline_scores.csv')
    evaluation_df.loc[len(evaluation_df)] = ['HawkesSumExpKern', 'ny_taxi', model.score(), model.score([[ny_test_times]])]
    evaluation_df.to_csv('../results/baseline_scores.csv', index=False)


def perform_autoput_exp_test():
    autoput = pd.read_csv('../data/autoput/012017_bg-nis-stan3-traka1-prepared.csv')
    autoput['time_from_zero'] = autoput['epoch_rounded'] - autoput['epoch_rounded'].min()
    train_set = autoput[autoput.index < autoput.shape[0] * 0.8]
    test_set = autoput[autoput.index > autoput.shape[0] * 0.8]
    auto_train_times = np.array(train_set.time_from_zero).astype('float64')
    auto_test_times = np.array(test_set.time_from_zero).astype('float64')
    decays = [0.1, 0.5, 1.]

    model, exp_simulation = fit_exp_hawkes_and_simulate([[auto_train_times]], decays, round(np.max(auto_test_times))+10)
    print("predicted: " + str(exp_simulation.timestamps))
    print("real: " + str(auto_train_times))

    predicted_in_test_period = exp_simulation.timestamps[0][exp_simulation.timestamps[0] > auto_test_times[0]]

    real_zero_extends = np.full((len(predicted_in_test_period) - len(auto_test_times)), 0)
    test_df = pd.DataFrame({'real': np.append(np.around(auto_test_times), real_zero_extends, 0),
                            'predict': np.around(predicted_in_test_period)}, columns=['real', 'predict'])
    test_df.to_csv('../results/autoput_obtained_times_sumexp.csv', index=False)

    # evaluation_df = pd.DataFrame(columns=['name', 'dataset', "score_on_train",  "score_on_test"])
    evaluation_df = pd.read_csv('../results/baseline_scores.csv')
    evaluation_df.loc[len(evaluation_df)] = ['HawkesSumExpKern', 'autoput', model.score(), model.score([[auto_test_times]])]
    evaluation_df.to_csv('../results/baseline_scores.csv', index=False)


def perform_finance_exp_test():
    timestamps_list = fetch_hawkes_bund_data()
    one_day_movement_up = timestamps_list[0][0]
    finance_train_times = one_day_movement_up[:round(len(one_day_movement_up) * 0.8)]
    finance_test_times = one_day_movement_up[round(len(one_day_movement_up) * 0.8):]
    decays = [0.1, 0.5, 1.]

    model, exp_simulation = fit_exp_hawkes_and_simulate([[finance_train_times]], decays, round(np.max(finance_test_times))+10)
    print("predicted: " + str(exp_simulation.timestamps))
    print("real: " + str(finance_train_times))

    predicted_in_test_period = exp_simulation.timestamps[0][exp_simulation.timestamps[0] > finance_test_times[0]]

    real_zero_extends = np.full((len(predicted_in_test_period) - len(finance_test_times)), 0)
    test_df = pd.DataFrame({'real': np.append(np.around(finance_test_times), real_zero_extends, 0),
                            'predict': np.around(predicted_in_test_period)}, columns=['real', 'predict'])
    test_df.to_csv('../results/finance_obtained_times_sumexp.csv', index=False)

    # evaluation_df = pd.DataFrame(columns=['name', 'dataset', "score_on_train",  "score_on_test"])
    evaluation_df = pd.read_csv('../results/baseline_scores.csv')
    evaluation_df.loc[len(evaluation_df)] = ['HawkesSumExpKern', 'finance', model.score(), model.score([[finance_test_times]])]
    evaluation_df.to_csv('../results/baseline_scores.csv', index=False)


if __name__ == '__main__':
    perform_ny_exp_test()
    perform_autoput_exp_test()
    perform_finance_exp_test()






