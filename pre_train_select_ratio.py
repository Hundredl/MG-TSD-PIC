import pickle
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
import os
import copy
from time_grad_estimator_2 import TimeGradEstimator
from trainer import Trainer

import wandb
wandb.login()

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='timegrad', help='model name')
    parser.add_argument('--cuda_num', type=str, default='8')
    parser.add_argument('--result_path', type=str, default='./results/#')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-05)
    parser.add_argument('--diff_steps',type=int, default=100)
    parser.add_argument('--num_cells',type=int, default=128)

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--input_size', type=int, default=552)
    parser.add_argument('--batch_size', type=int, default=128)
    


    # 返回一个命名空间，包含传递给命令的参数
    return parser.parse_args()

alias = {'fina':'m3_other','elec':'electricity_nips','exc':'exchange_rate_nips','wiki':'wiki-rolling_nips',
         'cup':'kdd_cup_2018_without_missing','solar':'solar_nips','traf':'traffic',
         'fre':'fred_md'}

args = parse_args()
wandb.init(project="timegrad",save_code=True,config=args)

model_name=args.model_name
cuda_num = args.cuda_num
result_path = args.result_path
epoch = args.epoch
diff_steps=args.diff_steps

dataset_name = args.dataset
input_size = args.input_size
batch_size = args.batch_size
learning_rate=args.learning_rate
num_cells=args.num_cells

print(args)


def create_granularity_data(dataset_train, dataset_test,gran=['1']):
    coarse_data_dict = {}
    dataset_train_coarse = []
    dataset_test_coarse = []
    dataDict_train_coarse = {}

    for dataDict in dataset_train:
        # reshape the array to have n columns
        coarse_array = []
        for item in dataDict['target']:
            arr = item[0:7008].reshape(-1, int(gran))
            avg_arr = np.mean(arr, axis=1)
            # print(sum_arr.shape)
            coarse_array.append(avg_arr)
        dataDict_train_coarse['target'] = np.array(coarse_array)
        dataDict_train_coarse['feat_static_cat'] = dataDict['feat_static_cat']
        dataDict_train_coarse['start'] = dataDict['start']
        dataset_train_coarse.append(dataDict_train_coarse)



    for dataDict in dataset_test:
        dataDict_test_coarse = {} # create a new dictionary for the coarse-grained dataset, should be put under the for loop

        coarse_array = []
        print(f"len of dataDict['target'] is {(dataDict['target'].shape)}")
        for item in dataDict['target']:
            lenn=len(item)
            # print("length of item ")
            # print(lenn)
            arr = item[:(lenn-1)].reshape(-1, int(gran))
        # sum every 4 elements of the array
            avg_arr = np.mean(arr, axis=1)
            #coarse_array.append(np.repeat(avg_arr,int(gran)))
            coarse_array.append(avg_arr)

        dataDict_test_coarse['target'] = np.array(coarse_array) 
        dataDict_test_coarse['feat_static_cat'] = dataDict['feat_static_cat']
        dataDict_test_coarse['start'] = dataDict['start']
        dataset_test_coarse.append(dataDict_test_coarse)
       
    coarse_data_dict[gran] = {}
    coarse_data_dict[gran]['train'] = dataset_train_coarse
    coarse_data_dict[gran]['test'] = dataset_test_coarse

    return coarse_data_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = get_dataset("solar_nips", regenerate=False)


train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                   max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

print("prepare the fine-grained dataset")
dataset_train=train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)


for data_i in dataset_test:
    print(data_i['target'].shape)

fine_data=create_granularity_data(dataset_train,dataset_test,gran='1')
dataset_train_fine=fine_data['1']['train']
dataset_test_fine=fine_data['1']['test']

#create coarse data
# gran='168'
# coarse_data= create_granularity_data(dataset_train=dataset_train, 
#                                      dataset_test=dataset_test, 
#                                      gran=gran)
# data_train_coarse=coarse_data[gran]['train']
# data_test_coarse=coarse_data[gran]['test']


for data_i in dataset_test_fine:
    print(data_i['target'].shape)

print("train fine-grained dataset shape")
print(dataset_train_fine[0]['target'].shape)
print("test fine-grained dataset shape")
print(dataset_test_fine[0]['target'].shape)
print("freq")
print(dataset.metadata.freq)
print("prediction length")
print(dataset.metadata.prediction_length)


#does the test and train the same??
# print("orginal dataset target shape")
# print(list(dataset_test)[0]['target'].shape)

import warnings
warnings.filterwarnings("ignore")
estimator = TimeGradEstimator(
    target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    prediction_length=dataset.metadata.prediction_length, 
    context_length=dataset.metadata.prediction_length, 
    cell_type='GRU',
    input_size=552,
    freq=dataset.metadata.freq,
    loss_type='l2',
    scaling=True,
    diff_steps=100,
    beta_end=0.1,
    beta_schedule="linear",
    num_cells=num_cells,
    trainer=Trainer(device=device,
                    epochs=epoch,
                    learning_rate=learning_rate,
                    num_batches_per_epoch=100,
                    batch_size=batch_size,)
)

# estimator2 = TimeGradEstimator(
#     target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
#     prediction_length=6, 
#     context_length=24, 
#     cell_type='GRU',
#     input_size=552,
#     freq=dataset.metadata.freq,
#     loss_type='l2',
#     scaling=True,
#     diff_steps=diff_steps,
#     beta_end=0.1,
#     beta_schedule="linear",
#     lags_seq=[1,6,42],
#     trainer=Trainer(device=device,
#                     epochs=1,
#                     learning_rate=learning_rate,
#                     num_batches_per_epoch=100,
#                     batch_size=batch_size,)
# )


print("train the timegrad estimator on the fine-grained dataset")
predictor = estimator.train(dataset_train_fine, num_workers=8)
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test_fine,
                                                predictor=predictor, num_samples=100)




# print("train the timegrad estimator on the coarse-grained dataset")
# predictor2= estimator2.train(data_train_coarse,num_workers=8,validation_data=data_test_coarse) #t
# forecast_it_2, ts_it_2 = make_evaluation_predictions(dataset=data_test_coarse,
#                                                     predictor=predictor2,
#                                                     num_samples=100)
print("make predictions")
forecasts = list(forecast_it)
targets = list(ts_it)
print("forecaset shape")
print(forecasts[0].samples.shape)

# print("make prediction (coarse dataset)")
# forecasts_2 = list(forecast_it_2)
# targets_2 = list(ts_it_2)
# with open('targets_1week.pkl', 'wb') as f:
#     pickle.dump(targets_2, f)

print("===================================================")
print("make evaluations on forecase from timegrad")
evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], 
                                target_agg_funcs={'sum': np.sum})

agg_metric, item_metrics = evaluator(targets, forecasts, 
                                        num_series=len(dataset_test_fine))


current_path=os.getcwd()
print(current_path)
# Set the file path for loading the pickle file

load_path =os.path.join(current_path, 'samples_list.pt')

# Open the file in binary mode and load the object using pickle.load
samples_list = torch.load(load_path)

print(samples_list[1].size())



samples_list_2=[]
for i in range(100):
    inter_forecast=copy.deepcopy(forecasts)
    for day in range(7):
        print("one day forecast")
        tem=samples_list[i][day,:,:,:].cpu()
        print(tem.size())  #torch.Size([100, 6, 137])
        inter_forecast[day].samples=tem.numpy()
    samples_list_2.append(inter_forecast)

with open('sample_list_numpy.pkl', 'wb') as f:
    pickle.dump(samples_list_2, f)
# print("initialize evaluator")

# #replicat the targets_2
# # targets_coarse=copy.deepcopy(targets_2)
# # for day in range(7):
# #     repeated_data=np.tile(targets_2[day].values[:,np.newaxis],(1,4,1))
# #     targets_coarse[day]=pd.DataFrame(repeated_data) #torch.Size([100, 6, 137])


# # #convert sample list for tensor to array

# # print("evaluate the discrepany at each timestamp between sampling distribution and target distribution")
# # CRPS_Sum_100=np.zeros(100)
# # ND_Sum_100=np.zeros(100)
# # NRMSE_Sum_100=np.zeros(100)
# # for i in range(100):
# #     agg_metric, item_metrics = evaluator(targets_coarse, samples_list_2[i], 
# #                                         num_series=len(dataset_test_fine))
# #     CRPS_Sum_100[i] = agg_metric["m_sum_mean_wQuantileLoss"]
# #     ND_Sum_100[i] = agg_metric["m_sum_ND"]
# #     NRMSE_Sum_100[i] = agg_metric["m_sum_NRMSE"]
    



# # idx_0=np.argmin(CRPS_Sum_100)
# # idx_1=np.argmin(ND_Sum_100)
# # idx_2=np.argmin(NRMSE_Sum_100)