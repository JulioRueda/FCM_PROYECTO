from omegaconf import DictConfig
import numpy as np
import hydra 
import mlflow
import torch 
import torch.nn as nn
import pandas as pd
import os
from sklearn.metrics import f1_score
from data_utils import trainTestValidationSplit, loadTrainTestValidation, load_dataset
from data_utils import MLP, AdjustNeuralNetworkWeights, getPartition,FetchInformativeSamples
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


@hydra.main(config_path="../conf", config_name='config', version_base=None)
def run_experiment(cfg: DictConfig):

    #MLFLOW
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    base = rf'C:\Users\jc.ruedah\Desktop\FCM_carpeta\FCM_PROJECT\data\processed'
    #INVOCAR MANUALMENTE AL DATASET, 
    I, gt = load_dataset(cfg.dataset.name,cfg.dataset.path)

    if not os.path.exists(rf"{base}\{cfg.dataset.name}_Train_Trial_{cfg.experiment.seed}.csv"):
      
      (T, V, t, classes) = trainTestValidationSplit(I, gt, 0.75, 0.15, 0.10)

      df_T = pd.DataFrame(T['idxs'], columns=['idxs']).to_csv(rf"{base}\{cfg.dataset.name}_Train_Trial_{cfg.experiment.seed}.csv", index=False)
      df_V = pd.DataFrame(V['idxs'], columns=['idxs']).to_csv(rf"{base}\{cfg.dataset.name}_Val_Trial_{cfg.experiment.seed}.csv", index=False)
      df_t = pd.DataFrame(t['idxs'], columns=['idxs']).to_csv(rf"{base}\{cfg.dataset.name}_Test_Trial_{cfg.experiment.seed}.csv", index=False)
    else:
      (T, V, t, classes) = loadTrainTestValidation(I, gt, base, cfg.dataset.name, cfg.experiment.seed)
    #END_FILE_VERIFY

    C = classes.shape[0]
    B = T['X'].shape[1]
    print('datos cargados')
    #Modelo base
    M = []
    torch.manual_seed(cfg.experiment.seed) # control randomness of model, given some trial. Only varies when trial is different
    M = MLP(_n_features=T['X'].shape[1], _n_hidden_1=cfg.experiment.n_neurons, _n_classes=classes.shape[0])
    M.to(device)
    M.compile(_device=device,
              _loss_fn=nn.CrossEntropyLoss(),
              #_optimizer=torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9),
              _optimizer=torch.optim.Adam(M.parameters(), lr=1e-2)
            )
    M, elapsed = AdjustNeuralNetworkWeights(M, T, V)
    f1_macro_T   = f1_score(y_true=T['y'], y_pred=M.predict_class(T['X']), average='macro')
    f1_macro_t   = f1_score(y_true=t['y'], y_pred=M.predict_class(t['X']), average='macro')

    #Modelos por partición
    E    = {'X':np.empty(shape=[0, B]), 'y':np.empty(shape=[0]), 'idxs':np.empty(shape=[0, 1])}
    T_hat= {'X':np.empty(shape=[0, B]), 'y':np.empty(shape=[0]), 'idxs':np.empty(shape=[0, 1])}
    P    = getPartition(T, cfg.experiment.subsets, _random_state=cfg.experiment.seed) #Training subsets

    print('modelo base creado')


    for k in range(1,cfg.experiment.subsets+1):
      print(rf'-->{cfg.experiment.subsets}:{k}')
      with mlflow.start_run(run_name="nombre del registro /fila"):
        
        if k < cfg.experiment.subsets:
          T_k = P[k]
        else:
          T_k = {'X':np.empty(shape=[0,B]), 'y':np.empty(shape=[0]), 'idxs':np.empty(shape=[0,1])}

        # Union of the k-th training subset and the missclassified pixels
        T_tilde = {'X':np.concatenate((E['X'], T_k['X']), axis=0),
                  'y':np.concatenate((E['y'], T_k['y']), axis=0)}

        # Neural Network Weights Adjustment
        M, elapsed = AdjustNeuralNetworkWeights(M, T_tilde, V)

        # Union of k - 1 subsets
        T_hat['X'] = np.concatenate((T_hat['X'], T_k['X']), axis=0)
        T_hat['y'] = np.concatenate((T_hat['y'], T_k['y']), axis=0)
        T_hat['idxs'] = np.concatenate((T_hat['idxs'], T_k['idxs']), axis=0)

        # Fetching informative samples
        E = {'X':np.empty(shape=[0, B]), 'y':np.empty(shape=[0]), 'idxs':np.empty(shape=[0, 1])}
        #E, prob_1st, pred_1st, prob_2nd, pred_2nd, correct, uncertain = FetchInformativeSamples(M, T_hat, phi, theta, C)
        E = FetchInformativeSamples(M, T_hat, cfg.experiment.tol, cfg.experiment.m, C)
        
        f1_macro_T_k   = f1_score(y_true=T_tilde['y'], y_pred=M.predict_class(T_tilde['X']), average='macro')
        f1_macro_t_k   = f1_score(y_true=t['y'], y_pred=M.predict_class(t['X']), average='macro')

        mlflow.log_param('trial', cfg.experiment.seed)
        mlflow.log_param('K', cfg.experiment.subsets)
        mlflow.log_param('tol', cfg.experiment.tol)
        mlflow.log_param('fuzziness', cfg.experiment.m)
        mlflow.log_param('neurons', cfg.experiment.n_neurons)
        mlflow.log_metric('f1_BASE_Train', f1_macro_T)
        mlflow.log_metric('f1_BASE_test', f1_macro_t)
        mlflow.log_param('k_i', k)
        mlflow.log_metric('f1_BASE_Train_k', f1_macro_T_k)
        mlflow.log_metric('f1_BASE_test_k', f1_macro_t_k)
        mlflow.pytorch.log_model(M, artifact_path="model")
        mlflow.end_run()
        print(cfg.experiment.seed, cfg.experiment.subsets,cfg.experiment.tol,cfg.experiment.m,cfg.experiment.n_neurons)
        print(f1_macro_T,f1_macro_t)
        print(k,f1_macro_T_k,f1_macro_t_k )



if __name__ == '__main__':
    run_experiment()