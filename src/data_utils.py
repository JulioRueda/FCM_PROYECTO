import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import time 
import copy 


def load_dataset(DATASET_ID, DATASET_PATH):
  if DATASET_ID == "mnist":
    mnist = pd.read_csv(DATASET_PATH, header=None)
    mat_I = mnist.drop(columns=[0], axis=1, inplace=False)
    mat_gt= mnist[[0]] + 1
  elif DATASET_ID == "fashion":
    fashion = pd.read_csv(DATASET_PATH, header=None)
    mat_I = fashion.drop(columns=[0], axis=1, inplace=False)
    mat_gt= fashion[[0]] + 1
  elif DATASET_ID == "cifar10":
    cifar10 = pd.read_csv(DATASET_PATH, header=None)
    mat_I = cifar10.drop(columns=[0], axis=1, inplace=False)
    mat_gt= cifar10[[0]] + 1

  I     = []  #Hyperspectral input image

  match DATASET_ID:
    case "pavia":
      I = mat_I["paviaU"].copy()
      gt= mat_gt["paviaU_gt"].copy()
    case "botswana":
      I = mat_I["Botswana"].copy()
      gt= mat_gt["Botswana_gt"].copy()
    case "indian":
      I = mat_I["indian_pines_corrected"].copy()
      gt= mat_gt["indian_pines_gt"].copy()
    case "ksc":
      I = mat_I["KSC"].copy()
      gt= mat_gt["KSC_gt"].copy()
    case "salinas":
      I = mat_I["salinas_corrected"].copy()
      gt= mat_gt["salinas_gt"].copy()
    case "houston":
      I = mat_I['ori_data'].copy()
      gt= mat_gt['map'].copy()
    case "mnist":
      I = mat_I.to_numpy()
      gt= mat_gt.to_numpy()
    case "fashion":
      I = mat_I.to_numpy()
      gt= mat_gt.to_numpy()
    case "cifar10":
      I = mat_I.to_numpy()
      gt= mat_gt.to_numpy() 
    case _:
      print("Unrecognized dataset")

  # Normalize input image
  I     = I / I.max()
  gt    = (1.0*gt - 1.0) #0->-1; 1->0; 2->1;...;9->8
  N     = I.shape[0] * I.shape[1]
  return I,gt

def trainTestValidationSplit(_data, _gt, _train, _val, _test):
  # Obtain input image properties
  N   = _data.shape[0] #N: number of observations
  M   = _data.shape[1] #M: number of reatures
  B   = 1              #B: bands
  N_p = N * B          #N_p: total number of observations

  # Pixel-wise indexing
  pxl_cols = np.tile(np.arange(N), (M,1))
  pxl_rows = np.tile(np.arange(M), (N,1)).T
  pxl_indxs= N * pxl_rows + pxl_cols #pixel-based indexes

  Indexes  = np.dstack((pxl_rows, pxl_cols, pxl_indxs)) # indexing image

  # Rearrange input image I into L
  _I       = _data.copy()
  _GT      = _gt.copy()
  _Indexes = np.reshape(np.arange(N_p), newshape=(N_p,1), order='C')

  # Select non zero-valued pixels
  mask_zero= np.ravel(_GT == -1)

  I_data   = _I[~mask_zero,:]
  GT       = _GT[~mask_zero,:].ravel()
  Indexes  = _Indexes[~mask_zero,:]

  classes  = np.unique(GT)
  N_classes= classes.shape[0]

  # Generate train (T) and test (t) subsets
  N_obs    = I_data.shape[0]
  N_T      = int(_train * N_obs)
  N_v      = int(_val * N_obs)
  N_t      = N - N_T - N_v

  #np.random.seed(42)  #For reproducibility
  while True:
    idxs   = np.arange(N_obs)
    T_idxs = np.random.choice(idxs, size=N_T, replace=False, p=None)
    _idxs  = np.setdiff1d(idxs, T_idxs) #T_idxs complements

    v_idxs = np.random.choice(_idxs, size=N_v, replace=False, p=None)
    t_idxs = np.setdiff1d(_idxs, v_idxs)

    if np.unique(GT[T_idxs]).shape[0] == N_classes & np.unique(GT[v_idxs]).shape[0] == N_classes & np.unique(GT[t_idxs]).shape[0] == N_classes:
      break
    #END_IF
  #END_WHILE

  if np.isin(T_idxs, t_idxs).any():
    print("Warning: Training and testing sets overlap")
  elif np.isin(T_idxs, v_idxs).any():
    print("Warning: Training and validation sets overlap")
  elif np.isin(v_idxs, t_idxs).any():
    print("Warning: Testing and validation sets overlap")
  #END_IF

  T        = {'X':I_data[T_idxs,:], 'y':GT[T_idxs], 'idxs':Indexes[T_idxs,:]}
  t        = {'X':I_data[t_idxs,:], 'y':GT[t_idxs], 'idxs':Indexes[t_idxs,:]}
  V        = {'X':I_data[v_idxs,:], 'y':GT[v_idxs], 'idxs':Indexes[v_idxs,:]}

  return T, V, t, classes

def loadTrainTestValidation(_data, _GT, base, dataset_name, _trial):
  # Obtain input image properties
  N   = _data.shape[0] #N: number of observations
  M   = _data.shape[1] #M: number of reatures
  B   = 1              #B: bands
  N_p = N * B          #N_p: total number of observations

  # Rearrange input image I into L
  I  = _data.copy()
  GT = _GT.copy()

  # Load indexdes
  T_idxs = pd.read_csv(rf"{base}\{dataset_name}_Train_Trial_{_trial}.csv").to_numpy()
  V_idxs = pd.read_csv(rf"{base}\{dataset_name}_Val_Trial_{_trial}.csv").to_numpy()
  t_idxs = pd.read_csv(rf"{base}\{dataset_name}_Test_Trial_{_trial}.csv").to_numpy()

  #
  T = {'X':I[T_idxs[:,-1],:], 'y':GT[T_idxs[:,-1]].ravel(), 'idxs':T_idxs}
  V = {'X':I[V_idxs[:,-1],:], 'y':GT[V_idxs[:,-1]].ravel(), 'idxs':V_idxs}
  t = {'X':I[t_idxs[:,-1],:], 'y':GT[t_idxs[:,-1]].ravel(), 'idxs':t_idxs}

  classes = np.unique(T['y'])

  return T, V, t, classes 

class EarlyStopping:
  def __init__(self, patience=15, min_delta=0.0, restore_best_weights=True):
    self.patience = patience
    self.min_delta= min_delta
    self.restore_best_weights = restore_best_weights
    self.best_model = None
    self.best_train_loss = None
    self.best_val_loss = None
    self.best_epoch = 0
    self.last_epoch = 0
    self.counter = 0
    self.status = ""

  def __call__(self, model, epoch, val_loss, train_loss=-1.0):
    self.last_epoch     = epoch

    if self.best_val_loss is None:
      self.best_train_loss= train_loss
      self.best_val_loss  = val_loss
      self.best_model     = copy.deepcopy(model.state_dict())
    elif self.best_val_loss - val_loss >= self.min_delta:
      self.best_train_loss= train_loss
      self.best_val_loss  = val_loss
      self.best_model     = copy.deepcopy(model.state_dict())
      self.best_epoch     = epoch
      self.counter        = 0
      self.status         = f"Improvement found, counter reset to {self.counter}"
    else:
      self.counter       += 1
      self.status         = f"No improvement in the last {self.counter} epochs."
      if self.counter >= self.patience:
        self.status = f"Early stopping triggered after {self.counter} epochs."
        if self.restore_best_weights:
          model.load_state_dict(self.best_model)
          self.status += f" Restoring model weights from epoch {self.best_epoch}."
        return True
    return False

class MLP(nn.Module):
  def __init__(self, _n_features=1, _n_hidden_1=400, _n_classes=1):
    super().__init__()

    self.layers = nn.Sequential(
        nn.Linear(in_features=_n_features, out_features=_n_hidden_1, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=_n_hidden_1, out_features=_n_classes, bias=True),
    )

    self.initialize_weights()

    self.loss_logs = {'batch_loss':[], 'train_loss':[], 'val_loss':[]}
  #___INIT___

  def forward(self, _X):
    return self.layers(_X)
  #_FORWARD_

  def initialize_weights(self):
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        nn.init.constant_(module.bias, 0)
  #_INITIALIZE_WEIGHTS_

  def compile(self,  _device, _loss_fn, _optimizer):
    self.device     = _device
    self.loss_fn    = _loss_fn
    self.optimizer  = _optimizer
  #_COMPILE_

  def train_one_epoch(self, _T_data_loader):
    total_T_samples = 0
    total_T_loss    = 0.0
    avg_T_loss      = 0.0

    BATCH_T_LOSSES  = []

    self.train(True)
    for i_batch, T_data in enumerate(_T_data_loader, 0):
      T_inputs, T_labels = T_data
      T_inputs, T_labels = T_inputs.to(self.device), T_labels.to(self.device)

      self.optimizer.zero_grad()

      T_outputs = self(T_inputs)

      T_loss    = self.loss_fn(T_outputs, T_labels)
      T_loss.backward()

      self.optimizer.step()

      BATCH_T_LOSSES  += [T_loss.item()]
      total_T_loss    += T_loss.item() * T_inputs.size(0)
      total_T_samples += T_inputs.size(0)
    #FOR_BATCH_TRAINING_

    avg_T_loss = total_T_loss / total_T_samples

    return avg_T_loss, BATCH_T_LOSSES
  #_TRAIN_ONE_EPOCH_

  #def fit(self, _T_data_loader=None, _V_data_loader=None, _epochs=100):
  def fit(self, _X_T=None, _y_T=None, _validation_data=(None, None), _validation_batch_size=128, _batch_size=128, _shuffle=True, _epochs=100, _early_stop=None, _verbose=True):
    T_data        = torch.utils.data.TensorDataset(torch.FloatTensor(_X_T), torch.LongTensor(_y_T))
    T_data_loader = torch.utils.data.DataLoader(T_data, batch_size=_batch_size, shuffle=_shuffle, generator=torch.Generator())

    self.loss_logs['batch_loss'] = []
    self.loss_logs['train_loss'] = []
    self.loss_logs['val_loss']   = []

    avg_T_loss    = 0
    avg_V_loss    = 0

    for epoch in range(1, _epochs + 1):
      avg_T_loss, batch_T_losses = self.train_one_epoch(T_data_loader)
      avg_V_loss = self.evaluate(_X=_validation_data[0], _Y=_validation_data[1], _batch_size=_validation_data[1].shape[0])

      self.loss_logs['batch_loss'] += batch_T_losses
      self.loss_logs['train_loss'] += [avg_T_loss]
      self.loss_logs['val_loss']   += [avg_V_loss]

      #print(f"Epoch: {epoch:>02}\tTrain loss: {avg_T_loss:.6f}\tValidation loss: {avg_V_loss:.6f}")

      if _early_stop(self, epoch, avg_V_loss, avg_T_loss):
        print(f"Epoch: {epoch:>02}\tTrain loss: {avg_T_loss:.6f}\tValidation loss: {avg_V_loss:.6f}")
        print(_early_stop.status)
        break
    #FOR_EPOCHS

    return self.loss_logs
  #_TRAIN_

  #def evaluate(self, _data_loader):
  def evaluate(self, _X, _Y, _shuffle=True, _batch_size=128):
    data        = torch.utils.data.TensorDataset(torch.FloatTensor(_X), torch.LongTensor(_Y))
    data_loader = torch.utils.data.DataLoader(data, batch_size=_batch_size, shuffle=_shuffle, generator=torch.Generator())

    total_samples = 0
    total_loss    = 0.0
    avg_loss      = 0.0

    self.eval()
    with torch.no_grad():
      for i_batch, data in enumerate(data_loader):
        inputs, labels= data
        inputs, labels= inputs.to(self.device), labels.to(self.device)
        outputs       = self(inputs)
        loss          = self.loss_fn(outputs, labels)

        total_loss   += loss.item() * inputs.size(0)
        total_samples+= inputs.size(0)
      #FOR_EVALUATE
    avg_loss = total_loss / total_samples

    return avg_loss
  #_EVALUATE_

  def predict(self, _X):
    X       = torch.FloatTensor(_X)

    self.eval()
    with torch.no_grad():
      y_probs = torch.softmax(self(X.to(self.device)), dim=1)

    return y_probs.cpu().detach().numpy()
  #_PREDICT_

  def predict_class(self, _X):
    y_probs= self.predict(_X)
    y_pred = np.argmax(y_probs, axis=1)

    return y_pred
  #_PREDICT_CLASS_

  def predict_top_k(self, _X, _k=1):
    y_probs = self.predict(_X)

    y_preds = np.argsort(y_probs, axis=1)
    y_probs = np.sort(y_probs, axis=1)

    return y_probs[:,-_k:], y_preds[:,-_k:]

def FetchInformativeSamples(_M, _T_hat, _phi, _theta, _C):
    centroids = []
    X = pd.DataFrame(_T_hat['X'])
    Y = pd.DataFrame(_T_hat['y'])

    #calculate centroids con promedio simple de los xj de cada observación por cada clase
    for c in set(Y[0]): # Iterate over unique values in the first column of Y
        c_i = np.array(X.loc[Y[0] == c].mean()) # Use boolean indexing with .loc
        centroids.append(c_i)
    M = []
    m = _theta # Assuming m=2 as a common value in FCM, adjust if needed

    #calculate fuzzy matrix M
    for i in range(len(centroids)):
        fil = []
        vi = centroids[i]
        #formula to calculate degree of membership
        for k in X.values:
            xk = k
            # Add a small epsilon to the denominator to avoid division by zero
            denominator = sum([(np.linalg.norm(vi-xk)/(np.linalg.norm(vj - xk) + 1e-8))**(2/(m - 1)) for vj in centroids])
            mu_ik = 1/(denominator + 1e-8) # Add epsilon here as well
            fil.append(mu_ik)
        M.append(fil)

    #Calculate degree of membership difference
    absolute_diff = []
    for row in np.array(M).T:
      absolute_diff.append(np.abs(np.sort(row)[-2:][0] - np.sort(row)[-2:][1]))
    #query selection
    X__ = X.copy()
    X__['decision'] = [dif_i<_phi for dif_i in absolute_diff]

    X_salida = X__.loc[X__['decision'] == True].drop(columns='decision')
    Y_salida = Y.loc[X_salida.index] # Select corresponding rows from Y

    E = {'X':X_salida.to_numpy(), 'y':Y_salida.to_numpy().ravel(), 'idxs':_T_hat['idxs'][X_salida.index]} # Select corresponding rows from idxs and ravel y
    return E

def FetchInformativeProba(_M, _T_hat, _phi, _theta, _C):
  y_preds = _M.predict(_T_hat['X'])
  absolute_diff = []
  for row in y_preds:
    absolute_diff.append(np.abs(np.sort(row)[-2:][0] - np.sort(row)[-2:][1]))

  X = pd.DataFrame(_T_hat['X'])
  Y = pd.DataFrame(_T_hat['y'])
  X__ = X.copy()
  X__['decision'] = [dif_i<_phi for dif_i in absolute_diff]

  X_salida = X__.loc[X__['decision'] == True].drop(columns='decision')
  Y_salida = Y.loc[X_salida.index] # Select corresponding rows from Y

  E = {'X':X_salida.to_numpy(), 'y':Y_salida.to_numpy().ravel(), 'idxs':_T_hat['idxs'][X_salida.index]} # Select corresponding rows from idxs and ravel y
  return E

def getPartition(_T, _K, _random_state=None):
  if _random_state != None:
    np.random.seed(_random_state)

  N_T = _T['X'].shape[0]  #number of observations in _T
  n   = N_T // _K         #number of observations in the training subsets T_k

  train_idxs  = np.arange(N_T)
  train_idxs  = np.random.choice(train_idxs, size=N_T, replace=False, p=None)

  lower_bounds= list(range(0, N_T, n))
  upper_bounds= list(range(n - 1, N_T, n))

  P = {}
  for k in range(_K):
    if k < _K - 1:
      k_idxs = train_idxs[lower_bounds[k]:upper_bounds[k]]
    else:
      k_idxs = train_idxs[lower_bounds[k]:]
    #END_IF_K

    P[k] = {'X':_T['X'][k_idxs,:], 'y':_T['y'][k_idxs], 'idxs':_T['idxs'][k_idxs,:]}
  #END_FOR

  return P 

def AdjustNeuralNetworkWeights(_M, _T_tilde, _V):
  #init_W = _M.layers[1].kernel.numpy().copy()
  #early_callback= tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20, restore_best_weights=True)

  if _T_tilde['X'].shape[0] == 0:
    return _M, 0.0

  start  = time.process_time_ns()

  history= _M.fit(_T_tilde['X'], _T_tilde['y'],
                  _batch_size=128,
                  _shuffle=True,
                  _epochs=2000,
                  _verbose=False,
                  _validation_data=(_V['X'], _V['y']),
                  _validation_batch_size=_V['y'].shape[0],
                  _early_stop=EarlyStopping(patience=20)
                 )

  end    = time.process_time_ns()

  elapsed= (end - start)*1e-9

  #final_W= _M.layers[1].kernel.numpy().copy()

  #if np.any(final_W - init_W) == False:
  #    print("\n\n********** PESOS IGUALES *********\n\n")

  return _M, elapsed