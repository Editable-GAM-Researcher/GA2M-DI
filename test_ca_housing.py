import torch
from GA2M_DI import *
import torch.utils.data as Data
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from interpret.glassbox import ExplainableBoostingRegressor


def generate_dataset_new(X,label):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return torch.from_numpy(X).to(device), torch.from_numpy(label).to(device)


class Normalizer:
    def __init__(self):
        self.mean_x = 0
        self.std_x = 1
        self.mean_y = 0
        self.std_y = 1
        self.trans = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
    def fit(self,x,y):
        self.trans = self.trans.fit(x)
        self.mean_y, self.std_y = np.mean(y),np.std(y)
        if self.std_y < 1e-10:
            self.std_y = 1
    
    def transform(self,x,y,normalize_y=True):
        if normalize_y:
            return self.trans.transform(x),(y-self.mean_y)/self.std_y
        else:
            return self.trans.transform(x),y


def train(file_name,X_train,X_test,y_train,y_test,batch_size=256):

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    N_train = X_train.shape[0]
    N_val = X_val.shape[0]
    N_test = X_test.shape[0]


    training_input, training_target = generate_dataset_new(X_train, y_train)
    val_input, val_target = generate_dataset_new(X_val, y_val)
    test_input, test_target = generate_dataset_new(X_test, y_test)

    torch_dataset_train = Data.TensorDataset(training_input, training_target)
    torch_dataset_val = Data.TensorDataset(val_input, val_target)
    torch_dataset_test = Data.TensorDataset(test_input, test_target)

    train_loader = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True)
    val_loader=Data.DataLoader(dataset=torch_dataset_val, batch_size=batch_size*100, shuffle=False)
    test_loader=Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size*100, shuffle=False)
    n_epochs = 300
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = GA2M(feature_num=X_train.shape[1],layerSizeList=[16,64,64,64],basis_num=128).to(device)
    optimizer =torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    lowest_val_loss = 1e+15
    report_test_loss = 1e+15
    paient = 0
    eps = 1
    test_model_name = {}
    for epoch in range(n_epochs):
        train_loss = 0.0
        eps = eps*0.9
        for step,(X_batch,target_batch) in enumerate(train_loader):
            X_batch.requires_grad_()
            y_hat = model(X_batch)
            fit_loss = torch.mean((target_batch - y_hat.flatten()) ** 2)
            reg_1 = torch.mean(torch.log(torch.mean(F.softmax(model.first_order_attention_weights,dim=-1)[:,:,1:],dim=-1)+eps),dim=-1)
            reg_2 = torch.mean(torch.log(torch.sum(F.softmax(model.second_order_attention_weights,dim=-1)[:,:,1:],dim=-1)+eps),dim=-1)
            reg_3 = torch.mean(torch.mean(torch.mean(model.latent**2,dim=-1),dim=-1),dim=-1)
            loss = fit_loss + 1e-3*reg_1 + 1e-2*reg_2 + 1e-2*reg_3
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_loss += fit_loss.item()*X_batch.shape[0]

        train_loss = train_loss / N_train

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            for step, (X_batch, y_batch) in enumerate(val_loader):
                output = model(X_batch)
                loss = torch.sum((output.flatten()- y_batch.flatten())**2)
                epoch_val_loss += loss.detach().cpu().numpy()
            val_loss = epoch_val_loss/N_val
            
            epoch_test_loss = 0.0
            for step, (X_batch, y_batch) in enumerate(test_loader):
                output = model(X_batch)
                loss = torch.sum((output.flatten()- y_batch.flatten())**2)
                epoch_test_loss += loss.detach().cpu().numpy()
            if len(test_model_name) < 1 and (epoch>50):
                PATH = './models/'+file_name+'_'+str(epoch)+ "_model.pt"
                test_model_name[PATH] = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    }, PATH)
            elif (epoch>50) and val_loss < max(test_model_name.values()):
                PATH = './models/'+file_name+'_'+str(epoch)+ "_model.pt"
                test_model_name[PATH] = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    }, PATH)
            
            if len(test_model_name) > 5:
                remove_keys = []
                threshold = sorted(test_model_name.values())[5]
                for key,value in test_model_name.items():
                    if value > threshold:
                        remove_keys.append(key)
                for key in remove_keys:
                    test_model_name.pop(key)

            test_loss = epoch_test_loss/N_test
            if (val_loss < lowest_val_loss) and (epoch>50):
                lowest_val_loss = val_loss
                report_test_loss = test_loss
                paient = 0
            else:
                paient += 1
            if epoch % 10 == 0:
                print(f'Epoch: {epoch+1} Train mse = {np.round(train_loss,4)}, valid mse = {np.round(val_loss,4)}, test mse = {np.round(test_loss,4)}, report mse = {np.round(report_test_loss,4)}')

    test_pred = 0
    count = 0
    for key in test_model_name.keys():
        checkpoint = torch.load(key)
        model.load_state_dict(checkpoint['model_state_dict'])
        count += 1
        test_pred += model(test_input)
    loss = torch.mean((test_pred.flatten()/count- test_target.flatten())**2).detach().cpu().numpy()
    print(f'Final testing error is: test mse = {np.round(loss,6)}')    

    return loss



def test_gam(file):
    df = pd.read_csv('./datasets/'+file+'.csv')
    df.pop('ocean_proximity')
    df.dropna(axis=0, inplace=True)
    y = df.pop('median_house_value').values.astype('float32')/100000
    X = df.values.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    normalizer = Normalizer()
    normalizer.fit(X_train,y_train)
    X_train,y_train = normalizer.transform(X_train,y_train)
    X_test,y_test = normalizer.transform(X_test,y_test)
    report_test_loss = train(file,X_train,X_test,y_train,y_test,
                            batch_size=256)
    print(f'Testing mse of proposed method is {report_test_loss}')
    ebm = ExplainableBoostingRegressor(max_rounds=1000000,
                                        validation_size=0.2,
                                        max_interaction_bins=256)
    ebm.fit(X_train, y_train)
    y_predict = ebm.predict(X_test)
    test_MSE = np.mean((y_predict-y_test)**2)
    print(f'Testing mse of EBM is {test_MSE}')
    return report_test_loss
    


if __name__ == '__main__':
    file = 'ca_housing'
    test_loss = test_gam(file=file)
        
