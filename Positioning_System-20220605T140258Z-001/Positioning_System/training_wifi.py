import os 
import torch
import joblib
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from wifi import check_format, load, get_rps, get_boundary, normalize_rssi, normalize_pos, add_noise

torch.set_default_dtype(torch.double)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'dataset': 'IPIN2020_Track3_2F', #['NTU_CSIE_5F', 'DSI', 'IPIN2016_Tutorial', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F']
    'method': 'DSAR',#['RandomForest', 'WiDeep', 'DSAR']
}

if config['dataset'] == 'NTU_CSIE_5F':
    from Model.ntu_csie_5f import *
elif config['dataset'] == 'DSI':
    from Model.dsi import *
elif config['dataset'] == 'IPIN2016_Tutorial':
    from Model.ipin2016_tutorial import *
elif config['dataset'] == 'IPIN2020_Track3_2F':
    from Model.ipin2020_track3_2f import *
elif config['dataset'] == 'IPIN2020_Track3_3F':
    from Model.ipin2020_track3_3f import *
elif config['dataset'] == 'IPIN2020_Track3_5F':
    from Model.ipin2020_track3_5f import *



class RadioMap(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.from_numpy(inputs)
        self.labels = torch.from_numpy(labels)
        
    def __getitem__(self, index):
        return (self.inputs[index], self.labels[index])
        
    def __len__(self):
        return len(self.labels)



def RandomForest(dataset, training_wifi_pos):
    dir_path = osp.join(os.getcwd(), 'Result', dataset, 'RandomForest', 'randomforest.joblib')
    rps = get_rps(training_wifi_pos)
    inputs = training_wifi_pos[:,:-2]
    labels = training_wifi_pos[:,-2:]

    model = RandomForestRegressor(n_estimators=len(rps), criterion='mse')
    model.fit(inputs, labels)

    joblib.dump(model, dir_path, compress=0)



def WiDeep(dataset, training_wifi_pos, boundary):
    dir_path = osp.join(os.getcwd(), 'Result', dataset, 'WiDeep')

    rps = get_rps(training_wifi_pos)
    loaders = {}
    for rp_pos, rssi in rps.items():
        norm_rssi = normalize_rssi(rssi)
        pos = np.full((norm_rssi.shape[0],2), rp_pos, dtype=np.float64)
        norm_pos = normalize_pos(pos, boundary)
        radio_map = RadioMap(norm_rssi, norm_pos)
        loaders[rp_pos] = DataLoader(dataset=radio_map, shuffle=True, batch_size=4)

    input_size = training_wifi_pos.shape[1] - 2
    
    for rp_id, rp_pos in enumerate(loaders):

        loader = loaders[rp_pos]
        network = DenoisingAutoEncoder(input_size).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = torch.nn.MSELoss()

        network.train()
        for epoch in range(1000):
            losses = 0
            for inputs, labels in loader:
                
                noisy_inputs = add_noise(inputs).to(device)
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = network(noisy_inputs)

                loss = criterion(inputs, outputs)
                loss.backward()
                optimizer.step()
                losses += loss.item()

        print('='*50)
        print(f'rp id: {rp_id}, training loss: {losses}') 
        network_name = str(rp_pos) + '.pth'
        torch.save(network, osp.join(dir_path, network_name))




def DSAR(dataset, training_wifi_pos, boundary):
    dir_path = osp.join(os.getcwd(), 'Result', dataset, 'DSAR')
    rps = get_rps(training_wifi_pos)

    inputs = normalize_rssi(training_wifi_pos[:,:-2])
    labels = normalize_pos(training_wifi_pos[:,-2:], boundary)
    radio_map = RadioMap(inputs, labels)
    loader = DataLoader(dataset=radio_map, shuffle=True, batch_size=16)

    val_inputs = []
    val_labels = []
    for rp_pos in rps:
        val_inputs.append(np.average(rps[rp_pos], axis=0))
        val_labels.append(np.array(rp_pos, dtype=np.float64))

    val_inputs = normalize_rssi(np.array(val_inputs, dtype=np.float64))
    val_labels = normalize_pos(np.array(val_labels, dtype=np.float64), boundary)
    val_radio_map = RadioMap(val_inputs, val_labels)
    val_loader = DataLoader(dataset=val_radio_map, shuffle=True, batch_size=8)

    epochs = 10000
    input_size = training_wifi_pos.shape[1] - 2
    hidden_size = len(rps)

    network = AutoRegression(input_size, hidden_size).to(device)        
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = torch.nn.MSELoss()

    best_val_loss = np.inf
    losses = np.zeros((epochs,3), dtype=np.float64)

    network.train()
    for epoch in range(epochs):
        reconst_losses = 0.0
        regression_losses = 0.0

        for inputs, labels in loader:

            noisy_inputs = add_noise(inputs).to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            reconst_rssi, pos = network(noisy_inputs)

            reconst_loss = criterion(inputs, reconst_rssi)
            regression_loss = criterion(labels, pos)

            loss = reconst_loss + regression_loss
            loss.backward()
            optimizer.step()

            reconst_losses += reconst_loss.item()
            regression_losses += regression_loss.item()
        
        losses[epoch,0] = reconst_losses
        losses[epoch,1] = regression_losses

        network.eval()
        val_losses = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                reconst, pos = network(inputs)
                val_losses += (labels - pos).pow(2).sum()

        losses[epoch,2] = val_losses

        if val_losses < best_val_loss:
            best_val_loss = val_losses
            torch.save(network, osp.join(dir_path, 'dsar.pth'))

        print(f'Epoch {epoch}, reconstruction losses: {reconst_losses}, regression losses: {regression_losses}, validation losses: {val_losses}')

    np.save(osp.join(dir_path, 'losses.npy'), losses)



if __name__ == '__main__':

    check_format()

    dataset, method = config.values()

    training_wifi_pos, testing_wifi_pos = load(dataset)

    boundary = get_boundary(training_wifi_pos[:,-2:], testing_wifi_pos[:,-2:])

    if method == 'RandomForest':
        RandomForest(dataset, training_wifi_pos)

    
    elif method == 'WiDeep':
        WiDeep(dataset, training_wifi_pos, boundary)
    

    elif method == 'DSAR':
        DSAR(dataset, training_wifi_pos, boundary)