import os 
import torch
import heapq
import joblib
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.double)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'dataset': 'IPIN2020_Track3_5F', #['NTU_CSIE_5F', 'DSI', 'IPIN2016_Tutorial', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F']
    'method': 'DSAR',#['WKNN', 'RandomForest', 'WiDeep', 'DSAR']
}

def check_format():
    assert config['method'] in ['WKNN', 'RandomForest', 'WiDeep', 'DSAR'], \
        'config method should be [WKNN, RandomForest, WiDeep, DSAR]'
    assert config['dataset'] in ['NTU_CSIE_5F', 'DSI', 'IPIN2016_Tutorial', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F'], \
        'config dataset should be [NTU_CSIE_5F, DSI, IPIN2016_Tutorial, IPIN2020_Track3_2F, IPIN2020_Track3_3F, IPIN2020_Track3_5F]'



def load(dataset):
    dir_path = osp.join(os.getcwd(), 'Dataset', dataset)

    training_wifi_pos = np.load(osp.join(dir_path, 'training_wifi_pos.npy'))
    testing_wifi_pos = np.load(osp.join(dir_path, 'testing_wifi_pos.npy'))

    return training_wifi_pos, testing_wifi_pos



def normalize_rssi(rssi):
    return (rssi + 100) / 100



def normalize_pos(pos, boundary):
    max_x, min_x, max_y, min_y = boundary
    norm_pos = np.zeros_like(pos)
    norm_pos[:,0] = (pos[:,0] - min_x) / (max_x - min_x)
    norm_pos[:,1] = (pos[:,1] - min_y) / (max_y - min_y)
    return norm_pos



def restore_pos(norm_pos, boundary):
    max_x, min_x, max_y, min_y = boundary
    pos = np.zeros_like(norm_pos)
    pos[:,0] = norm_pos[:,0] * (max_x - min_x) + min_x
    pos[:,1] = norm_pos[:,1] * (max_y - min_y) + min_y
    return pos



def get_rps(training_wifi_pos):

    rps = {}
    for i in range(training_wifi_pos.shape[0]):
        pos = tuple(training_wifi_pos[i,-2:])
        if pos not in rps:
            rps[pos] = training_wifi_pos[None,i,:-2]
        else:
            rps[pos] = np.r_[rps[pos], training_wifi_pos[None,i,:-2]]
    return rps
        


def add_noise(rssi):
    noisy_rssi = rssi + torch.normal(0, 0.1, rssi.shape)
    noisy_rssi = torch.clip(noisy_rssi, 0, 1)
    mask = torch.rand(rssi.shape) < 0.1
    noisy_rssi[mask] = 0
    return noisy_rssi
    


def get_boundary(training_pos, testing_pos):
    max_x = max(training_pos[:,0].max(), testing_pos[:,0].max())
    min_x = min(training_pos[:,0].min(), testing_pos[:,0].min())
    max_y = max(training_pos[:,1].max(), testing_pos[:,1].max())
    min_y = min(training_pos[:,1].min(), testing_pos[:,1].min())
    
    return (max_x, min_x, max_y, min_y)



def RBF(origin, reconstruction, gamma=16, axis=None):
    return np.e**(-gamma * np.linalg.norm(origin - reconstruction, axis=axis)**2)



def plot_wifi_pos(dataset, method, training_pos, testing_pos, pred_pos, pred_err):

    dir_path = osp.join(os.getcwd(), 'Result', dataset, method)

    fig, (axs0, axs1, axs2) = plt.subplots(1, 3, figsize=(6,4), sharex=True, sharey=True)

    axs0.plot(training_pos[:,0], training_pos[:,1], 'ob')
    axs0.set_title('Reference Points')
    axs0.grid()

    axs1.plot(testing_pos[:,0], testing_pos[:,1], 'og')
    axs1.set_title('Testing Points')
    axs1.grid()

    axs2.plot(pred_pos[:,0], pred_pos[:,1], 'or')
    axs2.set_title('Prediction Points')
    axs2.grid()

    fig.tight_layout()
    plt.savefig(osp.join(dir_path, dataset), bbox_inches='tight')
    np.save(osp.join(dir_path, 'loc_err.npy'), pred_err)
    np.save(osp.join(dir_path, 'pred_pos.npy'), pred_pos)



def WKNN(query_wifi_pos, rps_, k=3):

    eps = 1e-6
    pred_pos = np.zeros((query_wifi_pos.shape[0],2), dtype=np.float64)
    loc_err = np.zeros(query_wifi_pos.shape[0], dtype=np.float64)
    
    rps = {}
    for rp_pos in rps_:
        rps[rp_pos] = np.average(rps_[rp_pos], axis=0)
    
    for i in range(query_wifi_pos.shape[0]):
        query_rssi = query_wifi_pos[i,:-2]
        gt_pos = query_wifi_pos[i,-2:]

        min_heap = []
        for rp_pos in rps:
            rssi_dis = np.linalg.norm(query_rssi - rps[rp_pos])
            heapq.heappush(min_heap, (rssi_dis, rp_pos))
        
        k_small = heapq.nsmallest(k, min_heap)
        denominator = 0
        for rssi_dis, rp_pos in k_small:
            denominator += 1 / (rssi_dis+eps)**2

        
        for rssi_dis, rp_pos in k_small:
            rp_pos = np.array(rp_pos, dtype=np.float64)
            pred_pos[i] += 1 / (rssi_dis+eps)**2 / denominator * rp_pos

        loc_err[i] = np.linalg.norm(pred_pos[i] - gt_pos)
                
    return pred_pos, loc_err



def RandomForest(query_wifi_pos, dataset):
    model_path = osp.join(os.getcwd(), 'Result', dataset, 'RandomForest', 'randomforest.joblib')
    assert osp.exists(model_path), 'please train the random forest before using it'

    model = joblib.load(model_path)

    inputs = query_wifi_pos[:,:-2]
    labels = query_wifi_pos[:,-2:]

    pred_pos = model.predict(inputs)
    loc_err = np.linalg.norm(pred_pos - labels, axis=1)

    return pred_pos, loc_err



def WiDeep(query_wifi_pos, rps, dataset):
    dir_path = osp.join(os.getcwd(), 'Result', dataset, 'WiDeep')

    pred_pos = np.zeros((query_wifi_pos.shape[0],2), dtype=np.float64)
    loc_err = np.zeros(query_wifi_pos.shape[0], dtype=np.float64)

    networks = {}
    for rp_pos in rps:
        network_name = str(rp_pos) + '.pth'
        path = osp.join(dir_path, network_name)
        networks[rp_pos] = torch.load(path)

    inputs = normalize_rssi(query_wifi_pos[:,:-2])
    labels = query_wifi_pos[:,-2:]
    rps_pos = np.array(list(networks.keys()), dtype=np.float64)

    for i in range(inputs.shape[0]):
        rp_prob = np.zeros(len(rps), dtype=np.float64)
        query_rssi = torch.from_numpy(inputs[None,i,:]).to(device)

        for rp_id, rp_pos in enumerate(networks):
            network = networks[rp_pos].to(device)
            network.eval()
            with torch.no_grad():
                reconst_rssi = network(query_rssi)
                rp_prob[rp_id] = RBF(inputs[i], reconst_rssi.cpu().numpy())

        pred_pos[i] = np.average(rps_pos, axis=0, weights=rp_prob)
        loc_err[i] = np.linalg.norm(labels[i] - pred_pos[i])
    
    return pred_pos, loc_err



def DSAR(query_wifi_pos, dataset, boundary):

    dir_path = osp.join(os.getcwd(), 'Result', dataset, 'DSAR')
    inputs = normalize_rssi(query_wifi_pos[:,:-2])
    labels = query_wifi_pos[:,-2:]

    path = osp.join(dir_path, 'dsar.pth') 
    network = torch.load(path).to(device) # if u don't have gpu, add map_location='cpu' in load's parameter
    network.eval()

    with torch.no_grad():
        inputs = torch.from_numpy(inputs).to(device)
        rssi_reconst, pos = network(inputs)
        rssi_reconst = rssi_reconst.cpu().numpy()
        pred_pos = restore_pos(pos.cpu().numpy(), boundary)
        loc_err = np.linalg.norm(pred_pos - labels, axis=1)

    return pred_pos, loc_err, rssi_reconst





if __name__ == '__main__':
    
    check_format()

    dataset, method = config.values()

    training_wifi_pos, testing_wifi_pos = load(dataset)

    boundary = get_boundary(training_wifi_pos[:,-2:], testing_wifi_pos[:,-2:])

    rps = get_rps(training_wifi_pos)
    
    if method == 'WKNN':

        testing_pos, testing_err = WKNN(testing_wifi_pos, rps)

    elif method == 'RandomForest':

        testing_pos, testing_err = RandomForest(testing_wifi_pos, dataset)

    elif method == 'WiDeep':

        testing_pos, testing_err = WiDeep(testing_wifi_pos, rps, dataset)
    
    elif method == 'DSAR':

        testing_pos, testing_err, rssi_reconst = DSAR(testing_wifi_pos, dataset, boundary)


    plot_wifi_pos(dataset, method, training_wifi_pos[:,-2:], testing_wifi_pos[:,-2:], testing_pos, testing_err)