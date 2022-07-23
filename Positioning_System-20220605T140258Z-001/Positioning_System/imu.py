import os
import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from Model.ronin_lstm import BilinearLSTMSeqNetwork

torch.set_default_dtype(torch.double)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(dataset):
    dir_path = osp.join(os.getcwd(), 'Dataset', dataset)
    glob_acce = np.load(osp.join(dir_path, 'glob_acce.npy'))
    glob_gyro = np.load(osp.join(dir_path, 'glob_gyro.npy'))
    dt = np.load(osp.join(dir_path, 'dt.npy'))
    wifi_pos = np.load(osp.join(dir_path, 'wifi_pos.npy'))
    wifi_pos_index = np.load(osp.join(dir_path, 'wifi_pos_index.npy'))

    return glob_acce, glob_gyro, dt, wifi_pos, wifi_pos_index



def lstm_inference(glob_acce, glob_gyro, dt):
    inputs = np.c_[glob_gyro, glob_acce]
    inputs = gaussian_filter1d(inputs, sigma=0.001, axis=0, mode='nearest')

    checkpoint_path = osp.join(os.getcwd(), 'Model', 'ronin_lstm_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
    network = BilinearLSTMSeqNetwork(6, 2, 1, device, lstm_layers=3, lstm_size=100).to(device)
    network.load_state_dict(checkpoint.get('model_state_dict'))

    network.eval()
    with torch.no_grad():
        inputs = torch.tensor(inputs[None,:,:], dtype=torch.double).to(device)
        vx_vy = np.squeeze(network(inputs).cpu().numpy())
    
    offset = vx_vy * dt
    return offset


def rotation(offset, degree):
    s = np.sin(degree/180*np.pi)
    c = np.cos(degree/180*np.pi)
    r_m = [[c, -s], [s, c]]
    rotated_offset = r_m @ offset.T
    return rotated_offset.T


def plot_trajectory(dataset, offset, gt, gt_index):

    dir_path = osp.join(os.getcwd(), 'Result', dataset, 'RoNIN')
    np.save(osp.join(dir_path, 'offset.npy'), offset)
    
    init_pos = gt[None,0,:]
    imu = np.r_[init_pos, offset]
    imu_pos = np.cumsum(imu, axis=0)
    
    localization_errors = np.linalg.norm(imu_pos[gt_index] - gt, axis=1)
    avg_error = format(np.average(localization_errors), '.4f')

    plt.plot(gt[:,0], gt[:,1], 'og', label='ground truth')
    plt.plot(imu_pos[:,0], imu_pos[:,1], label='RoNIN track')

    for i, index in enumerate(wifi_pos_index):
        point_x = np.r_[gt[i,0], imu_pos[index,0]]
        point_y = np.r_[gt[i,1], imu_pos[index,1]]
        plt.plot(point_x, point_y, '--k')

    plt.title(f'RoNIN LSTM Trajectory\n Avg error({avg_error})m')                               
    plt.xlabel('x (m)')                                                   
    plt.ylabel('y (m)')     
    plt.legend(bbox_to_anchor=(1, 0.65), loc='upper left')
    plt.grid()
    plt.savefig(osp.join(dir_path, 'ronin.png'), bbox_inches='tight')
    plt.close()
    


    
if __name__ == '__main__':

    datasets = ['NTU_CSIE_5F', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F']
    
    for dataset in datasets:

        glob_acce, glob_gyro, dt, wifi_pos, wifi_pos_index = load(dataset)
        
        offset = lstm_inference(glob_acce, glob_gyro, dt)

        if dataset == 'NTU_CSIE_5F':
            offset = rotation(offset, -105)
        else:
            offset = rotation(offset,-5)
        
        plot_trajectory(dataset, offset, wifi_pos[:,-2:], wifi_pos_index)

