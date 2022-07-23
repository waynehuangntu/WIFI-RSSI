import os
import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from wifi import load, get_boundary, normalize_rssi, DSAR, RBF


torch.set_default_dtype(torch.double)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_similarity_loc_err(dataset):

    training_wifi_pos, testing_wifi_pos = load(dataset)

    boundary = get_boundary(training_wifi_pos[:,-2:], testing_wifi_pos[:,-2:])

    _, training_err, training_rssi_reconst = DSAR(training_wifi_pos, dataset, boundary)
    
    gamma = np.mean(training_err)
    
    training_similarity = RBF(normalize_rssi(training_wifi_pos[:,:-2]), training_rssi_reconst, gamma, axis=1)

    _, testing_err, testing_rssi_reconst = DSAR(testing_wifi_pos, dataset, boundary)
    testing_similarity = RBF(normalize_rssi(testing_wifi_pos[:,:-2]), testing_rssi_reconst, gamma, axis=1)

    similarity = np.r_[training_similarity, testing_similarity]
    loc_err = np.r_[training_err, testing_err]
    my_rho = np.corrcoef(similarity, loc_err)

    print(dataset, my_rho[0,1])

    return training_similarity, training_err, testing_similarity, testing_err


def plot_similarity(datasets):
    
    titles = ['(a) NTU_CSIE_5F', '(b) IPIN2020_Track3_2F', '(c) IPIN2020_Track3_3F', '(d) IPIN2020_Track3_5F']
    
    dataset_err = {}
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))

    for i in range(len(datasets)):

        training_similarity, training_err, testing_similarity, testing_err = get_similarity_loc_err(datasets[i])

        dataset_err[datasets[i]] = training_err
        
        r, c = i // 2, i % 2

        axs[r,c].plot(training_similarity, training_err, 'ob', label='training point', alpha=0.7)
        axs[r,c].plot(testing_similarity, testing_err, 'og', label='testing point', alpha=0.7)
        axs[r,c].set_ylabel('Localization Error(m)', fontsize=8)
        axs[r,c].set_xlabel(f'Similarity', fontsize=8)
        axs[r,c].set_title(f"{titles[i].replace('_', ' ')}", fontsize=8, y=-0.25)
        axs[r,c].legend(loc='best')

    fig.tight_layout()
    dir_path = osp.join(os.getcwd(), 'Result')
    plt.savefig(osp.join(dir_path, 'Similarity.png'),  bbox_inches='tight')
    plt.close()

    np.save(osp.join(dir_path, 'dataset_err.npy'), dataset_err)




if __name__ == '__main__':

    datasets = ['NTU_CSIE_5F', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F']

    plot_similarity(datasets)