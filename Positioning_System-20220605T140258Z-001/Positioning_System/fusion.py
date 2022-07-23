import os
import scipy.stats
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from wifi import DSAR, get_boundary, get_rps, normalize_rssi, RBF


def load(dataset):
    offset_path = osp.join(os.getcwd(), 'Result', dataset, 'RoNIN', 'offset.npy')
    dir_path = osp.join(os.getcwd(), 'Dataset', dataset)
    offset = np.load(offset_path)
    wifi_pos= np.load(osp.join(dir_path, 'wifi_pos.npy'))
    wifi_pos_index = np.load(osp.join(dir_path, 'wifi_pos_index.npy'))
    training_wifi_pos = np.load(osp.join(dir_path,'training_wifi_pos.npy'))
    testing_wifi_pos = np.load(osp.join(dir_path,'testing_wifi_pos.npy'))

    return offset, wifi_pos, wifi_pos_index, training_wifi_pos, testing_wifi_pos


def get_similarity(origin, rssi_reconst, gamma):
    rssi_origin = normalize_rssi(origin)
    return RBF(rssi_origin, rssi_reconst, gamma, axis=1)


def get_imu_track(offset, init_pos):
    init_pos = init_pos[None,:]
    imu_track = np.r_[init_pos, offset]
    imu_track = np.cumsum(imu_track, axis=0)
    return imu_track


def calculate_cos(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calculate_sin(v1, v2):
    return np.cross(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calculate_scale(v1, v2):
    return np.linalg.norm(v2) / np.linalg.norm(v1)


def fusion(method, gt_pos, offset, wifi_pos, wifi_pos_index, weights=None):
    
    fusion_pos = gt_pos[None,0,:]
    fusion_track = gt_pos[None,0,:]

    np.random.seed(0)
    particle = np.random.normal(loc=gt_pos[0,:], scale=(2,2), size=(20,2))
    particle_weigths = np.full(20,1)
    
    for i in range(1, wifi_pos_index.shape[0]):
        
        interval = offset[wifi_pos_index[i-1]:wifi_pos_index[i]]
        step_length = np.cumsum(interval, axis=0)[-1]

        Pit = fusion_pos[-1] + step_length
        Pwt = wifi_pos[i]

            
        if method == 'Wi-Fi_Confidence':

            Pt = Pwt * weights[i] + Pit * (1-weights[i])

        elif method == 'Particle_Filter':

            particle += step_length
            dis = np.linalg.norm(particle - Pwt, axis=1)
            mean = 2 * np.sqrt(2)
            std = np.std(dis)
            w = scipy.stats.norm(dis, std).pdf(mean)
            particle_weigths = particle_weigths*w / (particle_weigths*w).sum()

            Pt = np.average(particle, weights=particle_weigths, axis=0)


        c = calculate_cos((Pit - fusion_pos[-1]), (Pt - fusion_pos[-1]))
        s = calculate_sin((Pit - fusion_pos[-1]), (Pt - fusion_pos[-1]))
        alpha = calculate_scale((Pit - fusion_pos[-1]), (Pt - fusion_pos[-1]))
        rotation = [[c, -s], [s, c]]
        rot_interval = (rotation @ interval.T) * alpha
        fusion_track = np.r_[fusion_track, rot_interval.T]
        fusion_pos = np.r_[fusion_pos, Pt[None,:]]
        
    fusion_track = np.cumsum(fusion_track, axis=0)

    return fusion_pos, fusion_track


def plot_fusion(dataset, fusion_method, fusion_pos, fusion_track, imu_track, wifi_pos, gt_pos):

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    mksz = 5

    axs[0,0].plot(wifi_pos[:,0], wifi_pos[:,1], 'o-b', label='Wi-Fi track', markersize=mksz)
    axs[0,0].plot(gt_pos[:,0], gt_pos[:,1], 'og', markersize=mksz, alpha=0.7)
    axs[0,0].set_title('(a) Wi-Fi Prediction', fontsize=8, y=-0.2)
    axs[0,0].grid()

    axs[0,1].plot(imu_track[:,0], imu_track[:,1], '-c')
    axs[0,1].plot(gt_pos[:,0], gt_pos[:,1], 'og', markersize=mksz, alpha=0.7)
    axs[0,1].set_title('(b) IMU Prediction', fontsize=8, y=-0.2)
    axs[0,1].grid()

    axs[1,0].plot(fusion_track[:,0], fusion_track[:,1], '-r')
    axs[1,0].plot(fusion_pos[1:,0], fusion_pos[1:,1], 'ok', markersize=mksz)
    axs[1,0].plot(gt_pos[:,0], gt_pos[:,1], 'og', markersize=mksz, alpha=0.7, label='ground truth')
    axs[1,0].set_title(f'(c) Fusion Result', fontsize=8, y=-0.25)
    axs[1,0].grid()

    for i in range(1, wifi_pos.shape[0]):
        point_x = np.r_[wifi_pos[i,0], fusion_pos[i,0]]
        point_y = np.r_[wifi_pos[i,1], fusion_pos[i,1]]
        axs[1, 1].plot(point_x, point_y, '--k')

    axs[1,1].plot(fusion_track[:,0], fusion_track[:,1], '-r', label='fusion track')
    axs[1,1].plot(fusion_pos[1:,0], fusion_pos[1:,1], 'ok', label='fusion point')
    axs[1,1].plot(imu_track[:,0], imu_track[:,1], '-c', label='IMU track')
    axs[1,1].plot(wifi_pos[:,0], wifi_pos[:,1], 'ob', label='Wi-Fi point', alpha=0.7)
    axs[1,1].set_title(f'(d) Detailed Fusion Result', fontsize=8, y=-0.25)
    axs[1,1].grid()

    
    lg0 = fig.legend(bbox_to_anchor=(0.25, -0.1), prop={'size': 8}, loc="lower left", ncol=3)
    #lg1 = fig.suptitle(dataset, fontsize=20)
    fig.tight_layout()

    file_path = osp.join(os.getcwd(), 'Result', f'Track_{dataset}_{fusion_method}')
    fig.savefig(file_path, bbox_extra_artists=(lg0,), bbox_inches='tight')
    plt.close()



def plot_detailed_err(dataset, fusion_method, wifi_err, imu_err, dsar_fusion_err):
    
    avg_wifi_err = format(np.average(wifi_err), '.4f')
    avg_imu_err = format(np.average(imu_err), '.4f')
    avg_fusion_err = format(np.average(dsar_fusion_err), '.4f')


    # plt.plot(wifi_err, 'o--b', label=f'Wi-Fi {avg_wifi_err}m(Avg. Error)')
    # plt.plot(imu_err, 'o--g', label=f'RoNIN {avg_imu_err}m(Avg. Error)')
    # plt.plot(dsar_fusion_err, 'o--r', label=f'{fusion_method} {avg_fusion_err}m(Avg. Error)')

    # plt.xlabel('Wi-Fi Step', fontsize=8)
    # plt.ylabel('Localization Error(m)', fontsize=8)
    # plt.title(f'{dataset} Detailed Localization Error', fontsize=8)
    # lg = plt.legend(bbox_to_anchor=(0.25, -0.15), loc='upper left')
    # file_path = osp.join(os.getcwd(), 'Result', f'Detailed_{dataset}_{fusion_method}')
    # plt.savefig(file_path, bbox_extra_artists=(lg,), bbox_inches='tight')
    # plt.close()

    plt.plot(wifi_err, 'o--b', label=f'Wi-Fi DSAR')
    plt.plot(imu_err, 'v--g', label=f'RoNIN')
    plt.plot(dsar_fusion_err, 'x--r', label=f'Wi-Fi Confidence')

    plt.xlabel('Wi-Fi Step', fontsize=8)
    plt.ylabel('Localization Error(m)', fontsize=8)
    plt.legend(loc='best')
    file_path = osp.join(os.getcwd(), 'Result', f'Detailed_{dataset}_{fusion_method}')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()




def save_rel_dis(dataset, rel_dis):
    file_path = osp.join(os.getcwd(), 'Result', dataset, 'DSAR', 'rel_dis.npy')
    np.save(file_path, rel_dis)


if __name__ == '__main__':

    datasets = ['NTU_CSIE_5F', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F']
    fusion_methods = ['Wi-Fi_Confidence']#, 'Particle_Filter']

    dataset_err = np.load(osp.join(os.getcwd(), 'Result', 'dataset_err.npy'), allow_pickle='TRUE').item()

  
    record = np.zeros((4,5))
    float_formatter = "{:.4f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})


    for r, dataset in enumerate(datasets):

        offset, wifi_pos, wifi_pos_index, training_wifi_pos, testing_wifi_pos = load(dataset)
        boundary = get_boundary(training_wifi_pos[:,-2:], testing_wifi_pos[:,-2:])
        rps = get_rps(training_wifi_pos)

        gt_pos = wifi_pos[:,-2:]
        dsar_pos, wifi_err, rssi_reconst = DSAR(wifi_pos, dataset, boundary)
        gamma = max(dataset_err[dataset])
        dsar_weight = get_similarity(wifi_pos[:,:-2], rssi_reconst, gamma)

        imu_track = get_imu_track(offset, dsar_pos[0])
        imu_pos = imu_track[wifi_pos_index]


        imu_err = np.linalg.norm(imu_pos - gt_pos, axis=1)
        record[r,0] = np.average(wifi_err)
        record[r,1] = np.average(imu_err)

        for c, fusion_method in enumerate(fusion_methods):

            init_dsar_fusion_pos, init_dsar_fusion_track = fusion(fusion_method, gt_pos, offset, dsar_pos, wifi_pos_index, dsar_weight)
            init_dsar_fusion_err = np.linalg.norm(init_dsar_fusion_pos - gt_pos, axis=1)
            avg_init_dsar = format(np.average(init_dsar_fusion_err), '.4f')

            dsar_fusion_pos, dsar_fusion_track = fusion(fusion_method, dsar_pos, offset, dsar_pos, wifi_pos_index,  dsar_weight)
            dsar_fusion_err = np.linalg.norm(dsar_fusion_pos - gt_pos, axis=1)    
            avg_dsar = format(np.average(dsar_fusion_err), '.4f')

            print(avg_init_dsar, avg_dsar, fusion_method)
            if fusion_method != 'Particle_Filter':

                plot_fusion(dataset, fusion_method, dsar_fusion_pos, dsar_fusion_track, imu_track, dsar_pos, gt_pos)
                plot_detailed_err(dataset, fusion_method, wifi_err, imu_err, dsar_fusion_err)

            record[r,c+2] = np.average(dsar_fusion_err)
    
    print(record)
