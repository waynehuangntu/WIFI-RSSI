import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from wifi import load


if __name__ == '__main__':
    datasets = ['NTU_CSIE_5F', 'DSI', 'IPIN2016_Tutorial', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F']
    titles = ['(a) NTU_CSIE_5F', '(b) DSI', '(c) IPIN2016_Tutorial', '(d) IPIN2020_Track3_2F', '(e) IPIN2020_Track3_3F', '(f) IPIN2020_Track3_5F']

    fig, axs = plt.subplots(2, 9, figsize=(18,10))

    for i in range(len(datasets)):

        dir_path = osp.join(os.getcwd(), 'Result', datasets[i], 'DSAR')

        training_wifi_pos, testing_wifi_pos = load(datasets[i])

        pred_pos = np.load(osp.join(dir_path, 'pred_pos.npy'))

        r = i // 3
        c = (i*3) % 9
            
        axs[r,c].plot(training_wifi_pos[:,-2], training_wifi_pos[:,-1], 'ob')
        axs[r,c].set_title('Reference Points')
        axs[r,c].grid()

        axs[r,c+1].plot(testing_wifi_pos[:,-2], testing_wifi_pos[:,-1], 'og')
        axs[r,c+1].set_title('Testing Points')
        axs[r,c+1].grid()
        axs[r,c+1].sharex(axs[r,c])
        axs[r,c+1].sharey(axs[r,c])
        axs[r,c+1].set_xlabel(titles[i], fontsize=15)

        axs[r,c+2].plot(pred_pos[:,0], pred_pos[:,1], 'or')
        axs[r,c+2].set_title('Prediction')
        axs[r,c+2].grid()
        axs[r,c+2].sharex(axs[r,c])
        axs[r,c+2].sharey(axs[r,c])

    fig.tight_layout()
    file_path = osp.join(os.getcwd(), 'Result', 'dataset.png')
    plt.savefig(file_path, bbox_inches='tight')


