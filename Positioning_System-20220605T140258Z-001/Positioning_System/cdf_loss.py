import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt


def plot_cdfs(loc_errs, methods):
    
    titles = ['(a) NTU_CSIE_5F', '(b) DSI', '(c) IPIN2016_Tutorial', '(d) IPIN2020_Track3_2F', '(e) IPIN2020_Track3_3F', '(f) IPIN2020_Track3_5F']
    colors = ['b', 'g', 'r', 'c']

    fig, axs = plt.subplots(2, 3, figsize=(18,10))

    for idx, key in enumerate(loc_errs):
        
        loc_err_gp = loc_errs[key]
        r, c = idx // 3, idx % 3
        print('='*25 + key + '='*25)
        for i in range(len(loc_err_gp)):
            
            avg_err = format(np.average(loc_err_gp[i]), '.4f')
            print(avg_err)
            histogram, bins = np.histogram(loc_err_gp[i], bins=np.arange(10))
            cdf = np.cumsum(histogram) / np.sum(histogram)
            axs[r, c].plot((bins[1:]+bins[:-1])/2, cdf, f'-{colors[i]}', label=f'{methods[i]} {avg_err}m(Avg. Error)', linewidth=3, alpha=0.7)

        axs[r, c].set_ylabel('CDF', fontsize=15)
        axs[r, c].set_xlabel(f'Localization Error(m)\n {titles[idx]}', fontsize=15)
        axs[r, c].grid()
        axs[r, c].legend(loc='best')

    fig.tight_layout()
    dir_path = osp.join(os.getcwd(), 'Result')
    plt.savefig(osp.join(dir_path, 'CDF.png'),  bbox_inches='tight')
    plt.close()
    


def plot_losses(losses):
    titles = ['(a) NTU_CSIE_5F', '(b) DSI', '(c) IPIN2016_Tutorial', '(d) IPIN2020_Track3_2F', '(e) IPIN2020_Track3_3F', '(f) IPIN2020_Track3_5F']

    fig, axs = plt.subplots(2, 3, figsize=(18,10))
    for idx, key in enumerate(losses):
        loss = losses[key]
        r, c = idx // 3, idx % 3
        axs[r, c].plot(loss[200:,0], label='reconstruction loss', alpha=0.7)
        axs[r, c].plot(loss[200:,1], label='regression loss', alpha=0.7)
        axs[r, c].plot(loss[200:,2], label='validation loss', alpha=0.7)

        axs[r, c].set_ylabel('Loss', fontsize=15)
        axs[r, c].set_xlabel(f'Epoch\n {titles[idx]}', fontsize=15)
        axs[r, c].grid()
        axs[r, c].legend(loc='best')
    
    fig.tight_layout()
    dir_path = osp.join(os.getcwd(), 'Result')
    plt.savefig(osp.join(dir_path, 'Loss.png'),  bbox_inches='tight')
    plt.close()



if __name__ == '__main__':

    datasets = ['NTU_CSIE_5F', 'DSI', 'IPIN2016_Tutorial', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F']

    methods = ['WKNN', 'RandomForest', 'WiDeep', 'DSAR']

    loc_errs = {dataset:[] for dataset in datasets}
    losses = {}

    for dataset in datasets:

        for method in methods:

            dir_path = osp.join(os.getcwd(), 'Result', dataset, method)
            
            loc_err = np.load(osp.join(dir_path, 'loc_err.npy'))

            loc_errs[dataset].append(loc_err)

            if method == 'DSAR':
                loss = np.load(osp.join(dir_path, 'losses.npy'))
                losses[dataset] = loss

        


    plot_cdfs(loc_errs, methods)

    plot_losses(losses)