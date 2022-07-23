import os
import numpy as np
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(rc={'figure.figsize':(18,10)})

if __name__ == '__main__':

    datasets = ['NTU_CSIE_5F', 'DSI', 'IPIN2016_Tutorial', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_5F']
    titles = ['(a) NTU_CSIE_5F', '(b) DSI', '(c) IPIN2016_Tutorial', '(d) IPIN2020_Track3_2F', '(e) IPIN2020_Track3_3F', '(f) IPIN2020_Track3_5F']

    methods = ['WKNN', 'RandomForest', 'WiDeep', 'DSAR']

    fig, axs = plt.subplots(2, 3, figsize=(18,10))
    
    for idx, dataset in enumerate(datasets):
        testing_loc_errs = {}
        vertical_offset = []
        legends = []
        r, c = idx // 3, idx % 3
        for method in methods:

            dir_path = osp.join(os.getcwd(), 'Result', dataset, method)
            
            testing_loc_err = np.load(osp.join(dir_path, 'loc_err.npy'))
            testing_loc_errs[method] = testing_loc_err
            q1, q2, q3 = np.percentile(testing_loc_err, [25,50,75])
            interquartile = round(q3 -q1, 2)
            legends.append(interquartile)
            vertical_offset.append(q2)
        
        vertical_offset - np.array(vertical_offset)

        testing_loc_errs = pd.DataFrame.from_dict(testing_loc_errs)
        
        sns.boxplot(data=testing_loc_errs, linewidth=2.5, showfliers=False, ax=axs[r,c])

        for xtick in axs[r,c].get_xticks():
            axs[r,c].text(xtick, vertical_offset[xtick], legends[xtick], horizontalalignment='center',size=12,color='w',weight='semibold')
     
        axs[r,c].set_xlabel(f'Wi-Fi methods\n {titles[idx]}')
        axs[r,c].set_ylabel('Localization error(m)')
    plt.savefig(osp.join(os.getcwd(), 'Result', 'boxplot'))
    plt.close()