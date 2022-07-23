1. conda env create -f /path/to/environment.yml
2. preprocess.py -> preprocess raw data
3. training_wifi.py -> train the model for wifi prediction
4. wifi.py & im.py -> predict the traget position
5. fusion.py -> merge both location predicted by wifi and imu 
6. cdf_loss.py & similarity.py & plt_wifi_pos.py & boxplot.py -> plot the results all u need == 