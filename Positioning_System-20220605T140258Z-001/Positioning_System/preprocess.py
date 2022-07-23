import os
import utm
import quaternion
import numpy as np
import os.path as osp


def check_mkdirs():
    dataset_folders = ['NTU_CSIE_5F', 'DSI', 'IPIN2016_Tutorial', 'IPIN2020_Track3_1F', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F', 'IPIN2020_Track3_4F','IPIN2020_Track3_5F']
    for folder in dataset_folders:
        folder_path = osp.join(os.getcwd(), 'Dataset', folder)
        if not osp.exists(folder_path):
            os.makedirs(folder_path)
    
    result_folders = ['NTU_CSIE_5F', 'DSI', 'IPIN2016_Tutorial', 'IPIN2020_Track3_2F', 'IPIN2020_Track3_3F','IPIN2020_Track3_5F']
    methods = ['RoNIN', 'WKNN', 'RandomForest', 'WiDeep', 'DSAR']
    for folder in result_folders:
        for method in methods:
            folder_path = osp.join(os.getcwd(), 'Result', folder, method)
            if not osp.exists(folder_path):
                os.makedirs(folder_path)
        


def align_imu_wifi(acce, gyro, grv, wifi_pos_time, dt_scale):
    begin = wifi_pos_time[0,-1]
    end = wifi_pos_time[-1,-1]

    acce = acce[(acce[:,0]>=begin) & (acce[:,0]<=end)]
    gyro = gyro[(gyro[:,0]>=begin) & (gyro[:,0]<=end)]
    grv = grv[(grv[:,0]>=begin) & (grv[:,0]<=end)]
    
    imu_len = min(acce.shape[0], gyro.shape[0])
    acce = acce[:imu_len]
    gyro = gyro[:imu_len]

    dt = [acce[1:,1]-acce[:-1,1], gyro[1:,1]-gyro[:-1,1]]
    dt = np.mean(dt) * dt_scale

    glob_grv = np.zeros((imu_len,4), dtype=np.float64)
    for i in range(imu_len):
        index = np.argmin(np.abs(grv[:,0] - acce[i,0]))
        glob_grv[i] = grv[index,-4:]

    wifi_pos_index = np.zeros(wifi_pos_time.shape[0], dtype=np.int64)
    for i in range(wifi_pos_index.shape[0]):
        index = np.argmin(np.abs(acce[:,0] - wifi_pos_time[i,-1]))
        wifi_pos_index[i] = index
    wifi_pos = wifi_pos_time[:,:-1]

    ori_q = quaternion.from_float_array(glob_grv)
    acc_q = quaternion.from_float_array(np.c_[np.zeros((imu_len,1)), acce[:,-3:]])
    gyro_q = quaternion.from_float_array(np.c_[np.zeros((imu_len,1)), gyro[:,-3:]])
    glob_acc = quaternion.as_float_array(ori_q * acc_q * ori_q.conj())[:, 1:] 
    glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:] 

    return glob_acc, glob_gyro, dt, wifi_pos, wifi_pos_index



def save_track_info(dir_path, glob_acce, glob_gyro, dt, wifi_pos, wifi_pos_index):
    print('='*50)
    print(dir_path)
    print('global acce shape', glob_acce.shape)
    print('global gyro shape', glob_gyro.shape)
    print('average sensor time interval', dt)
    print('wifi pos shape', wifi_pos.shape)
    print('wifi pos index shape', wifi_pos_index.shape)

    glob_acce_path = osp.join(dir_path, 'glob_acce.npy') 
    glob_gyro_path = osp.join(dir_path, 'glob_gyro.npy')
    dt_path = osp.join(dir_path, 'dt.npy') 
    wifi_pos_path = osp.join(dir_path, 'wifi_pos.npy')
    wifi_pos_index_path = osp.join(dir_path, 'wifi_pos_index.npy') 

    np.save(glob_acce_path, glob_acce)
    np.save(glob_gyro_path, glob_gyro)    
    np.save(dt_path, dt)   
    np.save(wifi_pos_path, wifi_pos)
    np.save(wifi_pos_index_path, wifi_pos_index) 



def save_fingerprint(dir_path, training_wifi_pos, testing_wifi_pos):
    print('='*50)
    print(dir_path)
    print('training wifi pos shape', training_wifi_pos.shape)
    print('testing wifi pos shape', testing_wifi_pos.shape)

    training_wifi_pos_path = osp.join(dir_path, 'training_wifi_pos.npy') 
    testing_wifi_pos_path = osp.join(dir_path, 'testing_wifi_pos.npy') 

    np.save(training_wifi_pos_path, training_wifi_pos)
    np.save(testing_wifi_pos_path, testing_wifi_pos)



def NTU_CSIE_5F():
    fingerprint_file_path = osp.join(os.getcwd(), 'Raw', 'NTU_CSIE_5F', 'Fingerprint.txt')
    
    bssid_index = {}
    training_wifi_pos = []
    testing_wifi_pos = []

    with open(fingerprint_file_path, 'r') as f:
        position_index = []
        index = 0
        lines = f.read().splitlines()
        for i in range(len(lines)):
            elements = lines[i].split(',')
            if elements[0] == 'Position':
                position_index.append(i)
            else:
                bssid, rssi = elements
                if bssid not in bssid_index:
                    bssid_index[bssid] = index
                    index += 1
        
        total_APs = len(bssid_index)
        testing_set = set()
        
        for p_idx in position_index:
            fingerprint = np.full(total_APs+2, -100, dtype=np.float64)
            _, x, y = lines[p_idx].split(',')
            pos = (eval(x)/100, eval(y)/100)
            fingerprint[-2:] = pos

            p_idx += 1
            elements = lines[p_idx].split(',')
            while len(elements) == 2:
                bssid, rssi = elements
                bssid, rssi = bssid_index[bssid], eval(rssi)
                fingerprint[bssid] = rssi
                p_idx += 1
                if p_idx == len(lines): 
                    break
                elements = lines[p_idx].split(',')

            if pos not in testing_set:
                testing_set.add(pos)
                testing_wifi_pos.append(fingerprint)
            else:
                training_wifi_pos.append(fingerprint)

        training_wifi_pos = np.array(training_wifi_pos, dtype=np.float64)
        testing_wifi_pos = np.array(testing_wifi_pos, dtype=np.float64)
    
    
    def get_imu_wifi(file_path, bssid_index):
        total_APs = len(bssid_index)
        wifi_pos_time = []
        acce = []
        gyro = []
        grv = []

        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
            i = 0
            while i < len(lines):
                elements = lines[i].split(',')
                if elements[0] == 'Position':
                    fingerprint = np.full(total_APs+3, -100, dtype=np.float64)
                    _, app_timestamp, x, y = elements
                    pos_time = (eval(x)/100, eval(y)/100, eval(app_timestamp))
                    fingerprint[-3:] = pos_time

                    i += 1
                    elements = lines[i].split(',')
                    while elements[0] == 'WiFi':
                        _, bssid, rssi = elements
                        bssid, rssi = bssid_index.get(bssid, 'KeyError'), eval(rssi)
                        if bssid != 'KeyError':
                            fingerprint[bssid] = rssi
                        i += 1
                        elements = lines[i].split(',')
                    wifi_pos_time.append(fingerprint)
    
                if elements[0] == 'ACC':
                    app_timestamp = eval(elements[1])
                    sen_timestamp = eval(elements[2])
                    acce_x = eval(elements[3])
                    acce_y = eval(elements[4])
                    acce_z = eval(elements[5])
                    acce.append([app_timestamp, sen_timestamp, acce_x, acce_y, acce_z])
                    
                elif elements[0] == 'GYRO': 
                    app_timestamp = eval(elements[1])
                    sen_timestamp = eval(elements[2])
                    gyro_x = eval(elements[3])
                    gyro_y = eval(elements[4])
                    gyro_z = eval(elements[5])
                    gyro.append([app_timestamp, sen_timestamp, gyro_x, gyro_y, gyro_z])

                elif elements[0] == 'GRV': 
                    app_timestamp = eval(elements[1])
                    sen_timestamp = eval(elements[2])
                    rot_w = eval(elements[3])
                    rot_x = eval(elements[4]) # quaternion
                    rot_y = eval(elements[5])
                    rot_z = eval(elements[6])
                    grv.append([app_timestamp, sen_timestamp, rot_w, rot_x, rot_y, rot_z])
                
                i += 1

        acce = np.array(acce, dtype=np.float64)
        gyro = np.array(gyro, dtype=np.float64)
        grv = np.array(grv, dtype=np.float64)
        wifi_pos_time = np.array(wifi_pos_time, dtype=np.float64)

        return acce, gyro, grv, wifi_pos_time


    track_file_path = osp.join(os.getcwd(), 'Raw', 'NTU_CSIE_5F', 'Track1.txt')
    acce, gyro, grv, wifi_pos_time = get_imu_wifi(track_file_path, bssid_index)
    glob_acce, glob_gyro, dt, wifi_pos, wifi_pos_index = align_imu_wifi(acce, gyro, grv, wifi_pos_time, 1e-9)
    save_path = osp.join(os.getcwd(), 'Dataset', 'NTU_CSIE_5F')
    save_track_info(save_path,  glob_acce, glob_gyro, dt, wifi_pos, wifi_pos_index)
    save_fingerprint(save_path, training_wifi_pos, testing_wifi_pos)

    

def DSI():
    training_wifi_path = osp.join(os.getcwd(), 'Raw', 'DSI', 'rm_rss.csv')
    trianing_pos_path = osp.join(os.getcwd(), 'Raw', 'DSI', 'rm_crd.csv')

    testing_wifi_path = osp.join(os.getcwd(), 'Raw', 'DSI', 'tj_rss.csv')
    testing_pos_path = osp.join(os.getcwd(), 'Raw', 'DSI', 'tj_crd.csv')

    training_wifi = np.genfromtxt(training_wifi_path, delimiter=',', dtype=np.float64)
    training_wifi = np.where(training_wifi == -150, -100, training_wifi)
    training_pos = np.genfromtxt(trianing_pos_path, delimiter=',', dtype=np.float64)
    training_wifi_pos = np.c_[training_wifi, training_pos]

    testing_wifi = np.genfromtxt(testing_wifi_path, delimiter=',', dtype=np.float64)
    testing_wifi = np.where(testing_wifi == -150, -100, testing_wifi)
    testing_pos = np.genfromtxt(testing_pos_path, delimiter=',', dtype=np.float64)
    testing_wifi_pos = np.c_[testing_wifi, testing_pos]

    save_path = osp.join(os.getcwd(), 'Dataset', 'DSI')
    save_fingerprint(save_path, training_wifi_pos, testing_wifi_pos)



def IPIN2016_Tutorial():

    training_wifi_pos = np.array([]).reshape(0,170)
    testing_wifi_pos = np.array([]).reshape(0,170)
    for i in range(9):
        wifi_path = osp.join(os.getcwd(), 'Raw', 'IPIN2016_Tutorial', f'fingerprints_0{i}.csv')
        wifi_pos = np.genfromtxt(wifi_path, delimiter=',', dtype=np.float64)
        wifi_pos = wifi_pos[1:,:-7]
        rssi = wifi_pos[:,:-2]
        pos = wifi_pos[:,-1:-3:-1]
        rssi = np.where(rssi == 100, -100, rssi)
        pos = pos/100
        wifi_pos = np.c_[rssi, pos]

        if i == 0:
            testing_wifi_pos = np.r_[testing_wifi_pos, wifi_pos]
        else:
            training_wifi_pos = np.r_[training_wifi_pos, wifi_pos]
    
    training_wifi_pos = np.array(training_wifi_pos, dtype=np.float64)
    testing_wifi_pos = np.array(testing_wifi_pos, dtype=np.float64)
    
    save_path = osp.join(os.getcwd(), 'Dataset', 'IPIN2016_Tutorial')
    save_fingerprint(save_path, training_wifi_pos, testing_wifi_pos)



def IPIN2020_Track3():
    floor_track = {1:'V01.txt', 2:'V03.txt', 3:'V05.txt', 4:'V07.txt', 5:'V09.txt'}
    floor_testing = {1:['T01_01.txt', 'V01.txt'], 
                     2:['T02_01.txt', 'T05_01.txt', 'T07_01.txt', 'T10_01.txt', 'T14_01.txt', 'V03.txt'],
                     3:['T03_01.txt', 'T06_01.txt', 'T08_01.txt', 'T15_01.txt', 'T18_01.txt', 'T31_01.txt', 'V05.txt'],
                     4:['T11_01.txt', 'T12_01.txt', 'V07.txt'],
                     5:['T04_01.txt', 'T09_01.txt', 'T13_01.txt', 'T16_01.txt', 'T19_01.txt', 'T33_01.txt', 'V09.txt']}

    dir_path = osp.join(os.getcwd(), 'Raw', 'IPIN2020_Track3')

    def classify_floor(file_path):
        floor_set = set()
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                if len(line) > 4 and line[0:4] == 'POSI':
                    _, _, _, _, _, floor, _ = line.split(';')
                    floor_set.add(eval(floor))
        
        return floor_set

    floor_file_paths = {1:[], 2:[], 3:[], 4:[], 5:[]}
    floor_track_file_path = {1:[], 2:[], 3:[], 4:[], 5:[]}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = osp.join(root, file)
            floor_set = classify_floor(file_path)
            if len(floor_set) != 1:
                continue
            floor = floor_set.pop()
            floor_file_paths[floor].append(file_path)

            if file == floor_track[floor]:
                floor_track_file_path[floor].append(file_path)
    
    for floor in floor_file_paths:

        bssid_index = {}
        index = 0
        origin_pos = np.full(2, np.inf, dtype=np.float64)
        for file_path in floor_file_paths[floor]:
            with open(file_path, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    if len(line) < 4:
                        continue

                    if line[0:4] == 'WIFI':
                        _, _, _, _, bssid, _, _ = line.split(';')
                        if bssid not in bssid_index:
                            bssid_index[bssid] = index
                            index += 1

                    elif line[0:4] == 'POSI':
                        _, _, _, lat, lon, _, _ = line.split(';')
                        lat, lon = eval(lat), eval(lon)
                        x, y, _, _ = utm.from_latlon(lat, lon)
                        origin_pos[0] = min(origin_pos[0], x)
                        origin_pos[1] = min(origin_pos[1], y)
        
        training_wifi_pos = []
        testing_wifi_pos = []
        wifi_pos_time = []
        acce = []
        gyro = []
        grv = []
        total_APs = len(bssid_index)
        for file_path in floor_file_paths[floor]:
            _, file_name = osp.split(file_path)
            with open(file_path, 'r') as f:
                time_position = []
                time_wifi = [] 
                start = False
                lines = f.read().splitlines()
                for line in lines:
                    if len(line) < 4:
                        continue

                    if line[0:4] == 'POSI':
                        _, app_timestamp, _, lat, lon, _, _ = line.split(';')
                        app_timestamp, lat, lon = eval(app_timestamp), eval(lat), eval(lon)
                        x, y, _, _ = utm.from_latlon(lat, lon)
                        x = x - origin_pos[0]
                        y = y - origin_pos[1]
                        time_position.append([app_timestamp, x, y])
                        start = True  

                    elif line[0:4] == 'WIFI' and start:
                        
                        _, app_timestamp, sen_timestamp, _, bssid, _, rssi = line.split(';')
                        app_timestamp, bssid, rssi = eval(app_timestamp), bssid_index[bssid], eval(rssi)
                        time_wifi.append([app_timestamp, bssid, rssi])

                    elif file_name == floor_track[floor]:
                        if line[0:4] == 'ACCE': # m/s^2
                            elements = line.split(';')
                            app_timestamp = eval(elements[1])
                            sen_timestamp = eval(elements[2])
                            acce_x = eval(elements[3])
                            acce_y = eval(elements[4])
                            acce_z = eval(elements[5])
                            acce.append([app_timestamp, sen_timestamp, acce_x, acce_y, acce_z])

                        elif line[0:4] == 'GYRO': # rad/s
                            elements = line.split(';')
                            app_timestamp = eval(elements[1])
                            sen_timestamp = eval(elements[2])
                            gyro_x = eval(elements[3])
                            gyro_y = eval(elements[4])
                            gyro_z = eval(elements[5])
                            gyro.append([app_timestamp, sen_timestamp, gyro_x, gyro_y, gyro_z])

                        elif line[0:4] == 'AHRS': # AHRS;AppTS(s);SensorTS(s);PitchX(º);RollY(º);YawZ(º);RotVecX();RotVecY();RotVecZ();Accuracy(int)
                            elements = line.split(';')
                            app_timestamp = eval(elements[1])
                            sen_timestamp = eval(elements[2])
                            rot_x = eval(elements[6]) # quaternion
                            rot_y = eval(elements[7])
                            rot_z = eval(elements[8])
                            rot_w = 1 - rot_x**2 - rot_y**2 - rot_z**2 
                            rot_w = 0 if rot_w < 0 else np.sqrt(rot_w)
                            grv.append([app_timestamp, sen_timestamp, rot_w, rot_x, rot_y, rot_z])

            time_position = np.array(time_position, dtype=np.float64) 
            time_wifi = np.array(time_wifi, dtype=np.float64) 


            for i in range(time_position.shape[0]):

                diff = np.abs(time_wifi[:,0] - time_position[i,0]) < 1
                tp_group = sorted(set(time_wifi[diff,0])) 

                for tp in tp_group:
                    fingerprint = np.full((total_APs+2), -100, dtype=np.float64)

                    indexes = np.where(time_wifi[:,0] == tp)[0]
                    for index in indexes:
                        fingerprint[int(time_wifi[index,1])] = time_wifi[index,2]
                    
                    fingerprint[-2:] = time_position[i,-2:]

                    if file_name in floor_testing[floor]:
                        testing_wifi_pos.append(fingerprint)
                    else:
                        training_wifi_pos.append(fingerprint)

            if file_name == floor_track[floor]:
                if floor == 5:
                    time_position = np.insert(time_position, 5, [161.26696200249896, 16.16833571, 17.86608429], axis=0)

                for i in range(time_position.shape[0]-1):
                    if i == 4 and floor == 5:
                        print(wifi_pos_time[4][-3:-1])

                    xl, yl = time_position[i,1:]
                    xr, yr = time_position[i+1,1:]

                    diff = ((time_wifi[:,0]-time_position[i,0]) > 0) & ((time_wifi[:,0]-time_position[i+1,0]) <= 0)
                    tp_group = sorted(set(time_wifi[diff,0])) # set of app timestamp
    
                    for tp in tp_group:
                        fingerprint = np.full((total_APs+3), -100, dtype=np.float64)
                        indexes = np.where(time_wifi[:,0] == tp)[0]

                        for index in indexes:
                            fingerprint[int(time_wifi[index,1])] = time_wifi[index,2]

                        t = time_position[i+1,0] - time_position[i,0]
                        wl = tp - time_position[i,0]
                        wr = time_position[i+1,0] - tp
                        x = xr * (wl/t) + xl * (wr/t)
                        y = yr * (wl/t) + yl * (wr/t)
                        fingerprint[-3:] = [x, y, tp]

                        wifi_pos_time.append(fingerprint)


        training_wifi_pos = np.array(training_wifi_pos, dtype=np.float64)
        testing_wifi_pos = np.array(testing_wifi_pos, dtype=np.float64)
        wifi_pos_time = np.array(wifi_pos_time, dtype=np.float64)

        acce = np.array(acce, dtype=np.float64)
        gyro = np.array(gyro, dtype=np.float64)
        grv = np.array(grv, dtype=np.float64)

        glob_acce, glob_gyro, dt, wifi_pos, wifi_pos_index = align_imu_wifi(acce, gyro, grv, wifi_pos_time, 1e-0)
        save_path = osp.join(os.getcwd(), 'Dataset', f'IPIN2020_Track3_{floor}F')
        save_track_info(save_path,  glob_acce, glob_gyro, dt, wifi_pos, wifi_pos_index)
        save_fingerprint(save_path, training_wifi_pos, testing_wifi_pos)




if __name__ == '__main__':

    # check_mkdirs()

    # NTU_CSIE_5F()

    # DSI()

    # IPIN2016_Tutorial()

    IPIN2020_Track3()