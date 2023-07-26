import os
import json
import matplotlib.pyplot as plt

# with open(dataset_path + file, 'r') as json_file:
#     data = json.load(json_file)

# with open(dataset_path + file, 'r') as json_file:
#     data = json.load(json_file)










dataset_path = 'dataset/medici/data/holter_bioharness_parsed/'

for root, dirs, files in os.walk(dataset_path):
    for file in sorted(files):
        if '.json' in file and '2022_09_29-10_07_06' in file:
            if 'template' in file:
                continue
            elif '-proc' in file:
                print('PROC', dataset_path + file)
                with open(dataset_path + file, 'r') as json_file:
                    data = json.load(json_file)
                    RR_time = data['RR']['Time']
                    if acc_filtered:
                        acc_time = data['Interpolated_Filtered_Accel']['Time']
                        if acc_filtered_sat:
                            acc_data = [acc if acc < 40 else 40 for acc in data['Interpolated_Filtered_Accel']['Module']]
                        else:
                            acc_data = data['Interpolated_Filtered_Accel']['Module']
                    else:
                        acc_time = data['Interpolated_Accel']['Time']
                        if acc_axes:
                            acc_data_x = data['Interpolated_Accel']['X']
                            acc_data_y = data['Interpolated_Accel']['Y']
                            acc_data_z = data['Interpolated_Accel']['Z']
                        else:
                            acc_data = data['Interpolated_Accel']['Module']
            elif '-quiet' in file:
                if '400_40' in file:
                    print('QUIET', dataset_path + file)
                    with open(dataset_path + file, 'r') as json_file:
                        data = json.load(json_file)
                        RR_quiet_time = data['RR_quiet_index']
            else:
                print(dataset_path + file)
                with open(dataset_path + file, 'r') as json_file:
                    data = json.load(json_file)
                    ecg_data = data['ECG']['EcgWaveform']
                    ecg_time = data['ECG']['Time']
            # with open('dataset_path + file', 'r') as json_file:
            #     config = json.load(json_file)

# print(acc_data)
# exit()


# print(len(ecg_data), len(acc_data))
# print(type(ecg_time[0]), type(acc_time[0]), type(ecg_data[0]), type(acc_data[0]))

ecg_time_start = datetime.datetime.strptime(ecg_time[0], "%d/%m/%Y %H:%M:%S.%f")

# index_test = 3
# delta = round((datetime.datetime.strptime(RR_quiet_time[index_test], "%d/%m/%Y %H:%M:%S.%f") - ecg_time_start) / datetime.timedelta(seconds = (1 / 250)))

half_win = 99
dataset_split = 0.7

X_out = []
for rr in RR_time:
    index = round((datetime.datetime.strptime(rr, "%d/%m/%Y %H:%M:%S.%f") - ecg_time_start) / datetime.timedelta(seconds = (1 / 250)))
    # X.append([[ecg_data[index - half_win : index + half_win]]])
    X_out.append([[[float(d) for d in ecg_data[index - half_win : index + half_win]]]])

X_in = []
for rr in RR_quiet_time:
    index = round((datetime.datetime.strptime(rr, "%d/%m/%Y %H:%M:%S.%f") - ecg_time_start) / datetime.timedelta(seconds = (1 / 250)))
    # X_out.append([[ecg_data[index - half_win : index + half_win]], [acc_data[index - half_win : index + half_win]]])
    if not acc_filtered and acc_axes:
        X_in.append(
            [[[float(d) for d in ecg_data[index - half_win : index + half_win]]],
            [[float(d) for d in acc_data_x[index - half_win : index + half_win]]],
            [[float(d) for d in acc_data_y[index - half_win : index + half_win]]],
            [[float(d) for d in acc_data_z[index - half_win : index + half_win]]]])
    else:
        X_in.append(
            [[[float(d) for d in ecg_data[index - half_win : index + half_win]]],
            [[float(d) for d in acc_data[index - half_win : index + half_win]]]])
















    for i, n in enumerate(range(len(X_in_test))):
        encoded_data = autoencoder.encoder(np.expand_dims(X_in_test[n], axis=0)).numpy()
        decoded_data = autoencoder.decoder(encoded_data).numpy()

        fig, ecg_plt = plt.subplots(figsize=(10,7))
        ecg_plt.plot(X_in_test[n, : , 0, 0], 'b', alpha=0.5)

        ecg_plt = ecg_plt.twinx()
        ecg_plt.plot(X_out_test[n, : , 0, 0], 'b')

        if not acc_filtered and acc_axes:
            acc_plt = ecg_plt.twinx()
            acc_plt.plot(X_in_test[n, : , 0, 1], 'r', alpha=0.5)
            acc_plt.plot(X_in_test[n, : , 0, 2], 'r', alpha=0.5)
            acc_plt.plot(X_in_test[n, : , 0, 3], 'r', alpha=0.5)
        else:
            acc_plt = ecg_plt.twinx()
            acc_plt.plot(X_in_test[n, : , 0, 1], 'r')
        acc_plt.set_ylim([0, 1])

        dec_plt = ecg_plt.twinx()
        dec_plt.plot(decoded_data[0, : -1, 0, 0], 'g')

        plt.title(f'{RR_time_test[n]}')
        print(f'{RR_time_test[n]}')

        plt.show()