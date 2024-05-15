import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
import random
import time
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# フラットな手の座標をMediaPipe Handランドマークに変換する関数
def flatten_to_landmarks(coordinates):
    landmarks = []
    for i in range(0, len(coordinates), 3):
        landmarks.append((coordinates[i], coordinates[i + 1], coordinates[i + 2]))
    return landmarks

# データの読み込みと前処理
def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    x_data = []
    y_data = []

    for i in range(len(df) - n_seq):
        x_sequence = df.iloc[i][[f'{j}_{c}' for j in range(num_joints) for c in ['x', 'y', 'z']]].values.flatten() 
        x_data.append(x_sequence)
        
        y_sequence = df.iloc[i+n_seq][[f'{j}_{c}' for j in range(num_joints) for c in ['x', 'y', 'z']]].values.flatten()
        y_data.append(y_sequence)

    x_data = np.array(x_data, dtype=np.float32)  # float32に変換
    y_data = np.array(y_data, dtype=np.float32)

    return x_data, y_data


# じゃんけんの手のラベル
janken_labels = {0: 'チョキ', 1: 'グー', 2: 'パー'}

# モデルの定義（Net クラスを使用）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(63, 630)
        self.fc2 = nn.Linear(630, 315)
        self.fc3 = nn.Linear(315, 100)
        self.fc4 = nn.Linear(100, 3)  # 出力層のユニット数をクラス数に合わせる

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# モデル、損失関数、オプティマイザ
# モデルのパラメータ
n_seq = 3
print(n_seq)
num_joints = 21
input_size = num_joints * 3  # num_joints は前のコードで定義されているものと仮定
hidden_size = 16
output_size = num_joints * 3
num_layers = 1
batch_size = 36
n_epochs = 100

# モデルのアーキテクチャ
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out



# DataLoaderの使用
def create_dataloader(x_train, y_train):
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


# モデル、損失関数、オプティマイザ
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

# 学習率のスケジューリング
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# メインの処理
if __name__ == "__main__":
    # データの読み込みと前処理
    #train_csv_path = '/home/iwata/Downloads/hand_all/hand_21s.csv'
    train_csv_path = '2_19_hand.csv'
    #train_csv_path = '/home/iwata/Downloads/hand_all/2_19_hand_pa-_only.csv'
    #train_csv_path = '/home/iwata/Downloads/hand_all/2_19_hand_choki_only.csv'
    #train_csv_path = '/home/iwata/Downloads/hand_all/hand_300.csv'
    #train_csv_path = '/home/iwata/Downloads/hand_all/hand_300.csv'

    #test_csv_path = '/home/iwata/Downloads/test_5/gu_test_5/gu_test.csv'
    test_csv_path = 'test_10/choki_test_10/choki_test.csv'
    #test_csv_path = '/home/iwata/Downloads/test_5/pa-_test_5/pa-_test.csv'
    
    #test_csv_path = '/home/iwata/Downloads/test_4/gu_test_4/gu_train.csv'
    #test_csv_path = '/home/iwata/Downloads/test_4/choki_test_4/choki_train.csv'
    #test_csv_path = '/home/iwata/Downloads/test_4/pa-_test_4/pa-_train.csv'
    
    #a = gu
    #a = choki
    #a = pa-
        
    x_train, y_train = preprocess_data(train_csv_path)
    x_test, y_test = preprocess_data(test_csv_path)

    #print(x_test)
    # DataLoaderの作成
    train_loader = create_dataloader(x_train, y_train)

    
    for i in range(1):
        print("ループ回数:", i+1)
    
        # トレーニングループ
        start_time = time.time()
        for epoch in range(n_epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.unsqueeze(1).to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

        training_time = time.time() - start_time
        #print(f'学習時間: {training_time:.2f} 秒')

        # テストデータで評価
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            test_outputs = model(torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device))
            test_loss = criterion(test_outputs, torch.tensor(y_test, dtype=torch.float32).to(device))
            processing_time_per_image = time.time() - start_time
            #print(f'Test Loss: {test_loss.item():.4f}')
            #print(f'処理時間_1: {processing_time_per_image:.6f} 秒')


    
        # 3つのテストサンプルごとに予測結果を処理し、1つずつずらして繰り返す
        window_size = 3  # ウィンドウサイズ（処理するテストサンプルの数）

        for sample_index in range(0, len(x_test)):  # ウィンドウを1つずつずらして処理する
        #for sample_index in range(0, len(x_test) - window_size + 1):  # ウィンドウを1つずつずらして処理する
            # ウィンドウ内のテストサンプルを取得
            x_test_batch = torch.tensor(x_test[sample_index:sample_index+window_size], dtype=torch.float32).unsqueeze(1).to(device)
            y_test_batch = torch.tensor(y_test[sample_index:sample_index+window_size], dtype=torch.float32).to(device)
            print(sample_index+1)

            with torch.no_grad():
                # 既存のモデルでx_testを予測
                start_time = time.time()
                predicted_tensor_x = model(x_test_batch)
                predicted_x = predicted_tensor_x.cpu().numpy()
                processing_time_per_image = time.time() - start_time
                #print(f'処理時間_2: {processing_time_per_image:.6f} 秒')

                # 新しいモデルのインスタンスを作成
                new_model = Net()
                #new_model.load_state_dict(torch.load('your_model.pth'))
                new_model.load_state_dict(torch.load('hanbetu_all.pth'))
                new_model.to(device)
                new_model.eval()

                # 既存の手の座標を指定
                # 予測結果の後に続く処理
                ground_truth_landmarks = flatten_to_landmarks(y_test_batch[0])  # 修正
                #ground_truth_landmarks = flatten_to_landmarks(y_test[sample_index])
                truth_landmarks = flatten_to_landmarks(x_test[sample_index])
                sample_landmarks_x = flatten_to_landmarks(predicted_x[0])  # 予測結果を利用

                # landmark_0 の x 座標と y 座標の差分を計算し、表示する
                x_difference = y_test[sample_index][0] - predicted_x[0][0]
                y_difference = y_test[sample_index][1] - predicted_x[0][1]
                #print(f'Landmark 0 の x 座標の差分: {x_difference}')
                #print(f'Landmark 0 の y 座標の差分: {y_difference}')
            
                # sample_landmarks_xに含まれるすべてのデータを修正してリストに変換
                corrected = []
                for landmark in sample_landmarks_x:
                    x_corrected = landmark[0] + x_difference
                    y_corrected = landmark[1] + y_difference
                    z_corrected = landmark[2]  # z座標は変更しないと仮定
                    corrected.append((x_corrected, y_corrected, z_corrected))
    
                # リストに変換
                corrected = [list(landmark) for landmark in corrected]
                #print("corrected:", corrected)


                # ground_truth_landmarksとsample_landmarks_xをTensorに変換
                ground_truth_tensor = torch.tensor(ground_truth_landmarks, dtype=torch.float32)
                sample_landmarks_x_tensor = torch.tensor(sample_landmarks_x, dtype=torch.float32)
                # MSELossを計算
                loss = criterion(ground_truth_tensor, sample_landmarks_x_tensor)
                #print(f'MSE Loss(sample): {loss.item():.4f}')

            
                # ground_truth_landmarksとcorrectedをTensorに変換
                ground_truth_tensor = torch.tensor(ground_truth_landmarks, dtype=torch.float32)
                corrected_tensor = torch.tensor(corrected, dtype=torch.float32)
                # MSELossを計算
                loss = criterion(ground_truth_tensor, corrected_tensor)
                print(f'MSE Loss(corrected): {loss.item():.4f}')


          

                # プロットの設定
                save_path = f'/home/iwata/Pictures/hand_{sample_index}_n{n_seq}_r{i+1}_w{window_size}_{n_epochs}.png'
                fig, ax = plt.subplots(figsize=(8, 8))
                #ax.set_aspect('equal', 'box')
                #print("ground_truth_landmarks:", ground_truth_landmarks)
                #print("sample_landmarks_x:",sample_landmarks_x)

                # 手の座標点の順序を指定するリスト（例）
                custom_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

                # Ground Truthの手の線をプロット
                for points in [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12], [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [0, 17]]:
                    x_points = [ground_truth_landmarks[i][0] for i in points]
                    y_points = [ground_truth_landmarks[i][1] for i in points]
                    ax.plot(x_points, y_points, linestyle='-', color='blue', linewidth=2)
                '''
                # sample_landmarks_xの手の線をプロット
                for points in [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12], [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [0, 17]]:
                    x_points = [sample_landmarks_x[i][0] for i in points]
                    y_points = [sample_landmarks_x[i][1] for i in points]
                    ax.plot(x_points, y_points, linestyle='-', color='red', linewidth=2)
                '''
                
                # correctedの手の線をプロット
                for points in [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12], [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [0, 17]]:
                    x_points = [corrected[i][0] for i in points]
                    y_points = [corrected[i][1] for i in points]
                    ax.plot(x_points, y_points, linestyle='-', color='red', linewidth=2)
                    
                # truth_landmarksの手の線をプロット
                for points in [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12], [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [0, 17]]:
                    x_points = [truth_landmarks[i][0] for i in points]
                    y_points = [truth_landmarks[i][1] for i in points]
                    ax.plot(x_points, y_points, linestyle='-', color='green', linewidth=2)
                
        
                #plt.legend()

                # 画像を保存
                plt.savefig(save_path)

                # 画像を表示
                plt.show()



            