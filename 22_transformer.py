import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# データの読み込みと前処理
def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    x_data = []
    y_data = []

    for i in range(len(df) - n_seq):
        x_sequence = df.iloc[i:i+n_seq][[f'{j}_{c}' for j in range(num_joints) for c in ['x', 'y', 'z']]].values.flatten()
        y_sequence = df.iloc[i+n_seq][['8_x', '8_y', '8_z']].values.flatten()  # 8番の関節座標だけ取得
        x_data.append(x_sequence)
        y_data.append(y_sequence)

    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)

    return x_data, y_data

# DataLoaderの使用
def create_dataloader(x_train, y_train):
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


# モデルのパラメータ
n_seq = 1
num_joints = 21
input_size = num_joints * 3 * n_seq  # 各関節の座標 (x, y, z) を持つ
hidden_size = 63
output_size = 3  # 8番関節の座標 (x, y, z) のみ
num_layers = 4
batch_size = 36
n_epochs = 100
n_heads = 3


#位置エンコーディングの定義
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#モデルに入力するために次元を拡張する
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Linear(c_in, d_model) 

    def forward(self, x):
        x = self.tokenConv(x)
        return x

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
        d_model, d_input, d_output,
        dim_feedforward = 512, dropout = 0.1, nhead = 8):
        
        super(Transformer, self).__init__()
        

        #エンべディングの定義
        self.token_embedding_src = TokenEmbedding(d_input, d_model)
        self.token_embedding_tgt = TokenEmbedding(d_output, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        #エンコーダの定義
        encoder_layer = TransformerEncoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        encoder_norm = LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layer, 
                                                      num_layers=num_encoder_layers,
                                                      norm=encoder_norm
                                                     )
        
        #デコーダの定義
        decoder_layer = TransformerDecoderLayer(d_model=d_model, 
                                                nhead=nhead, 
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                batch_first=True,
                                                activation='gelu'
                                               )
        decoder_norm = LayerNorm(d_model)
        self.transformer_decoder = TransformerDecoder(decoder_layer, 
                                                      num_layers=num_decoder_layers, 
                                                      norm=decoder_norm)
        
        #出力層の定義
        self.output = nn.Linear(d_model, d_output)
        

    def forward(self, src, tgt, mask_src, mask_tgt):
        #mask_src, mask_tgtはセルフアテンションの際に未来のデータにアテンションを向けないためのマスク
        
        embedding_src = self.positional_encoding(self.token_embedding_src(src))
        memory = self.transformer_encoder(embedding_src, mask_src)
        
        embedding_tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        outs = self.transformer_decoder(embedding_tgt, memory, mask_tgt)
        
        output = self.output(outs)
        return output

    def encode(self, src, mask_src):
        return self.transformer_encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)

    def decode(self, tgt, memory, mask_tgt):
        return self.transformer_decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), memory, mask_tgt)

#マスクの定義

def create_mask(src, tgt):
    
    seq_len_src = src.shape[1]
    seq_len_tgt = tgt.shape[1]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt).to(device)
    mask_src = generate_square_subsequent_mask(seq_len_src).to(device)

    return mask_src, mask_tgt


def generate_square_subsequent_mask(seq_len):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask


# Transformerモデルのアーキテクチャ
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, n_heads):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        transformer_output = self.transformer(x)
        out = self.fc(transformer_output[-1])
        return out

# モデル、損失関数、オプティマイザ
model = TransformerModel(input_size, hidden_size, output_size, num_layers, n_heads).to(device)
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# メインの処理
if __name__ == "__main__":
    # データの読み込みと前処理
    train_csv_path = 'test_10/choki_test_10/choki_test.csv'
    test_csv_path = 'test_10/choki_test_10/choki_test.csv'
    
    x_train, y_train = preprocess_data(train_csv_path)
    x_test, y_test = preprocess_data(test_csv_path)

    # DataLoaderの作成
    train_loader = create_dataloader(x_train, y_train)

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

    # テストデータで評価
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        test_outputs = model(torch.tensor(x_test, dtype=torch.float32).unsqueeze(1).to(device))
        test_loss = criterion(test_outputs, torch.tensor(y_test, dtype=torch.float32).to(device))
        processing_time_per_image = time.time() - start_time

    # 8番の関節座標をプロット
    for sample_index in range(0, len(x_test)):
        x_test_batch = torch.tensor(x_test[sample_index:sample_index+1], dtype=torch.float32).unsqueeze(1).to(device)
        y_test_batch = torch.tensor(y_test[sample_index:sample_index+1], dtype=torch.float32).to(device)

        with torch.no_grad():
            predicted_tensor = model(x_test_batch)
            predicted = predicted_tensor.cpu().numpy()

            ground_truth = y_test[sample_index]
            
            # プロットの設定
            save_path = f'/home/iwata/Pictures/hand_TF_point_{sample_index}_n{n_seq}_{n_epochs}.png'
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # プロットの範囲を統一
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal', 'box')

            ax.scatter(ground_truth[0], ground_truth[1], color='blue', label='Ground Truth')
            ax.scatter(predicted[0][0], predicted[0][1], color='red', label='Predicted')
            
            ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Ground Truth vs Predicted for Joint 8')

            # 画像を保存
            plt.savefig(save_path)

            # 画像を表示
            plt.show()

    print(f'Test Loss: {test_loss.item():.4f}')
    print(f'Training Time: {training_time:.4f} seconds')
    print(f'Processing Time per Image: {processing_time_per_image:.4f} seconds')
