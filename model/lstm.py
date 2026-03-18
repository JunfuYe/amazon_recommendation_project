import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random

# 1.Vector Retrieval LSTM
class VectorRetrievalLSTM(nn.Module):
    def __init__(self, visual_dim=2048, hidden_size=256):
        super(VectorRetrievalLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=visual_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, visual_dim)

    def forward(self, visual_seq):
        # visual_seq shape: [batch_size, seq_len, visual_dim]
        lstm_out, (h_n, c_n) = self.lstm(visual_seq)
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        attn_scores = self.attention_net(lstm_out) # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(lstm_out * attn_weights, dim=1) # [batch_size, hidden_size]
        context_vector = self.layer_norm(context_vector)
        pred_vector = self.fc_out(context_vector) # [batch_size, visual_dim]

        return pred_vector

# 2. Dataset with User ID Tracking
class VectorDataset(Dataset):
    def __init__(self, user_histories, user_ids, item_features_dict, max_seq_len=5):
        self.user_histories = user_histories
        self.user_ids = user_ids # Record which user each sequence belongs to
        self.item_features_dict = item_features_dict
        self.max_seq_len = max_seq_len
        self.visual_dim = 2048

    def __len__(self): return len(self.user_histories)

    def __getitem__(self, idx):
        history = self.user_histories[idx]
        user_id = self.user_ids[idx] # Which specific user is it

        if len(history) < 2: return self.__getitem__((idx + 1) % len(self))

        target_asin = str(history[-1])
        input_asins = history[:-1][-self.max_seq_len:]

        visual_seq = [self.item_features_dict.get(str(asin), torch.zeros(self.visual_dim)) for asin in input_asins]
        pad_len = self.max_seq_len - len(input_asins)
        for _ in range(pad_len): visual_seq.insert(0, torch.zeros(self.visual_dim))

        target_vector = self.item_features_dict.get(target_asin, torch.zeros(self.visual_dim))

        return {
            'user_id': user_id,
            'history_asins': "|".join([str(a) for a in input_asins]),
            'visual': torch.stack(visual_seq),
            'target_vector': target_vector,
            'target_asin': target_asin
        }

# 3. Training and Retrieval Main Program
def run_vector_retrieval_system():
    print("Launch the ultimate architecture based on feature vector retrieval")

    from google.colab import drive
    drive.mount('/content/drive')

    base_path = '/content/drive/My Drive/test/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading data
    df_map = pd.read_csv(base_path + 'item_feature_index.csv')
    if 'row_idx' not in df_map.columns and 'Unnamed: 0' in df_map.columns: df_map.rename(columns={'Unnamed: 0': 'row_idx'}, inplace=True)
    features_array = np.load(base_path + 'item_features_resnet50.npy', allow_pickle=True)

    item_features_dict = {}
    candidate_asins, candidate_vectors = [], []
    for _, row in df_map.iterrows():
        r_idx = int(row.get('row_idx', 0))
        asin = str(row['item_id'])
        if r_idx < features_array.shape[0]:
            vec = torch.tensor(features_array[r_idx]).float()
            item_features_dict[asin] = vec
            candidate_asins.append(asin)
            candidate_vectors.append(vec)

    candidate_matrix = torch.stack(candidate_vectors).to(device) # Table 2 Product Feature Library

    try: df1 = pd.read_csv(base_path + 'table1.csv')
    except: df1 = pd.read_csv(base_path + 'table1.csv', encoding='ISO-8859-1')
    df1.rename(columns={'user_id': 'reviewerID', 'parent_asin': 'asin', 'timestamp': 'unixReviewTime'}, inplace=True)
    if 'unixReviewTime' in df1.columns: df1.sort_values(['reviewerID', 'unixReviewTime'], inplace=True)

    # Extract user history and corresponding user IDs
    grouped = df1.groupby('reviewerID')['asin'].apply(list).reset_index()
    valid_groups = grouped[grouped['asin'].map(len) >= 3]
    user_ids = valid_groups['reviewerID'].tolist()
    user_histories = valid_groups['asin'].tolist()

    dataset = VectorDataset(user_histories, user_ids, item_features_dict)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # training model
    print("Start training vector prediction model")
    model = VectorRetrievalLSTM(visual_dim=2048).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['visual'].to(device))
            loss = criterion(pred, batch['target_vector'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, MSE Loss: {total_loss/len(train_loader):.4f}")

    # saving model
    model_save_path = base_path + 'vector_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"The model has been saved to: {model_save_path}")

    model.eval()

    try: df2 = pd.read_csv(base_path + 'table2.csv')
    except: df2 = pd.read_csv(base_path + 'table2.csv', encoding='ISO-8859-1')

    # Extract a sample for testing
    sample = val_dataset[random.randint(0, len(val_dataset)-1)]
    current_user_id = sample['user_id']
    history_str = sample['history_asins']

    print(f"Current test user ID: {current_user_id}")
    print(f"The user has purchased products in the past ASIN: {history_str}\n")

    with torch.no_grad():
        pred_vector = model(sample['visual'].unsqueeze(0).to(device))
        similarities = F.cosine_similarity(pred_vector, candidate_matrix, dim=1)
        top3_scores, top3_indices = torch.topk(similarities, 3)

    for i in range(3):
        idx = top3_indices[i].item()
        score = top3_scores[i].item()
        rec_asin = candidate_asins[idx]

        row = df2[df2['asin'] == rec_asin]
        title = str(row.iloc[0]['title'])[:40] if not row.empty else "unknown title"

        print(f"recommend #{i+1} | Similarity matching score: {score:.4f}")
        print(f"ASIN: {rec_asin}")
        print(f"merchandise: {title}\n")

if __name__ == '__main__':
    run_vector_retrieval_system()
