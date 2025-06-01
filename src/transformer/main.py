import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import use as mpl_use

from care import care

mpl_use('MacOSX')


class AnomalyAttention(nn.Module):
    def __init__(self, dim, heads=2, dropout=0.1):  # Changed heads to 2
        super(AnomalyAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into Q, K, V
        q, k, v = map(lambda t: t.reshape(B, T, self.heads, D // self.heads).transpose(1, 2), qkv)

        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn_probs, v)
        out = out.transpose(1, 2).reshape(B, T, D)

        return self.fc_out(out)


class AnomalyTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, heads=2):  # Changed heads to 2
        super(AnomalyTransformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=heads, dim_feedforward=hidden_dim, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_dim, 1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, input_dim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Make input 3D: [batch, seq_len, feature_dim]

        # Add positional encoding here
        x = x + self.pos_embedding[:, :x.size(1), :]  # pos_embedding shape: [1, max_seq_len, feature_dim]


        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final output layer and activation
        return torch.sigmoid(self.fc(x)).squeeze(-1)


class WeightedAnomalyLoss(nn.Module):
    def __init__(self, anomaly_weight=2.0, normal_weight=1.0):
        super(WeightedAnomalyLoss, self).__init__()
        self.anomaly_weight = anomaly_weight
        self.normal_weight = normal_weight
        self.criterion = nn.BCELoss(reduction='none')  # Use element-wise loss computation

    def forward(self, predictions, targets):
        loss = self.criterion(predictions, targets)
        weights = torch.where(targets == 1, self.anomaly_weight, self.normal_weight)
        weighted_loss = loss * weights

        return weighted_loss.mean()


class RewardedAnomalyLoss(nn.Module):
    def __init__(self, tp_penalty=1, tn_penalty=1, fp_penalty=10, fn_penalty=10):
        super(RewardedAnomalyLoss, self).__init__()
        self.tp_penalty = tp_penalty
        self.tn_penalty = tn_penalty
        self.fp_penalty = fp_penalty
        self.fn_penalty = fn_penalty
        self.criterion = nn.BCELoss(reduction='none')

    def forward(self, predictions, targets):
        loss = self.criterion(predictions, targets)
        weights = torch.ones_like(targets)

        weights[(predictions > 0.5) & (targets == 1)] = self.tp_penalty  # TP penalty
        weights[(predictions < 0.5) & (targets == 0)] = self.tn_penalty  # TN penalty
        weights[(predictions > 0.5) & (targets == 0)] = self.fp_penalty  # FP penalty
        weights[(predictions < 0.5) & (targets == 1)] = self.fn_penalty  # FN penalty

        weighted_loss = loss * weights
        return weighted_loss.mean()


class Dataset(IterableDataset):
    def __init__(self, file_paths, train_prediction, balance=True):
        self.file_paths = file_paths
        self.train_prediction = train_prediction
        self.balance = balance

    def __iter__(self):
        for file_path in self.file_paths:
            # print(file_path)
            data_tensor, labels_tensor = load_dataset(file_path, self.train_prediction, self.balance)
            for x, y in zip(data_tensor, labels_tensor):
                yield x, y, file_path


def get_csv_file_paths(directory, include_ids=None):
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith('.csv') and (
                include_ids is None or int(os.path.splitext(file)[0]) in include_ids
        )
    ]


# def load_dataset(file_path, train_prediction, balance):
#     df = pd.read_csv(file_path, delimiter=';')
#
#     # if train_prediction == 'prediction':
#     #     df = df[df['train_test'] == train_prediction]
#
#     event_start_id, event_end_id, label = get_event_info(file_path)
#
#     df['label'] = 0
#     df.loc[(df['id'] >= event_start_id) & (df['id'] <= event_end_id), 'label'] = label
#     feature_columns = [col for col in df.columns if col.endswith('avg')]
#
#     # feature_columns = feature_columns[:46]  # test on other models
#
#     if train_prediction == 'prediction' or not balance:
#         data = df[feature_columns].dropna().values.astype(np.float32)
#         labels = df['label'].dropna().values.astype(np.float32)
#         return torch.tensor(data), torch.tensor(labels)
#
#     # Extract and clean data
#     data = df[feature_columns].dropna()  # Drop NaNs before balancing
#     labels = df['label'].loc[data.index]  # Align labels with cleaned data
#
#     # Count occurrences
#     num_ones = (labels == 1).sum()
#     # num_ones = 2000 if num_ones == 0 else num_ones  # force take a small amount of normal data
#
#     if num_ones == 0:
#         return torch.empty((0, len(feature_columns)), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
#
#     # Ensure there is enough data to balance
#     # Select all ones and randomly sample the same number of zeros
#     ones = data[labels == 1]
#     zeros = data[labels == 0].sample(n=num_ones, random_state=42)
#
#     # Combine and shuffle
#     data_balanced = pd.concat([ones, zeros]).sample(frac=1, random_state=42).reset_index(drop=True)
#     labels_balanced = pd.concat([labels[labels == 1], labels[labels == 0].sample(n=num_ones, random_state=42)]).reset_index(drop=True)
#     # Convert to tensors
#     data_tensor = torch.tensor(data_balanced.values.astype(np.float32))
#     labels_tensor = torch.tensor(labels_balanced.values.astype(np.float32))
#     return data_tensor, labels_tensor


def load_dataset(file_path, train_prediction, balance):
    df = pd.read_csv(file_path, delimiter=';')
    event_start_id, event_end_id, label = get_event_info(file_path)

    df['label'] = 0
    df.loc[(df['id'] >= event_start_id) & (df['id'] <= event_end_id), 'label'] = label
    feature_columns = [col for col in df.columns if col.endswith('avg')]

    if train_prediction == 'prediction' or not balance:
        data = df[feature_columns].dropna().values.astype(np.float32)
        labels = df['label'].dropna().values.astype(np.float32)
        return torch.tensor(data), torch.tensor(labels)

    # Drop rows with NaNs in features before balancing
    df = df.dropna(subset=feature_columns)

    # Filtered data and labels after NaN removal
    data = df[feature_columns]
    labels = df['label']

    num_ones = (labels == 1).sum()

    if num_ones == 0:
        return torch.empty((0, len(feature_columns)), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)

    # Select all positive samples
    ones = data[labels == 1]

    # Filter for status_type_id before selecting zeros
    zeros_df = df[(df['status_type_id'].isin([0, 2])) & (df['label'] == 0)]
    zeros = zeros_df[feature_columns].sample(n=num_ones, random_state=42)

    # Combine without shuffling: positives first, then negatives
    data_balanced = pd.concat([ones, zeros]).reset_index(drop=True)
    labels_balanced = torch.tensor([1]*len(ones) + [0]*len(zeros), dtype=torch.float32)

    # Convert to tensors
    data_tensor = torch.tensor(data_balanced.values.astype(np.float32))
    return data_tensor, labels_balanced


def get_event_info(file_path):
    base_dir = os.path.dirname(os.path.dirname(file_path))
    info_file_path = os.path.join(base_dir, 'event_info.csv')
    event_info = pd.read_csv(info_file_path, delimiter=';')
    event_id = os.path.splitext(os.path.basename(file_path))[0]
    metadata = event_info[event_info['event_id'] == int(event_id)]
    event_label = metadata["event_label"].values[0]
    event_start_id = metadata["event_start_id"].values[0]
    event_end_id = metadata["event_end_id"].values[0]
    label = 1 if event_label == "anomaly" else 0
    return event_start_id, event_end_id, label


def train_anomaly_transformer(model_name, file_paths, loss, batch_size, epochs, lr, input_dim, heads):
    dataset = Dataset(file_paths, 'train')
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = AnomalyTransformer(input_dim=input_dim, hidden_dim=128, num_layers=3, heads=heads)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = loss()

    x = []
    y = []
    fig, ax = plt.subplots()
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        for x_batch, y_batch, _ in dataloader:
            optimizer.zero_grad()
            scores = model(x_batch)

            scores_agg = scores.mean(dim=1)
            loss = criterion(scores_agg, y_batch)

            # loss = criterion(scores, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        x.append(epoch + 1)
        y.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        ax.clear()  # Clears only the axes, keeping the figure and colors stable
        ax.plot(x, y, marker='o', linestyle='-')  # Keep color fixed
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.draw()
        plt.pause(0.1)

    plt.savefig(model_name + ' loss' + '.png', dpi=300)
    plt.close(fig)
    plt.clf()
    return model


def test_anomaly_transformer(model, batch_size, file_paths):
    dataset = Dataset(file_paths, train_prediction='prediction')
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x_batch, y_batch, file_path in dataloader:
            scores = model(x_batch)
            predictions = (scores > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()

    print(f"Test Accuracy: {correct / total:.4f}")


def save_predictions(model, model_name, file_paths):
    dataset = Dataset(file_paths, train_prediction='prediction')
    dataloader = DataLoader(dataset, batch_size=32)

    model.eval()
    with torch.no_grad():
        results = {}
        for x_batch, y_batch, file_path_batch in dataloader:
            scores = model(x_batch)
            scores_agg = scores.mean(dim=1)
            for fp, scr in zip(file_path_batch, scores_agg.detach().cpu().tolist()):
                if fp not in results:
                    print(fp)
                    results[fp] = []
                results[fp].append({'score': scr})

        for fp, results in results.items():
            df_results = pd.DataFrame(results)
            output_file_path = fp.replace('datasets', f'predictions/{model_name}')
            new_dir = os.path.dirname(output_file_path)
            os.makedirs(new_dir, exist_ok=True)
            df_results.to_csv(output_file_path, index=False)
            print(f'Predictions saved to {output_file_path}')


# def plot(data_set, data_model, number):
#     data_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/datasets/{number}.csv'
#     prediction_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/predictions/{data_model}/{number}.csv'
#
#     data = pd.read_csv(data_path, delimiter=';')
#     event_start_id, event_end_id, label = get_event_info(data_path)
#     data['label'] = 0
#     data.loc[(data['id'] >= event_start_id) & (data['id'] <= event_end_id), 'label'] = label
#     scores = pd.read_csv(prediction_path, delimiter=';')
#
#     clipped = scores['score'].clip(lower=0.3286)
#
#     ground_truth = data['label']
#     min_score = clipped.min()
#     max_score = clipped.max()
#
#     # threshold = 0.328981
#     threshold = 0.4987412500
#
#     # Create figure with 2 rows, 1 column
#     fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # 2 rows, 1 column
#
#     # First plot (True Labels)
#     axes[0].plot(ground_truth, color='blue')
#     axes[0].set_title("True Labels")
#
#     # Second plot (Predictions + min/max lines)
#     axes[1].plot(clipped, color='red', label='Score')
#     axes[1].axhline(max_score, color='purple', linestyle='--', label=f'Max: {max_score:.10f}')
#     axes[1].axhline(min_score, color='green', linestyle='--', label=f'Min: {min_score:.10f}')
#     axes[1].axhline(threshold, color='blue', linestyle='--', label=f'Threshold: {threshold:.5f}')
#     axes[1].set_title("Score clipped")
#     axes[1].legend(loc='lower right')
#
#     # Adjust layout to avoid overlap
#     plt.tight_layout()
#
#     # Show plots
#     plt.show()


def calc_care(file_paths, model_name, threshold):
    events = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, delimiter=';')
        # prepare dataframe

        event_start_id, event_end_id, label = get_event_info(file_path)
        df['label'] = 0
        df.loc[(df['id'] >= event_start_id) & (df['id'] <= event_end_id), 'label'] = label

        data_set = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        number = os.path.splitext(os.path.basename(file_path))[0]
        prediction_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/predictions/{model_name}/{number}.csv'
        scores = pd.read_csv(prediction_path, delimiter=';')

        df['prediction'] = (scores['score'] > threshold).astype(int)

        events.append({'normal': True if label == 0 else False, 'data': df})
    c = care.calc(events)
    return c


def find_best_threshold(file_paths):
    best_threshold = 0.3
    best_care = 0.0
    for threshold in np.arange(best_threshold, 0.7, 0.0001):
        care = calc_care(file_paths, threshold=threshold)
        if care > best_care:
            best_care = care
            best_threshold = threshold
        print(threshold, care)

    print(f"Best Threshold: {best_threshold:.2f}, Best F1-score: {best_threshold:.4f}")
    return best_threshold


criteria = [
    nn.MSELoss,
    # WeightedAnomalyLoss,
    # RewardedAnomalyLoss
]

configs = [
    # {
    #     'name': 'Wind Farm A',
    #     'train': [68, 22, 72, 73, 0],
    #     'test': [40, 26, 42, 10, 45, 84, 17, 38, 71, 14, 92, 51],
    #     'input_dim': 46,
    #     'heads': 2
    # },
    # {
    #     'name': 'Wind Farm B',
    #     'train': [34, 7, 53],
    #     'input_dim': 63,
    #     'heads': 3
    # },
    {
        'name': 'Wind Farm C',
        'train': [55, 81, 47, 12, 4, 18, 28, 39, 66, 15, 78, 79, 30],
        'test': [11, 33, 44, 49, 31, 67, 9, 91, 5, 90, 70, 35, 16, 76, 61, 93, 75, 41, 58, 48, 88, 57, 32, 89, 59, 63, 80, 37, 29, 1, 20, 60],
        'input_dim': 238,
        'heads': 2
    }
]

for config in configs:
    for loss in criteria:
        dataset_name = config['name']
        model_name = dataset_name + ' ' + loss.__name__ + '_pos'
        train = config['train']
        input_dim = config['input_dim']
        heads = config['heads']

        print(f'Training {model_name}')
        start_time = time.time()
        file_paths = get_csv_file_paths(f'/Users/roman/Downloads/CARE_To_Compare/{dataset_name}/datasets', include_ids=train)
        # model = train_anomaly_transformer(model_name, file_paths, loss, batch_size=64, epochs=100, lr=0.0001,
        #                                   input_dim=input_dim, heads=heads)
        # torch.save(model, model_name)
        print(f'Model {model_name} trained')

        model = torch.load(model_name, weights_only=False)

        print('Saving predictions')
        test = config['test']
        file_paths = get_csv_file_paths(f'/Users/roman/Downloads/CARE_To_Compare/{dataset_name}/datasets', include_ids=test)
        save_predictions(model, model_name, file_paths)
        print('Predictions saved')

        end_time = time.time()
        elapsed = end_time - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"Training time: {hours}h {minutes}m")


# model = torch.load(model_name, weights_only=False)
# t = find_best_threshold(file_paths)
# calc_care(file_paths, model_name, threshold=t)
