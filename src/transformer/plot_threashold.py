import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")


def get_csv_file_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]


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


# def plot(data_set, data_model, number, threshold):
#     data_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/datasets/{number}.csv'
#     prediction_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/predictions/{data_model}/{number}.csv'
#
#     data = pd.read_csv(data_path, delimiter=';')
#     scores = pd.read_csv(prediction_path, delimiter=';')['score']
#
#     # Identify prediction region
#     prediction_region = data[data['train_test'] == 'prediction']
#     if prediction_region.empty:
#         print("No prediction region found.")
#         return
#
#     start_idx = prediction_region.index[0]
#     end_idx = prediction_region.index[-1]
#
#     # Extract scores only in the prediction region
#     pred_scores = scores[start_idx:end_idx + 1].reset_index(drop=True)
#
#     # ------------
#     # extended_start_idx = max(start_idx - 1000, 0)
#     # pred_scores = scores[extended_start_idx:end_idx + 1].reset_index(drop=True)
#     #
#     # highlight_start = start_idx - extended_start_idx
#     # highlight_end = end_idx - extended_start_idx
#     # ---------
#
#     # Categorize
#     above_threshold = [i for i, score in enumerate(pred_scores) if score > threshold]
#     below_threshold = [i for i, score in enumerate(pred_scores) if score <= threshold]
#
#     # Plot
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     ax.axvspan(0, len(pred_scores) - 1, alpha=0.2, color='#d08770', label="Region with anomalies")
#     ax.scatter(above_threshold, pred_scores.iloc[above_threshold], color='#d08770', label="Predicted anomaly", alpha=0.7, s=50)
#     ax.scatter(below_threshold, pred_scores.iloc[below_threshold], color='#81a1c1', label="Predicted normal", alpha=0.7, s=50)
#     ax.axhline(threshold, color='#5e81ac', linestyle='--', label=f"Threshold: {threshold:.4f}")
#
#     ax.set_title('Reconstruction Loss in Prediction Region')
#     ax.set_xlabel("Index in Prediction Region")
#     ax.set_ylabel("Prediction Score")
#     ax.legend()
#     ax.legend(loc='lower right')
#     ax.grid(True)
#     plt.tight_layout()
#     plt.savefig(data_model + '_final.pdf')
#     plt.show()
#     plt.close()


def plot(data_set, data_model, number, threshold):
    data_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/datasets/{number}.csv'
    prediction_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/predictions/{data_model}/{number}.csv'

    data = pd.read_csv(data_path, delimiter=';')
    scores = pd.read_csv(prediction_path, delimiter=';')['score']

    # Anomalous region setup
    event_start_id, event_end_id, label = get_event_info(data_path)
    data['label'] = 0
    data.loc[(data['id'] >= event_start_id) & (data['id'] <= event_end_id), 'label'] = label

    # Filter to prediction region
    prediction_region = data[data['train_test'] == 'prediction'].copy()
    prediction_scores = scores.loc[prediction_region.index]
    x_values = prediction_region['id']

    # Define anomaly sub-region within prediction region
    anomaly_mask = (x_values >= event_start_id) & (x_values <= event_end_id)
    start_idx = x_values[anomaly_mask].iloc[0] if anomaly_mask.any() else None
    end_idx = x_values[anomaly_mask].iloc[-1] if anomaly_mask.any() else None

    # Categorize points
    anomaly_x = x_values[prediction_scores > threshold]
    anomaly_r = prediction_scores[prediction_scores > threshold]
    normal_x = x_values[prediction_scores <= threshold]
    normal_r = prediction_scores[prediction_scores <= threshold]

    # Plot
    fig, axes = plt.subplots(figsize=(10, 8))

    if start_idx is not None and end_idx is not None:
        axes.axvspan(start_idx, end_idx, alpha=0.2, color='#d08770', label="Region with anomalies")

    axes.scatter(normal_x, normal_r, color='#81a1c1', label="Predicted normal", alpha=0.7)
    axes.scatter(anomaly_x, anomaly_r, color='#d08770', label="Predicted anomaly", alpha=0.7)
    axes.axhline(threshold, color='#5e81ac', linestyle='--', label="Threshold")

    axes.set_title('Anomaly Scores for Windfarm C - Dataset 4, MSE loss')
    axes.set_xlabel("Index in dataset")
    axes.set_ylabel("Prediction score")
    axes.legend(loc='lower right')
    axes.grid(True)

    plt.tight_layout()
    plt.savefig(data_model + f' {number} ' + '_prediction_scores.pdf')
    plt.show()
    plt.close()



configs = [
    # {'name': 'Wind Farm A', 'number': 40},
    # {'name': 'Wind Farm B', 'number': 27},
    {'name': 'Wind Farm C', 'number': 60,
     'threshold': {'MSELoss': 0.9904, 'WeightedAnomalyLoss': 0.9996, 'RewardedAnomalyLoss': 0.6000}},
]

criteria = [
    'MSELoss',
    # 'WeightedAnomalyLoss',
    # 'RewardedAnomalyLoss'
]

for config in configs:
    for loss in criteria:
        dataset_name = config['name']
        model_name = dataset_name + ' ' + loss
        threshold = config['threshold'][loss]
        number = config['number']
        file_paths = get_csv_file_paths(f'/Users/roman/Downloads/CARE_To_Compare/{dataset_name}/datasets')
        plot(dataset_name, model_name, number, threshold)
