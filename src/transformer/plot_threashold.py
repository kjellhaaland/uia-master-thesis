import os
import pandas as pd
import matplotlib.pyplot as plt


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


def plot(data_set, data_model, number, threshold):
    data_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/datasets/{number}.csv'
    prediction_path = f'/Users/roman/Downloads/CARE_To_Compare/{data_set}/predictions/{data_model}/{number}.csv'

    data = pd.read_csv(data_path, delimiter=';')
    event_start_id, event_end_id, label = get_event_info(data_path)
    data['label'] = 0
    data.loc[(data['id'] >= event_start_id) & (data['id'] <= event_end_id), 'label'] = label
    scores = pd.read_csv(prediction_path, delimiter=';')

    ground_truth = data['label']
    scores = scores['score']  # .clip(lower=0.65)

    min_score = scores.min()
    max_score = scores.max()

    # threshold = 0.328981
    # threshold = 0.7

    # Create figure with 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # 2 rows, 1 column

    # First plot (True Labels)
    axes[0].plot(ground_truth, color='blue')
    axes[0].set_title("True Labels")

    # Second plot (Predictions + min/max lines)
    axes[1].plot(scores, color='red', label='Score')
    axes[1].axhline(max_score, color='purple', linestyle='--', label=f'Max: {max_score:.2f}')
    axes[1].axhline(min_score, color='green', linestyle='--', label=f'Min: {min_score:.2f}')
    axes[1].axhline(threshold, color='blue', linestyle='--', label=f'Threshold: {threshold:.2f}')
    axes[1].set_title("Score")
    axes[1].legend(loc='lower right')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show plots
    # plt.savefig(data_model + ' threshold.png', dpi=300)
    # plt.close(fig)
    plt.show()


configs = [
    # {'name': 'Wind Farm A', 'number': 40},
    # {'name': 'Wind Farm B', 'number': 27},
    {'name': 'Wind Farm C', 'number': 33,
     'threshold': {'MSELoss': 0.87, 'WeightedAnomalyLoss': 0.9996, 'RewardedAnomalyLoss': 0.6000}},
]

criteria = [
    'MSELoss',
    'WeightedAnomalyLoss',
    'RewardedAnomalyLoss'
]

for config in configs:
    for loss in criteria:
        dataset_name = config['name']
        model_name = dataset_name + ' ' + loss
        threshold = config['threshold'][loss]
        number = config['number']
        file_paths = get_csv_file_paths(f'/Users/roman/Downloads/CARE_To_Compare/{dataset_name}/datasets')
        plot(dataset_name, model_name, number, threshold)
