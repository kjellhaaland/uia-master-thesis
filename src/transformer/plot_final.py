import os
import pandas as pd
import matplotlib.pyplot as plt


def get_csv_file_paths(directory, include_ids=None):
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith('.csv') and (
                include_ids is None or int(os.path.splitext(file)[0]) in include_ids
        )
    ]


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

    # data.loc[(data['status_type_id'].isin([1, 3, 4, 5])), 'label'] = 1
    mask = data['status_type_id'].isin([1, 3, 4, 5])

    # read scores
    scores = pd.read_csv(prediction_path, delimiter=';')


    ground_truth = data['label']
    # clipped = scores['score'].clip(lower=0.3289806)
    predictions = (scores['score'] > threshold).astype(int)

    # set predictions to be 0 where status id was abnormal
    # predictions[mask] = 0

    # Create figure with 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # 2 rows, 1 column

    # First plot (True Labels)
    axes[0].plot(ground_truth, color='blue')
    axes[0].set_title("True Labels")

    # Second plot (Predictions)
    axes[1].plot(predictions, color='red')
    axes[1].set_title("Anomalies")

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show plots
    # plt.savefig(data_model + ' anomalies.pdf', dpi=300)
    plt.show()


configs = [
    # {'name': 'Wind Farm A', 'number': 40,
    #  'include': [40, 26, 42, 10, 45, 84, 17, 38, 71, 14, 92, 51],
    #  'threshold': {'MSELoss': 0.996577, 'WeightedAnomalyLoss': 0.99966, 'RewardedAnomalyLoss': 0.997675}},
    # {'name': 'Wind Farm B', 'number': 27, 'include': [19, 27, 77, 2, 23, 87, 74, 86, 82],
    #  'threshold': {'MSELoss': 0.99665, 'WeightedAnomalyLoss': 0.999721, 'RewardedAnomalyLoss': 0.9965}},
    {'name': 'Wind Farm C', 'number': 33,
     'include':
        [11, 33, 44, 49, 31, 67, 9, 91, 5, 90, 70, 35, 16,
         76, 61, 93, 75, 41, 58, 48, 88, 57, 32, 89, 59, 63,
         80, 37, 29, 1, 20, 60],
     'threshold': {'MSELoss': 0.89, 'WeightedAnomalyLoss': 0.9996, 'RewardedAnomalyLoss': 0.6000}},
]
criteria = [
    'MSELoss',
    # 'WeightedAnomalyLoss',
    # 'RewardedAnomalyLoss'
]

for config in configs:
    for loss in criteria:
        dataset_name = config['name']
        model_name = dataset_name + ' ' + loss + '_pos'
        number = config['number']
        threshold = config['threshold'][loss]
        include = config['include']
        file_paths = get_csv_file_paths(f'/Users/roman/Downloads/CARE_To_Compare/{dataset_name}/datasets', include)
        plot(dataset_name, model_name, number, threshold)
