import os
import json
import pandas as pd

def process_metrics(data):
    # Extract metrics that start with 'Recall@'
    recalls = {k: v for k, v in data.items() if "Recall@" in k}
    return recalls

def average_metrics(data):
    # Calculate the average of metrics across all questions in a file
    metrics_sums = {}
    count = len(data)
    if count == 0:
        return {}

    # Initialize sum for each metric key in the first item
    for metrics in data:
        for key, value in metrics.items():
            if key in metrics_sums:
                metrics_sums[key] += value
            else:
                metrics_sums[key] = value

    # Compute average
    return {k: v / count for k, v in metrics_sums.items()}

def read_and_process_json_file(file_path):
    print(f"Reading and process {file_path} ...")
    with open(file_path, 'r') as file:
        data = json.load(file)
        all_recalls = [process_metrics(question_data) for question_data in data.values()]
        return average_metrics(all_recalls)

def process_folder(folder_path):
    results = {}
    datasets = set()
    
    # Read all JSON files and process them
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            parts = file_name.split('_')
            dataset_name = parts[0]
            model_name = '_'.join(parts[2:]).replace('.json', '')
            datasets.add(dataset_name)

            file_path = os.path.join(folder_path, file_name)
            recall_data = read_and_process_json_file(file_path)
            
            if model_name not in results:
                results[model_name] = {}
            results[model_name][dataset_name] = recall_data

    # Convert results dictionary to DataFrame for easier manipulation and display
    all_metrics = sorted({metric for recalls in results.values() for recall in recalls.values() for metric in recall.keys()})
    df_rows = []

    for model, datasets_data in results.items():
        row = {'Model': model}
        for dataset in datasets:
            metrics_data = datasets_data.get(dataset, {})
            for metric in all_metrics:
                row[f'{dataset} {metric}'] = metrics_data.get(metric, '')
        df_rows.append(row)

    df = pd.DataFrame(df_rows)
    df.set_index('Model', inplace=True)
    return df

# Specify the path to the folder containing the JSON files
folder_path = '.'
df = process_folder(folder_path)

interested_metrics = ["WIT Recall@10", "IGLUE Recall@1", "KVQA Recall@5", "MSMARCO Recall@5", "OVEN Recall@5", "LLaVA Recall@1", "Infoseek Recall@5", "Infoseek Pseudo Recall@5", "EVQA Recall@5", "EVQA Pseudo Recall@5", "OKVQA Recall@5", "OKVQA Pseudo Recall@5"]
interested_metrics = [i for i in interested_metrics if i in df.columns]

df = df[interested_metrics]

print(df.to_string())


# Optionally, save the DataFrame to a CSV file
df.to_csv('report.csv')
