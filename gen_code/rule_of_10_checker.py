import pandas as pd
import json
import os
import numpy as np

def check_rule_of_10(csv_file, json_file):
    df = pd.read_csv(csv_file)
    violations = []

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            cell_value = df.iat[row, col]
            column_name = df.columns[col]
            if pd.isna(cell_value):
                continue
            if isinstance(cell_value, (np.int64, np.float64)):
                cell_value = float(cell_value)
            if isinstance(cell_value, (int, float)) and cell_value < 10:
                violations.append({
                    "row": row + 1,
                    "column": column_name,
                    "value": cell_value,
                    "position": [row + 1, col + 1]
                })

    if not os.path.exists('json'):
        os.makedirs('json')

    with open(json_file, 'w') as f:
        json.dump({"violations": violations}, f)

    print(f"Total violations found: {len(violations)}")
    for violation in violations:
        print(f"Row: {violation['row']}, Column: {violation['column']}, Value: {violation['value']}, Position: {violation['position']}")

    return json_file

if __name__ == "__main__":
    csv_file_path = 'intermediate/supporting_information_cleaned.csv'
    json_file_path = 'json/rule_of_10_violations.json'
    check_rule_of_10(csv_file_path, json_file_path)