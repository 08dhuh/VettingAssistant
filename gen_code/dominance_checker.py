import pandas as pd
import json
import os

def check_dominance_rule(csv_file):
    df = pd.read_csv(csv_file)
    violations = []
    
    for index, row in df.iterrows():
        numeric_values = [value for value in row if isinstance(value, (int, float))]
        row_sum = sum(numeric_values)
        
        if row_sum == 0 or pd.isna(row_sum):
            continue
        
        for col in df.columns:
            cell_value = row[col]
            if isinstance(cell_value, (int, float)):
                dominance_ratio = cell_value / row_sum
                if dominance_ratio > 0.8:
                    cell_position = [index, df.columns.get_loc(col)]
                    violations.append({
                        "row": index,
                        "column": col,
                        "cell_value": cell_value,
                        "row_sum": row_sum,
                        "dominance_ratio": dominance_ratio,
                        "cell_position": cell_position
                    })
    
    is_percentage_data = False
    percentage_detection_reason = ""
    failing_columns = set(violation['column'] for violation in violations)
    
    if failing_columns:
        percentage_columns = ['age1', 'age2', 'age3', 'age4']
        if all(col in percentage_columns for col in failing_columns):
            is_percentage_data = True
            percentage_detection_reason = "All failing columns are identified as percentage/share columns."
        else:
            percentage_detection_reason = "Not all failing columns are identified as percentage/share columns."
    
    json_output = {
        "violations": violations,
        "is_percentage_data": is_percentage_data,
        "percentage_detection_reason": percentage_detection_reason,
        "failing_columns": list(failing_columns)
    }
    
    os.makedirs('json', exist_ok=True)
    json_file_path = 'json/dominance_violations.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)
    
    print(f"Total rows checked: {len(df)}")
    print(f"Total violations found: {len(violations)}")
    
    return json_file_path

check_dominance_rule('intermediate/supporting_information_cleaned.csv')