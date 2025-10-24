import requests
import pandas as pd
import os

def request_clinical_data_by_project(project_ids, save_dir, dataset='TCGA-RCC'):
    fields = [
        "case_id",
        "submitter_id",
        "project.project_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.ajcc_pathologic_stage"
        ]

    fields = ",".join(fields)

    cases_endpt = "https://api.gdc.cancer.gov/cases"

    # This set of filters is nested under an 'and' operator.
    filters = {
        "op": "and",
        "content":[
            {
            "op": "in",
            "content":{
                "field": "cases.project.project_id",
                "value": project_ids #"TCGA-KIRP", "TCGA-KIRC", "TCGA-KICH"
                }
            }
        ]
    }

    # A POST is used, so the filter parameters can be passed directly as a Dict object.
    params = {
        "filters": filters,
        "fields": fields,
        "format": "JSON",
        "size": "2000"
        }


    # Send the request to GDC API
    response = requests.post(cases_endpt, json=params)

    # Check if the request was successful
    if response.status_code == 200:
        print("Query successful")
        json_data = response.json()
    else:
        print(f"Query failed with status code: {response.status_code}")
        exit()

    # Extract the clinical data
    cases = json_data['data']['hits']
    print("The number of cases:", len(cases))

    # Convert the clinical data into a pandas DataFrame
    survival_data = []

    for case in cases:
        survival_data.append({
            'case_id': case['case_id'],
            'submitter_id': case['submitter_id'],
            'project_id': case['project']['project_id'],
            'days_to_last_follow_up': case['diagnoses'][0].get('days_to_last_follow_up', None),
            'ajcc_pathologic_stage': case['diagnoses'][0].get('ajcc_pathologic_stage', None),
            'days_to_death': case['demographic'].get('days_to_death', None),
            'vital_status': case['demographic'].get('vital_status', None)
        })

    df = pd.DataFrame(survival_data)

    # Display the first few rows of the survival data
    print(df.head())
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/{dataset}_survival_data.csv", index=False)
    print("Survival data saved to CSV")
    return

def request_survival_data_by_submitter(submitter_ids, save_dir, dataset='TCGA'):
    fields = [
        "case_id",
        "submitter_id",
        "project.project_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.ajcc_pathologic_stage"
    ]

    fields = ",".join(fields)

    cases_endpt = "https://api.gdc.cancer.gov/cases"

    # 🔹 Filter by submitter_id instead of project_id
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": submitter_ids  # e.g., ["TCGA-AB-1234", "TCGA-CD-5678"]
                }
            }
        ]
    }

    params = {
        "filters": filters,
        "fields": fields,
        "format": "JSON",
        "size": "2000"
    }

    response = requests.post(cases_endpt, json=params)

    if response.status_code == 200:
        print("Query successful")
        json_data = response.json()
    else:
        print(f"Query failed with status code: {response.status_code}")
        return

    cases = json_data['data']['hits']
    print("The number of cases:", len(cases))

    survival_data = []
    for case in cases:
        diagnosis = case.get('diagnoses', [{}])[0]  # Safe access
        demographic = case.get('demographic', {})

        survival_data.append({
            'case_id': case.get('case_id'),
            'submitter_id': case.get('submitter_id'),
            'project_id': case.get('project', {}).get('project_id'),
            'days_to_last_follow_up': diagnosis.get('days_to_last_follow_up'),
            'ajcc_pathologic_stage': diagnosis.get('ajcc_pathologic_stage'),
            'days_to_death': demographic.get('days_to_death'),
            'vital_status': demographic.get('vital_status')
        })

    df = pd.DataFrame(survival_data)
    print(df.head())

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/{dataset}_survival_data_by_submitter.csv", index=False)
    print(f"Survival data saved to {save_dir}/{dataset}_survival_data_by_submitter.csv")
