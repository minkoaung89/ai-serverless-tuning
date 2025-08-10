## AWS Lambda PoC



This folder contains the AWS Lambda Proof-of-Concept code used to validate the feasibility of deploying the ML-based auto-tuning logic in a real serverless environment.



### Files

- `lambda_function.py`: Core function with embedded classification logic using thresholds derived from the XGBoost model.

- `cloudwatch_metrics_fetch.py`: Script to query metrics from AWS CloudWatch.



### Setup

1. Deploy `lambda_function.py` in AWS Lambda.

2. Ensure proper IAM permissions for CloudWatch and Lambda updates.

3. Monitor logs via CloudWatch to validate adaptive memory updates.






