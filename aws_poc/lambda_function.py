import boto3
import datetime

lambda_client = boto3.client('lambda')
cloudwatch = boto3.client('cloudwatch')

# The Lambda function
FUNCTION_NAME = 'lambda-tuning-demo'

def lambda_handler(event, context):
    # --- Get CloudWatch metrics for the last 5 minutes ---
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(minutes=5)

    # Average Duration (ms)
    duration_metrics = cloudwatch.get_metric_statistics(
        Namespace='AWS/Lambda',
        MetricName='Duration',
        Dimensions=[{'Name': 'FunctionName', 'Value': FUNCTION_NAME}],
        StartTime=start,
        EndTime=end,
        Period=60,
        Statistics=['Average']
    )
    avg_duration = duration_metrics['Datapoints'][0]['Average'] if duration_metrics['Datapoints'] else 0

    # Invocation count
    invoke_metrics = cloudwatch.get_metric_statistics(
        Namespace='AWS/Lambda',
        MetricName='Invocations',
        Dimensions=[{'Name': 'FunctionName', 'Value': FUNCTION_NAME}],
        StartTime=start,
        EndTime=end,
        Period=60,
        Statistics=['Sum']
    )
    invocations = invoke_metrics['Datapoints'][0]['Sum'] if invoke_metrics['Datapoints'] else 0

    # --- Simulated ML-driven logic ---
    # These thresholds are derived from the trained XGBoost forecasting model run offline
    if invocations > 50 or avg_duration > 600:
        decision = "High load"
        new_memory = 512
    elif invocations > 20 or avg_duration > 300:
        decision = "Medium load"
        new_memory = 384
    else:
        decision = "Low load"
        new_memory = 256

    # --- Update Lambda configuration ---
    try:
        lambda_client.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            MemorySize=new_memory
        )
        update_status = f"Updated memory to {new_memory} MB"
    except Exception as e:
        update_status = f"Failed to update memory: {str(e)}"

    # --- Logging for CloudWatch ---
    print(f"Decision: {decision}")
    print(f"Avg Duration: {avg_duration:.2f} ms")
    print(f"Invocations: {invocations}")
    print(update_status)

    # --- Return JSON response ---
    return {
        "avg_duration": avg_duration,
        "invocations": invocations,
        "decision": decision,
        "new_memory": new_memory,
        "update_status": update_status
    }
