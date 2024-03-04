from swarm_one.pytorch import Client

swarm_one_client = Client(api_key="API_KEY")

import os
import wandb
import time

os.environ["WANDB_DISABLE_SERVICE"] = "True"
os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"

last_logged = {}


def init_wandb_run(task_id, project_name):
    return wandb.init(project=project_name, name=task_id, id=task_id, resume="allow")


def log_metrics(task_id, metrics, epoch=None):
    global last_logged
    # Ensure task_id is initialized in last_logged
    if task_id not in last_logged:
        last_logged[task_id] = {}

    for metric, values in metrics.items():
        # Dynamically add new metrics to last_logged if they don't exist
        if metric not in last_logged[task_id]:
            last_logged[task_id][metric] = 0  # Initialize with 0, indicating no values logged yet

        for i, value in enumerate(values):
            # Only log if this index hasn't been logged before
            if i >= last_logged[task_id][metric]:
                """
                Just for your model metrics
                """


                if 'epoch' in metric and metric != "epoch_index":
                    wandb.log({metric: value, 'epoch_index': i})
                elif 'step' in metric and metric != "step_index" and "val" not in metric:
                    wandb.log({metric: value, 'step_index': metrics["step_index"][i]})
                last_logged[task_id][metric] = i + 1  # Update the last logged position to the next index


def log_job_metrics(job_id, project_name):
    """
    Main process to fetch, log, and manage metrics for all tasks within a job.
    """
    while True:
        job_info = swarm_one_client.get_job_information(job_id)
        all_completed = all(info['status'] in ['COMPLETED', 'ABORTED'] for info in job_info.values())

        for task_id in job_info.keys():
            wandb_run = init_wandb_run(task_id, project_name)
            task_metrics = swarm_one_client.get_job_history(job_id, current=True).get(task_id, {})
            log_metrics(task_id, task_metrics)
            wandb_run.finish()

        if all_completed:
            break  # Exit if all tasks are completed or aborted e.g. finished

        time.sleep(3)  # Wait before the next iteration to check for new metrics

