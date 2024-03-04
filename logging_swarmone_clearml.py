import time
from clearml import Task

# Set the ClearML credentials and server URLs programmatically
# Task.set_credentials(
#     api_host='http://192.168.23.105:8080',
#     web_host='http://192.168.23.105:8008',
#     files_host='http://192.168.23.105:8081',
#     key='QY8O38W1PR85S6VFCP4W',
#     secret='iWOHfndZ4zj2B1u1UDop1AyZLYf0kKVjYad7YUtTf9Za5Z6gPL'
# )


last_logged = {}
clearml_tasks_task_ids = {}


def init_clearml_task(swarm_task_id, project_name):
    try:
        clearml_task_id = clearml_tasks_task_ids[swarm_task_id]
        clearml_task = Task.get_task(task_id=clearml_task_id)
        return clearml_task
    except KeyError as e:
        clearml_task = Task.create(project_name=project_name, task_name=swarm_task_id,
                                   task_type=Task.TaskTypes.training)
        clearml_tasks_task_ids[swarm_task_id] = clearml_task.task_id
    return clearml_task


def log_metrics(task_id, metrics, clearml_task):
    global last_logged

    if task_id not in last_logged:
        last_logged[task_id] = {}

    for metric, values in metrics.items():
        # Dynamically add new metrics to last_logged if they don't exist
        if metric not in last_logged[task_id]:
            last_logged[task_id][metric] = 0  # Initialize with 0, indicating no values logged yet

        for i, value in enumerate(values):
            # Only log if this index hasn't been logged before
            if i >= last_logged[task_id][metric]:
                if 'epoch' in metric and metric != "epoch_index":
                    clearml_task.get_logger().report_scalar(title=metric, series='epoch',
                                                            value=value, iteration=i)
                elif 'step' in metric and metric != "step_index" and "val" not in metric:
                    clearml_task.get_logger().report_scalar(title=metric, series='step', value=value,
                                                            iteration=metrics["step_index"][i])
                last_logged[task_id][metric] = i + 1  # Update the last logged position to the next index


def log_job_metrics(job_id, project_name, swarm_one_client):
    """
    Main process to fetch, log, and manage metrics for all tasks within a job.
    """
    while True:
        completed_or_aborted_task_ids = []
        job_info = swarm_one_client.get_job_information(job_id)
        all_completed = all(info['status'] in ['COMPLETED', 'ABORTED'] for info in job_info.values())
        job_metrics = swarm_one_client.get_job_history(job_id, current=True)

        for task_id, info in job_info.items():
            if task_id in completed_or_aborted_task_ids:
                continue

            task_metrics = job_metrics.get(task_id, {})
            if task_metrics:
                clearml_task = init_clearml_task(task_id, project_name)
                log_metrics(task_id, task_metrics, clearml_task)
                if info['status'] in ['COMPLETED', 'ABORTED']:
                    completed_or_aborted_task_ids.append(task_id)
                    clearml_task.close()
        if all_completed:
            break  # Exit if all tasks are completed or aborted

        time.sleep(30)  # Wait before the next iteration to check for new metrics