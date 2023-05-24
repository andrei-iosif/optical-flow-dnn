from clearml.automation import UniformParameterRange, LogUniformParameterRange, DiscreteParameterRange 
from clearml.automation import HyperParameterOptimizer
from clearml.automation import RandomSearch
 
from clearml import Task

if __name__ == "__main__":
    task = Task.init(project_name='RAFT Semantic Tuning',
                    task_name='optimizer_task',
                    task_type=Task.TaskTypes.optimizer,
                    reuse_last_task_id=False)
    
    optimizer = HyperParameterOptimizer(
        # specifying the task to be optimized, task must be in system already so it can be cloned
        base_task_id='0cbbce95948f409a9695143394e4a6dc',
        # setting the hyper-parameters to optimize
        hyper_parameters=[
            LogUniformParameterRange('Args/semantic_loss_weight', base=10, min_value=-1, max_value=2, step_size=0.5),
            UniformParameterRange('Args/gamma', min_value=0.75, max_value=0.9, step_size=0.05),
            DiscreteParameterRange('Args/lr', [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            DiscreteParameterRange('Args/wdecay', [5e-5, 1e-4, 5e-4])
        ],

        # setting the objective metric we want to maximize/minimize
        objective_metric_title='viper-val-epe',
        objective_metric_series='viper-val-epe',
        objective_metric_sign='min',
        
        # setting optimizer
        optimizer_class=RandomSearch,
        
        # configuring optimization parameters
        execution_queue='default',
        max_number_of_concurrent_tasks=1,
        # optimization_time_limit=120.,
        # compute_time_limit=120,
        total_max_jobs=20,
        min_iteration_per_job=0,
        max_iteration_per_job=5000,
    )

    # start the optimization process
    # this function returns immediately
    optimizer.start_locally()
    # set the time limit for the optimization process (2 hours)
    # optimizer.set_time_limit(in_minutes=120.0)
    # wait until process is done (notice we are controlling the optimization process in the background)
    optimizer.wait()
    # optimization is completed, print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=3)
    print([t.id for t in top_exp])
    # make sure background optimization stopped
    optimizer.stop()
