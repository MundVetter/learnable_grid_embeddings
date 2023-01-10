import submitit
import rotations_test
import argparse


def parse_args():
    training_args = rotations_test.get_arg_parser()
    parser = argparse.ArgumentParser("Submitit MNIST", parents=[training_args])
    parser.add_argument("--array_parallelism", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--partition", type=str, default="gpu_shared")
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--job_dir", type=str, default="job_dir")
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--cpus_per_task", type=int, default=10)
    parser.add_argument('--standard', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    executor = submitit.AutoExecutor(folder=args.job_dir)
    executor.update_parameters(
        name=f"MNIST_{args.test_rotation}",
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=args.cpus_per_task,
        timeout_min=args.timeout,
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.array_parallelism,
    )
    executor.submit(rotations_test.train, args)