import submitit
import main_c
import argparse


def parse_args():
    training_args = main_c.get_arg_parser()
    parser = argparse.ArgumentParser("Submitit MNIST", parents=[training_args])
    parser.add_argument("--array_parallelism", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--partition", type=str, default="gpu_shared")
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--job_dir", type=str, default="job_dir")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    executor = submitit.AutoExecutor(folder=args.job_dir)
    executor.update_parameters(
        name="MNIST",
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=3,
        timeout_min=args.timeout,
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        slurm_array_parallelism=args.array_parallelism,
    )

    pos_encodings = ['grid', 'naive', 'none']
    with executor.batch():
        for pos_encoding in pos_encodings:
            args.pos_encoding = pos_encoding
            job = executor.submit(main_c.train, args)