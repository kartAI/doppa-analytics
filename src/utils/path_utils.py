def create_hive_virtual_file_path(
        container: str,
        query_id: str,
        run_id: str,
        benchmark_run: int,
        iteration: int,
        file_name: str
) -> str:
    if benchmark_run == -1:
        benchmark_run = "*"

    if iteration == -1:
        iteration = "*"

    return f"az://{container}/query_id={query_id}/run_id={run_id}/benchmark_run={benchmark_run}/iteration={iteration}/{file_name}"
