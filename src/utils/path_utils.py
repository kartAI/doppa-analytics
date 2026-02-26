def create_hive_virtual_file_path(
        container: str,
        query_id: str,
        run_id: str,
        iteration: int,
        file_name: str
) -> str:
    if iteration == -1:
        iteration = "*"

    return f"az://{container}/query_id={query_id}/run_id={run_id}/iteration={iteration}/{file_name}"
