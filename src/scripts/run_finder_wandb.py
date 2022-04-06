from collections import defaultdict

import wandb
from tqdm import tqdm


def run():
    number_of_agents = 8
    area = "downtown"  # docklands, downtown, queensberry

    relevant_metric = "validation_advanced_metrics/fined_resources"  # violation_catched_quota fined_resources
    #relevant_metric = "validation_advanced_metrics/violation_catched_quota"
    #relevant_metric = "test_advanced_metrics/fined_resources"
    #relevant_metric = "test_final_advanced_metrics/fined_resources"
    api = wandb.Api(timeout=40)

    runs = filter_runs(api, area, number_of_agents)
    best_metric, best_run, best_row, runs_and_best_row_sorted = find_best_run_in_runs(runs, relevant_metric)

    for item in runs_and_best_row_sorted:
        metric, _, run = item
        print(f"metric: {round(metric, ndigits=2)} : {run.name}")

    bests_by_its_kind = find_bests_of_its_kind(runs_and_best_row_sorted)
    print_bests_of_its_kind(bests_by_its_kind)

    best_run_fully_independent = api.run('nst/mtop/ab5esr9d')

    test_viol_catched_quota_dif = best_run.summary["test_advanced_metrics/violation_catched_quota"] \
                                  - best_run_fully_independent.summary["test_advanced_metrics/violation_catched_quota"]
    test_fined_res_dif = best_run.summary["test_advanced_metrics/fined_resources"] \
                         - best_run_fully_independent.summary["test_advanced_metrics/fined_resources"]

    _, full_row_best_run = scan_history_for_metric(best_run, relevant_metric, only_fetch_relevant_metric=False)
    _, full_row_best_fully_independent = scan_history_for_metric(best_run_fully_independent,
                                                                 relevant_metric,
                                                                 only_fetch_relevant_metric=False)

    val_viol_catched_quota_dif = full_row_best_run["validation_advanced_metrics/violation_catched_quota"] \
                                 - full_row_best_fully_independent[
                                     "validation_advanced_metrics/violation_catched_quota"]

    val_fined_res_dif = full_row_best_run["validation_advanced_metrics/fined_resources"] \
                        - full_row_best_fully_independent["validation_advanced_metrics/fined_resources"]

    c = 3


def filter_runs(api, area, number_of_agents):
    filters = {"$and":
                   [{"config.number_of_agents": number_of_agents},
                    {"tags": area}
                    ]}
    runs = api.runs(path="nst/mtop", filters=filters)
    return runs


def find_best_run_in_runs(runs, relevant_metric):
    best_metric = -99999999999.0
    best_run = None
    best_row = None

    all_runs_with_best_metrics = []

    for run in tqdm(runs):
        result = scan_history_for_metric(run, relevant_metric)
        if not result:
            continue

        highest_metric_value, row_with_highest_metric = result

        all_runs_with_best_metrics.append((highest_metric_value, row_with_highest_metric, run))

        if highest_metric_value > best_metric:
            best_run = run
            best_metric = highest_metric_value
            best_row = row_with_highest_metric

    runs_and_best_row_sorted = sorted(all_runs_with_best_metrics, key=lambda t: t[0], reverse=True)

    return best_metric, best_run, best_row, runs_and_best_row_sorted


def scan_history_for_metric(run, relevant_metric, only_fetch_relevant_metric=True):
    if only_fetch_relevant_metric:
        history = run.scan_history(keys=[relevant_metric])  # only fetch these keys and only rows that have all keys
    else:
        history = run.scan_history()

    relevant_rows = [row for row in history if relevant_metric in row]
    relevant_rows_sorted_by_metric = sorted(relevant_rows, key=lambda row: row[relevant_metric], reverse=True)

    if len(relevant_rows_sorted_by_metric) == 0:
        return None

    row_with_highest_metric = relevant_rows_sorted_by_metric[0]
    highest_metric_value = row_with_highest_metric[relevant_metric]

    return highest_metric_value, row_with_highest_metric


def find_bests_of_its_kind(runs_and_best_row_sorted):
    bests_by_its_kind = defaultdict(list)

    for item in runs_and_best_row_sorted:
        best_metric, row_with_best_metric, run = item
        tags_of_run = run.tags

        if "mardam" in tags_of_run:  # mardam
            if "use_other_agent_features" in run.config and run.config["use_other_agent_features"] == False:
                bests_by_its_kind["mardam_no_oaf"].append(item)
            else:
                bests_by_its_kind["mardam"].append(item)
        elif "grcn_shared_agent" in tags_of_run:  # shared ind
            bests_by_its_kind["shared_ind"].append(item)
        elif "grcn_twin_att" in tags_of_run:  # ours
            bests_by_its_kind["grcn_twin"].append(item)
        elif "greed_wait" in tags_of_run:  # greedy wait
            bests_by_its_kind["greedy_wait"].append(item)
        elif "greedy" in tags_of_run and not ("greed_wait" in tags_of_run):  # greedy no wait
            bests_by_its_kind["greedy_no_wait"].append(item)
        else:  # unknown/not specified
            bests_by_its_kind["rest"].append(item)


    return bests_by_its_kind


def print_bests_of_its_kind(bests_by_its_kind):
    for kind in bests_by_its_kind.keys():
        print(f"---- {kind} ----")

        for item in bests_by_its_kind[kind]:
            metric, row_with_best_metric, run = item
            print(f"metric: {round(metric, ndigits=2)} : {run.name}")

        print("------------------------")


if __name__ == '__main__':
    run()
