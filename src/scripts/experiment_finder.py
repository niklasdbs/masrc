import glob
import json
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


#metric_to_compare = "validation_advanced_metrics/violation_catched_quota"
#metric_to_compare = "validation/return_mean_sum"

#metric_to_compare = "test/return_mean_sum"
metric_to_compare= "test_advanced_metrics/violation_catched_quota"
#metric_to_compare = "test_advanced_metrics/violation_durations_non_fined_resources"
agent_range = range(1, 16)
metrics = []

for model in ["grcn"]:#"grcn", "large"
    for add_other_agent_targets in ["True", "False"]:
        for agent_number in agent_range:
            runs = glob.glob(
                f"../multirun/*/*/+experiment=ddqn/ddqn,add_other_agents_targets_to_resource={add_other_agent_targets},agent/model@model=ddqn/{model},number_of_agents={agent_number}/")

            if len(runs) == 0:
                print(f"NOT FOUND {model} n_agents:{agent_number}")
                print("")
                continue

            run = sorted(runs)[-1]  # take last run

            file_path = Path(run)

            with open(file_path / "scalars.json", mode="r") as f:
                lines = f.readlines()
                json_lines = [json.loads(line) for line in lines]
                best_scalars = [json_line[metric_to_compare] for json_line in json_lines if metric_to_compare in json_line.keys()]

                # scalars = json.loads(lines[-1])

                if len(best_scalars) == 0: #not metric_to_compare in scalars.keys():
                    print(f"NOT FINISHED {model} n_agents:{agent_number}")
                    print("")
                    continue
                best_result = max(best_scalars)

                # return_mean_sum = scalars[metric_to_compare]

                print(f"ddqn/{model} n_agents:{agent_number} add_target:{add_other_agent_targets}")
                print(f"{metric_to_compare}: {round(best_result, ndigits=2)}")
                print("")

                metrics.append({"method": f"ddqn/{model}_add_target_{add_other_agent_targets}",
                                "n_agents": agent_number,
                                metric_to_compare: best_result})


# for model in ["grcn"]:#, "large"
#     for add_other_agent_targets in ["True", "False"]:
#         for agent_number in agent_range:
#             runs = glob.glob(
#                 f"../multirun/*/*/+experiment=ddqn/ddqn,add_other_agents_targets_to_resource={add_other_agent_targets},agent/model@model=ddqn/{model},number_of_agents={agent_number},shared_reward=True/")
#
#             if len(runs) == 0:
#                 print(f"NOT FOUND {model} n_agents:{agent_number}")
#                 print("")
#                 continue
#
#             run = sorted(runs)[-1]  # take last run
#
#             file_path = Path(run)
#
#             with open(file_path / "scalars.json", mode="r") as f:
#                 lines = f.readlines()
#                 json_lines = [json.loads(line) for line in lines]
#                 best_scalars = [json_line[metric_to_compare] for json_line in json_lines if metric_to_compare in json_line.keys()]
#
#                 # scalars = json.loads(lines[-1])
#
#                 if len(best_scalars) == 0: #not metric_to_compare in scalars.keys():
#                     print(f"NOT FINISHED {model} n_agents:{agent_number}")
#                     print("")
#                     continue
#                 best_result = max(best_scalars)
#
#                 # return_mean_sum = scalars[metric_to_compare]
#
#                 print(f"ddqn/{model} n_agents:{agent_number} add_target:{add_other_agent_targets} shared_reward:True")
#                 print(f"{metric_to_compare}: {round(best_result, ndigits=2)}")
#                 print("")
#
#                 metrics.append({"method": f"ddqn/{model}_add_target_{add_other_agent_targets}_shared_reward",
#                                 "n_agents": agent_number,
#                                 metric_to_compare: best_result})
#
#


#"greedy_without_wait"
for method in [ "mardam", "random","greedy_with_wait","greedy_without_wait"]:
    for agent_number in agent_range:
        # metric_to_compare= "test_advanced_metrics/violation_catched_quota"

        runs = glob.glob(
            f"../multirun/*/*/*/{method},number_of_agents={agent_number}/")

        runs = [run for run in runs if (Path(run) / "scalars.json").exists()]

        if len(runs) == 0:
            print(f"NOT FOUND {method} n_agents:{agent_number}")
            print("")
            continue

        run = sorted(runs)[-1]  # take last run

        file_path = Path(run)
        with open(file_path / "scalars.json", mode="r") as f:
            lines = f.readlines()

            json_lines = [json.loads(line) for line in lines]
            best_scalars = [json_line[metric_to_compare] for json_line in json_lines if
                            metric_to_compare in json_line.keys()]

            # scalars = json.loads(lines[-1])

            if len(best_scalars) == 0: # not metric_to_compare in scalars.keys():
                print(f"NOT FINISHED {method} n_agents:{agent_number}")
                print("")
                continue

            best_result = max(best_scalars)
            # return_mean_sum = scalars[metric_to_compare]
            # metric_to_compare= "validation_advanced_metrics/violation_catched_quota"

            print(f"{method} n_agents:{agent_number}")
            print(f"{metric_to_compare}: {round(best_result, ndigits=2)}")
            print("")

            metrics.append({"method": method,
                            "n_agents": agent_number,
                            metric_to_compare: best_result})


def plot(metrics):
    sns.set_theme(style="darkgrid")
    g = sns.lineplot(data=metrics,
                     x="n_agents",
                     y=metric_to_compare,
                     hue="method")
    g.figure.set_size_inches(10.5, 7.5)
    plt.show()


metrics = pd.DataFrame.from_dict(metrics)
# metrics.plot(x="method")
# metrics.groupby("method").plot(x="n_agents")
plot(metrics)
test = ["ddqn/grcn_add_target_True", "ddqn/grcn_add_target_False"]
plot(metrics.query("method == @test"))

test = ["ddqn/large_add_target_True", "ddqn/large_add_target_False"]
plot(metrics.query("method == @test"))

test = ["ddqn/grcn_add_target_True", "ddqn/grcn_add_target_False"]
plot(metrics.query("method == @test"))

test = ["ddqn/grcn_add_target_True", "ddqn/large_add_target_True", "greedy_with_wait"]
plot(metrics.query("method == @test"))

test = ["ddqn/grcn_add_target_False", "ddqn/large_add_target_False", "greedy_with_wait"]
plot(metrics.query("method == @test"))

a = 3
