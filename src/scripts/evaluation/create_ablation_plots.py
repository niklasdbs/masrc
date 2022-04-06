import matplotlib.pyplot as plt
import wandb
import seaborn as sns
import pandas as pd

def create_data_frame_for_run(run):
    history = run.scan_history()
    relevant_rows = [row for row in history if relevant_metric in row and row["step"] > 0]

    steps = [row["step"] for row in relevant_rows]
    metric = [row[relevant_metric] for row in relevant_rows]

    df = pd.DataFrame(data={"epoch": steps, "metric": metric})
    df["run"] = run.name
    return df

sns.set_theme(style="darkgrid")
relevant_metric = "validation_advanced_metrics/fined_resources"  # violation_catched_quota fined_resources

best_ours = 'nst/mtop/2lxvt496' #'2lxvt496' #'2d5j0kjv'(tar) # '3ehg2gki' #'/tmp/nst/mtop/2rn1ned1'  #'nst/mtop/318ttjuu' #'/nst/mtop/3iuvujdy' #"nst/mtop/1mwly9ji"
comparison = "nst/mtop/12ze8969"

api = wandb.Api(timeout=60)

best_run_ours = api.run(best_ours)
comparison_run = api.run(comparison)

df_ours = create_data_frame_for_run(best_run_ours)
df_comp = create_data_frame_for_run(comparison_run)

df = pd.concat([df_ours, df_comp], ignore_index=True)

sns.lineplot(x="epoch", y="metric", hue="run", data=df)
plt.show()
a = 3