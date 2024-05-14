import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sns.set_context(font_scale=1.25)

# # cost function - MAE, MSE, Huber

# df = pd.read_csv("../../01_data/02_output/02_case_2/cost_function/metrics/losses.csv")
# df = df.rename(columns={'test_name': 'test', 'split': 'conjunto'})
# df = df.melt(id_vars=['test', "niter"], value_vars=["test_metrics", "train_metrics"], var_name="conjunto", value_name="loss")
# df['test'] = df['test'].apply(lambda x: x.replace("/", ""))
# df = df[df["test"] != "mae_L1"]

# fig, ax = plt.subplots()
# sns.lineplot(data=df, x="niter", y="loss", hue="test", style="conjunto", dashes=[(2,2), (1,0)])
# ax.set_xlabel("Épocas")
# plt.grid()
# ax.set_ylabel("MSE")
# ax.set_yscale('log')
# plt.tight_layout()
# plt.show()
# fig.savefig("../../01_data/02_output/02_case_2/cost_function/figures/metrics_datos.pdf")

# # # cost function - L2, L1

# df = pd.read_csv("../../01_data/02_output/02_case_2/cost_function/metrics/losses.csv")
# df = df.rename(columns={'test_name': 'test', 'split': 'conjunto'})
# df = df.melt(id_vars=['test', "niter"], value_vars=["test_metrics", "train_metrics"], var_name="conjunto", value_name="loss")
# df['test'] = df['test'].apply(lambda x: x.replace("/", ""))
# df = df[df['test'].str.contains("mae|mae_L1") ]

# fig, ax = plt.subplots()
# plt.grid()
# sns.lineplot(data=df, x="niter", y="loss", hue="test", style="conjunto", dashes=[(2,2), (1,0)])
# ax.set_xlabel("Épocas")
# ax.set_ylabel("MSE")
# ax.set_yscale('log')
# plt.tight_layout()
# plt.show()
# fig.savefig("../../01_data/02_output/02_case_2/cost_function/figures/metrics_pesos.pdf")


# # # entrenamiento progresivo

# df_original = pd.read_csv("../../01_data/02_output/02_case_2/cost_function/metrics/losses.csv")
# df_original = df_original[df_original['test_name'] == "mae/"]
# df = pd.read_csv("../../01_data/02_output/02_case_2/entrenamiento_progresivo/metrics/losses.csv")
# df = pd.concat([df, df_original])
# df = df.rename(columns={'test_name': 'test', 'split': 'conjunto'})
# df = df.melt(id_vars=['test', "niter"], value_vars=["test_metrics", "train_metrics"], var_name="conjunto", value_name="loss")
# df['test'] = df['test'].apply(lambda x: x.replace("/", ""))

# fig, ax = plt.subplots(figsize=(9,6))
# sns.lineplot(data=df, x="niter", y="loss", hue="test", style="conjunto", dashes=[(2,2), (1,0)])
# # sns.move_legend(
# #     ax, "lower center",
# #     bbox_to_anchor=(0.5, 1), title=None, frameon=True, ncol=2
# # )
# plt.legend(loc="lower left")
# ax.set_xlabel("Épocas")
# ax.set_ylabel("MSE")
# ax.set_yscale('log')
# plt.grid()
# plt.tight_layout()
# plt.show()
# fig.savefig("../../01_data/02_output/02_case_2/entrenamiento_progresivo/figures/metrics_L2L1.pdf")


# # entrenamiento progresivo - grid search

# df = pd.read_csv("../../01_data/02_output/02_case_2/grid_search/metrics/losses.csv")
# df = df.rename(columns={'test_name': 'test', 'split': 'set'})
# # print(df[df.index == df['train_metrics'].argmin()]['test'].tolist()[0])
# best_configuration = df[df.index == df['train_metrics'].argmin()]['test'].tolist()[0]
# df.loc[df['test'] == best_configuration, 'configuration'] = 'mejor'
# df.loc[df['test'] != best_configuration, 'configuration'] = 'resto'

# df = df.melt(id_vars=['configuration', "niter"], value_vars=["test_metrics", "train_metrics"], var_name="conjunto", value_name="loss")

# df = df.rename(columns={'configuration': 'configuración'})
# df['conjunto'] = df['conjunto'].apply(lambda x: 'entrenamiento' if x == "train_metrics" else "test")

# fig, ax = plt.subplots()
# sns.lineplot(data=df, x="niter", y="loss", hue="configuración", style="conjunto")
# ax.set_xlabel("Épocas")
# ax.set_ylabel("MSE")
# ax.set_yscale('log')
# plt.show()
# fig.savefig("../../01_data/02_output/02_case_2/grid_search/figures/grid_search.png")

# neuronas n

# fpath = "../../01_data/02_output/02_case_2/1_system/"
# df_original = pd.read_csv(fpath + "metrics/losses.csv")
# df = df_original.rename(columns={'split': 'conjunto', 'test_name': 'no_neuronas'})
# df = df[df.apply(lambda row: "architecture_1" in row[1].split("__"), axis=1)]
# df["no_neuronas"] = df["no_neuronas"].str.replace("/", "").str.replace("architecture_1__", "")
# df = df.melt(id_vars=['no_neuronas', 'epochs'], value_vars=["test_metric", "train_metric"], var_name="conjunto", value_name="loss")
# df["conjunto"] = df["conjunto"].apply(lambda x: "entrenamiento" if x == "train_metric" else "test")

# df2 = df[df["epochs"] >= 15]
# df2 = df2.groupby(["no_neuronas", "conjunto"]).agg(loss = ("loss", "min")).reset_index()
# df2.loc[ df2['conjunto'] == "test",'loss'] = df.loc[df["loss"] == df2[df2["conjunto"] == "entrenamiento"]["loss"], "loss"]
# df2["no_neuronas"] = df2["no_neuronas"].astype(int)
# fig, ax1 = plt.subplots(nrows=1)
# ax1.set_xlabel("n")
# ax1.set_ylabel("min MSE")
# sns.lineplot(data=df2, x="no_neuronas", y="loss", style="conjunto", estimator=np.min, ax=ax1, marker="o")
# plt.legend(loc='upper right')
# plt.show()
# fig.savefig(fpath+"figures/mse_vs_n.png")

# root_dir = "../../01_data/02_output/02_case_2/1_system/"
# architecture = "architecture_1__32/"

# root_dir = "../../01_data/02_output/02_case_2/n_system/"
# architecture = "test_all_n/"

# df = pd.read_csv(root_dir+"metrics/losses.csv")
# df = df[df["test_name"] == architecture]
# fig, ax = plt.subplots(figsize=(7,3))
# plt.plot(df["epochs"], df["train_metric"],  "k-", label="Entrenamiento",) #  SXS:BBH:0217
# plt.plot(df["epochs"], df["test_metric"],  "k--", label="Test",) # SXS:BBH:0211
# plt.xlabel("Épocas")
# plt.ylabel("MSE")
# plt.legend()
# plt.grid()
# plt.tight_layout()

# fig.savefig(root_dir+architecture+"losses.pdf")

# m1 \neq m2

fpath = "../../01_data/02_output/02_case_2/1_system/"

df_original = pd.read_csv(fpath + "metrics/losses_copy.csv")
df = df_original.rename(columns={'split': 'conjunto'})
dropout_list = ["prueba_tonta_2/"]
df = df[df['test_name'].isin(dropout_list + ["architecture_1_act__tanh/"])]
df = df.melt(id_vars=['test_name', 'epochs'], value_vars=["test_metric", "train_metric"], var_name="conjunto", value_name="loss")
df["conjunto"] = df["conjunto"].apply(lambda x: "entrenamiento" if x == "train_metric" else "test")
df = df[df["conjunto"] == "entrenamiento"]
df["conjunto"] = df["test_name"].apply(lambda x: "SXS:BBH:0217" if x == "architecture_1_act__tanh/" else "SXS:BBH:0168")

fig, ax = plt.subplots(figsize=(7,3))
ax.set_xlabel("Épocas")
ax.set_ylabel("MSE")
sns.lineplot(data=df, x="epochs", y="loss", hue="conjunto")
plt.legend(title="sistema")
plt.grid()
plt.tight_layout()
plt.show()
fig.savefig(fpath+"figures/mse_epochs_prueba_tonta.png")
fig.savefig(fpath+"figures/mse_epochs_prueba_tonta.pdf")
