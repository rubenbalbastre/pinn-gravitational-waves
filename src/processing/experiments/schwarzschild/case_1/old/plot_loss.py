import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from  matplotlib.ticker import FuncFormatter

# sns.set_style('whitegrid')
# sns.set_context("paper", font_scale = 1.25) # 0.75 
sns.set_context(font_scale=1.25)

# ---------------------------------------------------------

# OPTIMISER

# fpath = "../../01_data/02_output/01_case_1/check_optimisers/"
# df = pd.read_csv(fpath + "metrics/losses.csv")
# df['optimizador'] = df['optimiser'].apply(lambda x: x.split("_")[0])
# df = df.rename(columns={'split': 'conjunto', 'test_metric': 'test', 'train_metric': 'entrenamiento'})
# df = df.melt(id_vars=['optimizador', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")
# df["optimizador"] = df["optimizador"].apply(lambda x: "SGD" if x == "Descent" else x)
# fig, ax = plt.subplots(figsize=(9,6))
# sns.lineplot(data=df, x="epochs", y="loss", hue="optimizador", style="conjunto", estimator="min", ci=None, dashes=[(2,2), (1,0)])
# ax.set_yscale('log')
# ax.set_ylabel('MSE')
# ax.set_xlabel('Épocas')
# plt.grid()
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()
# fig.savefig(fpath+"optimisers.pdf")

# ---------------------------------------------------------

# COST FUNCTION - DATOS

# fpath = "../../01_data/02_output/01_case_1/check_cost_function/"
# df_original = pd.read_csv(fpath + "metrics/losses.csv")
# df = df_original.rename(
#     columns={
#         'split': 'conjunto', 'funcion': 'penalización', 'test_metric': 'test', 'train_metric': 'entrenamiento'
#         }
#     )
# df = df.melt(id_vars=['penalización', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")

# fig, ax = plt.subplots(figsize=(6,9))
# act_f = df["penalización"].unique().tolist()
# colors = ["C"+str(index) for index in range(len(act_f))]
# for index, f in enumerate(act_f):
#     c = 5# 10
#     lw=c-(c-1)*(index/len(act_f))**(1)
#     train = df[(df["penalización"] == f) & (df["conjunto"] == "entrenamiento")]
#     ls=['-','--','-.',':'][index%4]
#     plt.plot(train["epochs"], train["loss"], linewidth=lw, label=f, color=colors[index])
#     test = df[(df["penalización"] == f) & (df["conjunto"] == "test")]
#     ls=['-','--','-.',':'][index%4]
#     plt.plot(test["epochs"], test["loss"], linewidth=lw+1, linestyle="dotted", color=colors[index])

# ax.set_yscale('log')
# ax.set_ylabel('MSE')
# ax.set_xlabel('Épocas')
# leg1 = plt.legend(loc='upper right', title="penalización")
# import matplotlib.lines as mlines
# red_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="-", label='entrenamiento')
# blue_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="dotted", label='test')
# leg2 = plt.legend(handles=[red_patch, blue_patch], loc="center right", title="Conjunto", bbox_to_anchor=(1.0,0.59)) # (.6,0.89)
# ax.add_artist(leg1)
# ax.add_artist(leg2)
# plt.grid()
# plt.tight_layout()
# plt.show()
# fig.savefig(fpath+"cost_function_datos.pdf")


# ---------------------------------------------------------

# COST FUNCTION - PESOS

# fpath = "../../01_data/02_output/01_case_1/check_cost_function_hyperparams/"
# df_original = pd.read_csv(fpath + "metrics/losses.csv")
# df = df_original.rename(columns={'split': 'conjunto', 'pen': 'penalizacion', 'test_metric': 'test', 'train_metric': 'entrenamiento'})
# df = df.melt(id_vars=['penalizacion', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")
# fig, ax = plt.subplots(figsize=(6,9))
# sns.lineplot(data=df, x="epochs", y="loss", hue="penalizacion", style="conjunto", estimator="min", ci=None, dashes=[(2,2), (1,0)])
# ax.set_yscale('log')
# ax.set_ylabel('MSE')
# ax.set_xlabel('Épocas')
# plt.grid() # linestyle="dashed"
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()
# fig.savefig(fpath+"cost_function_pesos.pdf")


# ---------------------------------------------------------

# test 1 (original architectura - different activation functions)

fpath = "../../01_data/02_output/01_case_1/case_1_system/"
df_original = pd.read_csv(fpath + "metrics/losses_old.csv")
df = df_original.rename(
    columns={
        'split': 'conjunto', 'test_name': 'función de activación', 'test_metric': 'test', 'train_metric': 'entrenamiento'
        }
    )
df = df[(~df['función de activación'].str.contains("adaptative")) & (df["función de activación"].str.contains("test_1"))]
df["función de activación"] = df["función de activación"].str.replace("test_1_", "").str.replace("/", "")
df = df.melt(id_vars=['función de activación', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")
df = df[(df["función de activación"] != "no_input_cos") & (df["función de activación"] != "custom_af")]

fig, ax = plt.subplots(figsize=(6,6))
act_f = df["función de activación"].unique().tolist()
colors = ["C"+str(index) for index in range(len(act_f))]
for index, f in enumerate(act_f):
# sns.lineplot(data=df, x="epochs", y="loss", hue="función de activación", style="conjunto")
    c = 10
    lw=c-(c-1)*(index/len(act_f))**(1)
    train = df[(df["función de activación"] == f) & (df["conjunto"] == "entrenamiento")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(train["epochs"], train["loss"], linewidth=lw, label=f, color=colors[index])
    test = df[(df["función de activación"] == f) & (df["conjunto"] == "test")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(test["epochs"], test["loss"], linewidth=lw+1, linestyle="dotted", color=colors[index])

ax.set_yscale('log')
ax.set_ylabel('MSE')

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
ax.set_xlabel('Épocas')
leg1 = plt.legend(loc='upper right', title="función de activación")
red_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="-", label='entrenamiento')
blue_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="dotted", label='test')
leg2 = plt.legend(handles=[red_patch, blue_patch], loc="center right", title="Conjunto", bbox_to_anchor=(1,0.45))
ax.add_artist(leg1)
ax.add_artist(leg2)
plt.tight_layout()
plt.grid()
plt.show()
fig.savefig(fpath+"figures/cost_function_test_1.pdf")
fig.savefig(fpath+"figures/cost_function_test_1.png")


# ---------------------------------------------------------

# test 2 (simple architectura - different activation functions)

fpath = "../../01_data/02_output/01_case_1/case_1_system/"
df_original = pd.read_csv(fpath + "metrics/losses_old.csv")
df = df_original.rename(
    columns={
        'split': 'conjunto', 'test_name': 'función de activación', 'test_metric': 'test', 'train_metric': 'entrenamiento'
        }
    )
df = df[
    (df['función de activación'].str.contains("simple")) 
    ]
df["función de activación"] = df["función de activación"].str.replace("simple_", "").str.replace("/", "")
df = df.melt(id_vars=['función de activación', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")
df = df[(df["función de activación"] != "no_input_cos") & (df["función de activación"] != "custom_af")]

fig, ax = plt.subplots(figsize=(6,6))
act_f = df["función de activación"].unique().tolist()
colors = ["C"+str(index) for index in range(len(act_f))]
for index, f in enumerate(act_f):
    c = 10
    lw=c-(c-1)*(index/len(act_f))**(1)
    train = df[(df["función de activación"] == f) & (df["conjunto"] == "entrenamiento")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(train["epochs"], train["loss"], linewidth=lw, label=f, color=colors[index])
    test = df[(df["función de activación"] == f) & (df["conjunto"] == "test")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(test["epochs"], test["loss"], linewidth=lw+1, linestyle="dotted", color=colors[index])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
ax.set_yscale('log')
ax.set_ylabel('MSE')
ax.set_xlabel('Épocas')
leg1 = plt.legend(loc='upper right', title="función de activación")
import matplotlib.lines as mlines
red_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="-", label='entrenamiento')
blue_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="dotted", label='test')
leg2 = plt.legend(handles=[red_patch, blue_patch], loc="center right", title="Conjunto", bbox_to_anchor=(1.0,0.45)) # (.6,0.89)
ax.add_artist(leg1)
ax.add_artist(leg2)
plt.tight_layout()
plt.grid()
plt.show()
fig.savefig(fpath+"figures/cost_function_test_2.pdf")
fig.savefig(fpath+"figures/cost_function_test_2.png")


# ---------------------------------------------------------


# test 3


fpath = "../../01_data/02_output/01_case_1/case_1_system/"
df_original = pd.read_csv(fpath + "metrics/losses_old.csv")
df = df_original.rename(
    columns={
        'split': 'conjunto', 'test_name': 'arquitectura', 'test_metric': 'test', 'train_metric': 'entrenamiento'
        }
    )
df = df[df["arquitectura"].str.contains("test_3_cos|test_1_no_input_cos|test_1_cos")]
df["arquitectura"] = df["arquitectura"].str.replace("test", "arquitectura").str.replace("/", "")
df = df.melt(id_vars=['arquitectura', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")
# df = df[(df["arquitectura"] != "no_input_cos") & (df["arquitectura"] != "custom_af")]

fig, ax = plt.subplots(figsize=(6,6))
act_f = df["arquitectura"].unique().tolist()
colors = ["C"+str(index) for index in range(len(act_f))]
for index, f in enumerate(act_f):
    c = 10
    lw=c-(c-1)*(index/len(act_f))**(1)
    train = df[(df["arquitectura"] == f) & (df["conjunto"] == "entrenamiento")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(train["epochs"], train["loss"], linewidth=lw, label=f, color=colors[index])
    test = df[(df["arquitectura"] == f) & (df["conjunto"] == "test")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(test["epochs"], test["loss"], linewidth=lw+1, linestyle="dotted", color=colors[index])

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
ax.set_yscale('log')
ax.set_ylabel('MSE')
ax.set_xlabel('Épocas')
leg1 = plt.legend(loc='upper right', title="arquitectura")
import matplotlib.lines as mlines
red_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="-", label='entrenamiento')
blue_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="dotted", label='test')
leg2 = plt.legend(handles=[red_patch, blue_patch], loc="center right", title="Conjunto", bbox_to_anchor=(1.0,0.61)) # (.6,0.89)
ax.add_artist(leg1)
ax.add_artist(leg2)
plt.tight_layout()
plt.grid()
plt.show()
fig.savefig(fpath+"figures/cost_function_test_3.pdf")
fig.savefig(fpath+"figures/cost_function_test_3.png")


# ---------------------------------------------------------


# test 4

fpath = "../../01_data/02_output/01_case_1/case_1_system/"
df_original = pd.read_csv(fpath + "metrics/losses_old.csv")
df = df_original.rename(
    columns={
        'split': 'conjunto', 'test_name': 'arquitectura', 'test_metric': 'test', 'train_metric': 'entrenamiento'
        }
    )
df = df[df["arquitectura"].str.contains("LSTM|GRU|test_1_cos")]
df["arquitectura"] = df["arquitectura"].str.replace("test", "arquitectura").str.replace("/", "")
df = df.melt(id_vars=['arquitectura', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")

fig, ax = plt.subplots(figsize=(6,6))
act_f = df["arquitectura"].unique().tolist()
colors = ["C"+str(index) for index in range(len(act_f))]
for index, f in enumerate(act_f):
    c = 10
    lw=c-(c-1)*(index/len(act_f))**(1)
    train = df[(df["arquitectura"] == f) & (df["conjunto"] == "entrenamiento")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(train["epochs"], train["loss"], linewidth=lw, label=f, color=colors[index])
    test = df[(df["arquitectura"] == f) & (df["conjunto"] == "test")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(test["epochs"], test["loss"], linewidth=lw+1, linestyle="dotted", color=colors[index])

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
ax.set_yscale('log')
ax.set_ylabel('MSE')
ax.set_xlabel('Épocas')
leg1 = plt.legend(loc='upper right', title="arquitectura")
import matplotlib.lines as mlines
red_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="-", label='entrenamiento')
blue_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="dotted", label='test')
leg2 = plt.legend(handles=[red_patch, blue_patch], loc="center right", title="Conjunto", bbox_to_anchor=(1.0,0.39)) # (.6,0.89)
ax.add_artist(leg1)
ax.add_artist(leg2)
plt.tight_layout()
plt.grid()
plt.show()
fig.savefig(fpath+"figures/cost_function_test_4.pdf")
fig.savefig(fpath+"figures/cost_function_test_4.png")


############################################################

# test 4

fpath = "../../01_data/02_output/01_case_1/case_1_system/"
df_original = pd.read_csv(fpath + "metrics/losses_old.csv")
df = df_original.rename(
    columns={
        'split': 'conjunto', 'test_name': 'arquitectura', 'test_metric': 'test', 'train_metric': 'entrenamiento'
        }
    )
df = df[
    (df["arquitectura"].str.contains("architecture_"))
    ]

df["arquitectura"] = df["arquitectura"].str.replace("architecture_1__", "").str.replace("/", "")
df = df.melt(id_vars=['arquitectura', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")

fig, ax = plt.subplots(figsize=(6,6))
act_f = df["arquitectura"].unique().tolist()
colors = ["C"+str(index) for index in range(len(act_f))]
for index, f in enumerate(act_f):
    c = 10
    lw=c-(c-1)*(index/len(act_f))**(1)
    train = df[(df["arquitectura"] == f) & (df["conjunto"] == "entrenamiento")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(train["epochs"], train["loss"], linewidth=lw, label=f, color=colors[index])
    test = df[(df["arquitectura"] == f) & (df["conjunto"] == "test")]
    ls=['-','--','-.',':'][index%4]
    plt.plot(test["epochs"], test["loss"], linewidth=lw+1, linestyle="dotted", color=colors[index])

ax.set_yscale('log')
ax.set_ylabel('MSE')
ax.set_xlabel('Épocas')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
leg1 = plt.legend(loc='upper right', title="n neuronas")
import matplotlib.lines as mlines
red_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="-", label='entrenamiento')
blue_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="dotted", label='test')
leg2 = plt.legend(handles=[red_patch, blue_patch], loc="center right", title="Conjunto", bbox_to_anchor=(0.8,0.9)) # (.6,0.89)
ax.add_artist(leg1)
ax.add_artist(leg2)
plt.tight_layout()
plt.grid()
plt.show()
fig.savefig(fpath+"figures/cost_function_test_0.pdf")
fig.savefig(fpath+"figures/cost_function_test_0.png")


# case n systems

# fpath = "../../01_data/02_output/01_case_1/case_n_systems/"
# df_original = pd.read_csv(fpath + "metrics/losses.csv")
# df = df_original.rename(
#     columns={
#         'split': 'conjunto', 'test_name': 'arquitectura', 'test_metric': 'test', 'train_metric': 'entrenamiento'
#         }
#     )
# df["arquitectura"] = df["arquitectura"].str.replace("architecture_", "arquitectura_").str.replace("/", "")
# df = df.melt(id_vars=['arquitectura', 'epochs'], value_vars=["test", "entrenamiento"], var_name="conjunto", value_name="loss")

# fig, ax = plt.subplots(figsize=(9,6))
# act_f = df["arquitectura"].unique().tolist()
# colors = ["C"+str(index) for index in range(len(act_f))]
# for index, f in enumerate(act_f):
#     c = 10
#     lw=c-(c-1)*(index/len(act_f))**(1)
#     train = df[(df["arquitectura"] == f) & (df["conjunto"] == "entrenamiento")]
#     ls=['-','--','-.',':'][index%4]
#     plt.plot(train["epochs"], train["loss"], linewidth=lw, label=f, color=colors[index])
#     test = df[(df["arquitectura"] == f) & (df["conjunto"] == "test")]
#     ls=['-','--','-.',':'][index%4]
#     plt.plot(test["epochs"], test["loss"], linewidth=lw+1, linestyle="dotted", color=colors[index])

# ax.set_yscale('log')
# ax.set_ylabel('MSE')
# ax.set_xlabel('Épocas')
# leg1 = plt.legend(loc='lower right', title="arquitectura")
# import matplotlib.lines as mlines
# red_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="-", label='entrenamiento')
# blue_patch = mlines.Line2D([0], [0], color='k', lw=4, linestyle="dotted", label='test')
# leg2 = plt.legend(handles=[red_patch, blue_patch], loc="center right", title="Conjunto", bbox_to_anchor=(.87,0.08)) # (.6,0.89)
# ax.add_artist(leg1)
# ax.add_artist(leg2)
# plt.grid()
# plt.tight_layout()
# fig.savefig(fpath+"metrics_n_systems.pdf")


# # number of neurons of architecture 1

# fpath = "../../01_data/02_output/01_case_1/case_1_system/"
# df_original = pd.read_csv(fpath + "metrics/losses.csv")
# df = df_original.rename(columns={'split': 'conjunto', 'test_name': 'no_neuronas'})
# df = df[df.apply(lambda row: "architecture_1" in row[1].split("__"), axis=1)]
# # df = df[df["no_neuronas"] == "architecture_1__28/"]
# df["no_neuronas"] = df["no_neuronas"].str.replace("/", "").str.replace("architecture_1__", "")
# df = df.melt(id_vars=['no_neuronas', 'epochs'], value_vars=["test_metric", "train_metric"], var_name="conjunto", value_name="loss")
# df["conjunto"] = df["conjunto"].apply(lambda x: "entrenamiento" if x == "train_metric" else "test")

# df2 = df[df["epochs"] >= 15]
# df2 = df2.groupby(["no_neuronas", "conjunto"]).agg(loss = ("loss", "min")).reset_index()
# df2["no_neuronas"] = df2["no_neuronas"].astype(int)
# print(df2)
# fig, ax1 = plt.subplots(nrows=1)
# ax1.set_xlabel("n")
# ax1.set_ylabel("min MSE")
# sns.lineplot(data=df2, x="no_neuronas", y="loss", style="conjunto", estimator=np.min, ax=ax1, marker="o")
# plt.tight_layout()
# plt.legend(loc='upper right')
# plt.show()
# fig.savefig(fpath+"mse_vs_n.png")