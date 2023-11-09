using Pkg; Pkg.activate("../../");
using Plots
using DataFrames
using CSV
gr()

# root_dir = "../../01_data/02_output/02_case_2/1_system/"
# architecture = "architecture_1__32/"

root_dir = "../../01_data/02_output/02_case_2/n_system/"
architecture = "test_all_n/"

df = DataFrame(CSV.File(root_dir*"metrics/losses.csv"))
df = filter(:test_name => n -> n == architecture, df)
plt = plot(df[!, "epochs"], df[!, "train_metric"], label="Entrenamiento", legend=:topright, framestyle=:box, legengfontsize=20, guidefontsize=16) #  SXS:BBH:0217
plot!(plt, df[!, "epochs"], df[!, "test_metric"], label="Test ", margin=5Plots.mm) # SXS:BBH:0211
xlabel!("Épocas")
ylabel!("MSE")
plot!(size=(1200,900))

savefig(plt, root_dir*architecture*"losses.pdf")