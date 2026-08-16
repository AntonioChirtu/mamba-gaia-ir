import optuna

study = optuna.load_study(study_name="mamba_retrieval_sweep", storage="sqlite:///sweep_copy.db")
df = study.trials_dataframe()

# Show the top 5 trials (assuming you are MINIMIZING loss)
# If you are maximizing accuracy, use .nlargest() instead
top_50 = df.nlargest(60, "value")

# Display trial number, the target value, and the hyperparameter columns
param_cols = [col for col in df.columns if col.startswith("params_")]
print(top_50[["number", "value", "state"] + param_cols])