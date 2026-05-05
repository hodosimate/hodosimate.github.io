import tools
import os
import timeit
import pandas as pd

pipeline_start = timeit.default_timer()

INPUT_FILE = "pd_country_weighted.csv" 

LOGIT_OUTPUT_FOLDER = "LOGIT_TRANSFORM"
LOGIT_VIS_FOLDER = "LOGIT_VISUALIZATIONS"

GARCH_DATA_FOLDER = "GARCH_RESULTS_DATA"
GARCH_VIS_FOLDER = "GARCH_RESULTS_VISUALS"

RESID_PDF_VIS_FOLDER = "GARCH_RESID_PDF_PLOTS"

MRL_VIS_FOLDER = "EVT_MRL_PLOTS"

EVT_DATA_FOLDER = "EVT_RESULTS_DATA"
EVT_VIS_FOLDER = "EVT_RESULTS_VISUALS"

VINE_DATA_FOLDER = "VINE_RESULTS_DATA"
VINE_VIS_FOLDER = "VINE_RESULTS_VISUALS"

MC_OUTPUT_FOLDER = "MC_SIMULATION_RESULTS"

print(f"--- Loading {INPUT_FILE} ---")
try:
    df = tools.load_csv(INPUT_FILE)
except Exception as e:
    print(e)
    exit()

# 3. Transform
print("Applying Logit Transformation...")
df_logit = tools.apply_logit(df)

# 4. Save Data
saved_path = tools.save_result(df_logit, INPUT_FILE, LOGIT_OUTPUT_FOLDER)
print(f"Data saved to: {saved_path}")

# 5. Generate Visualizations (Silent Mode)
print("Generating visualization images...")
tools.visualize_logit_check(saved_path, output_folder=LOGIT_VIS_FOLDER)

print("\n--- Starting GARCH Workflow ---")
print("Note: Only models passing the Ljung-Box independence test will be saved.")

try:
    tools.perform_garch(
        df=df_logit,
        data_folder=GARCH_DATA_FOLDER,
        vis_folder=GARCH_VIS_FOLDER,
        burn_in=0
    )
except Exception as e:
    print(f"Error during GARCH analysis: {e}")

print("\n--- Generating Residual PDF Plots ---")
resid_file_path = os.path.join(GARCH_DATA_FOLDER, "resid_results.csv")

if os.path.exists(resid_file_path):
    try:
        tools.plot_residual_pdf(
            csv_path=resid_file_path,
            output_folder=RESID_PDF_VIS_FOLDER
        )
    except Exception as e:
        print(f"Error during PDF plotting: {e}")

print("\n--- Starting EVT Mean Residual Life (MRL) Analysis ---")
resid_file_path = os.path.join(GARCH_DATA_FOLDER, "resid_results.csv")

if os.path.exists(resid_file_path):
    try:
        tools.mean_residual_life(
            csv_path=resid_file_path, 
            output_folder=MRL_VIS_FOLDER
        )
    except Exception as e:
        print(f"Error during MRL plotting: {e}")
else:
    print(f"Skipping MRL Analysis: '{resid_file_path}' not found. Did any series pass the GARCH filters?")

resid_file = os.path.join(GARCH_DATA_FOLDER, "resid_results.csv")

custom_thresholds = {

}

if os.path.exists(resid_file):
    try:
        resids = tools.load_csv(resid_file, parse_dates=True, index_col=0)

        tools.fit_semiparametric_distribution(
            df_residuals=resids,
            data_folder=EVT_DATA_FOLDER,
            vis_folder=EVT_VIS_FOLDER,
            threshold_dict=custom_thresholds
        )

        print("EVT Pipeline Complete.")
    except FileNotFoundError:
        print(f"Aborting EVT: Could not locate '{resid_file}' or 'resid_results.csv' anywhere.")
    except Exception as e:
        print(f"Error executing EVT fit: {e}")

print("\n--- Starting R-Vine Copula Workflow ---")

u_values_file = os.path.join(EVT_DATA_FOLDER, "copula_u_values.csv")

if os.path.exists(u_values_file):
    try:
        print("Generating Baseline Empirical Copula Pairs...")
        tools.plot_empirical_copula_pairs(u_values_file, vis_folder=VINE_VIS_FOLDER)

        print("Fitting Baseline R-Vine Copula Structure...")
        baseline_cop, baseline_sum = tools.fit_rvine_filtered(
            u_values_file,
            data_folder=VINE_DATA_FOLDER,
            vis_folder=VINE_VIS_FOLDER
        )

        tools.plot_copula_contours(u_values_file, baseline_sum, vis_folder=VINE_VIS_FOLDER)
        print("Baseline Copula Analysis Complete.")
    except Exception as e:
        print(f"Error during Baseline Copula Analysis: {e}")

    try:
        stress_start, stress_end = "2020-01-01", "2021-01-01"

        print(f"\nGenerating Stressed Empirical Copula Pairs ({stress_start} to {stress_end})...")
        tools.plot_empirical_copula_pairs(
            u_values_file,
            vis_folder=VINE_VIS_FOLDER,
            start_date=stress_start,
            end_date=stress_end
        )

        print("Fitting Stressed R-Vine Copula Structure...")
        stress_cop, stress_sum = tools.fit_rvine_filtered(
            u_values_file,
            start_date=stress_start,
            end_date=stress_end,
            data_folder=VINE_DATA_FOLDER,
            vis_folder=VINE_VIS_FOLDER
        )

        tools.plot_copula_contours(u_values_file, stress_sum, vis_folder=VINE_VIS_FOLDER, start_date=stress_start, end_date=stress_end)

        print("Copula Structure Optimization Complete.")

    except Exception as e:
        print(f"Error during R-Vine Copula analysis: {e}")

else:
    print(f"Skipping Copula Analysis: '{u_values_file}' not found. Did the EVT step complete?")

print(("\n--- Starting Terminal VaR simulation Workflow ---"))

if not os.path.exists(MC_OUTPUT_FOLDER):
    os.makedirs(MC_OUTPUT_FOLDER)

marginals_path = os.path.join(EVT_DATA_FOLDER, "semiparametric_models.pkl")
garch_params_path = os.path.join(GARCH_DATA_FOLDER, "garch_param.csv")
logit_csv_path = os.path.join(LOGIT_OUTPUT_FOLDER, "pd_country_weighted_logit.csv")
vol_csv_path = os.path.join(GARCH_DATA_FOLDER, "vol_results.csv")
resid_csv_path = os.path.join(GARCH_DATA_FOLDER, "resid_results.csv")

baseline_copula_path = os.path.join(VINE_DATA_FOLDER, "rvine_stress_start_end_model.json")

if os.path.exists(baseline_copula_path) and os.path.exists(marginals_path):
    print("\nExecuting BASELINE (Normal Market) Simulation...")
    try:
        base_sims, base_var = tools.simulate_1y_terminal_pd(
            copula_json_path=baseline_copula_path,
            marginals_pkl_path=marginals_path,
            garch_params_csv=garch_params_path,
            logit_csv=logit_csv_path, vol_csv=vol_csv_path, resid_csv=resid_csv_path,
            output_folder=MC_OUTPUT_FOLDER,
            n_simulations=10000, steps_per_year=252, confidence_level=0.999,
            vol_multiplier=1.0
        )
        print("\n--- Baseline 1-Year 99% VaR ---")
        print(base_var.to_string(index=False))
    except Exception as e:
        print(f"Error during Baseline MC: {e}")

stress_copula_path = os.path.join(VINE_DATA_FOLDER, "rvine_stress_20200101_20210101_model.json")

if os.path.exists(stress_copula_path) and os.path.exists(marginals_path):
    print("\nExecuting STRESSED (Crisis Market) Simulation...")
    try:
        df_vol = pd.read_csv(vol_csv_path, index_col=0, parse_dates=True)
        stress_start, stress_end = "2020-01-01", "2021-01-01"

        vol_stress_period = df_vol.loc[stress_start:stress_end].max()
        vol_long_term = df_vol.mean()
        #loc[stress_start:stress_end].mean()

        multiplier_series = vol_stress_period / vol_long_term
        calculated_multiplier = multiplier_series.mean()

        print(f"--> Dynamically calibrated Volatility Multiplier: {calculated_multiplier:.3f}x")

        stress_sims, stress_var = tools.simulate_1y_terminal_pd(
            copula_json_path=stress_copula_path,
            marginals_pkl_path=marginals_path,
            garch_params_csv=garch_params_path,
            logit_csv=logit_csv_path, vol_csv=vol_csv_path, resid_csv=resid_csv_path,
            output_folder=MC_OUTPUT_FOLDER,
            n_simulations=10000, steps_per_year=252, confidence_level=0.999,
            vol_multiplier=calculated_multiplier
        )
        print("\n--- Stressed 1-Year 99.9% VaR ---")
        print(stress_var.to_string(index=False))
    except Exception as e:
        print(f"Error during Stressed MC: {e}")
    
pipeline_end = timeit.default_timer()
print("\n Pipeline Complete.")
print("\n--- Execution Times ---")
print(f"Total Pipeline Time: {pipeline_end - pipeline_start:.4f} seconds")