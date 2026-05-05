import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import genpareto, gaussian_kde, probplot
import pickle
from scipy.optimize import minimize
import statsmodels.api as sm
import scipy.stats as stats
import pyvinecopulib as pv
import networkx as nx
import seaborn as sns

def find_file_recursive(filename, search_root='.'):
    for root, dirs, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_csv(filepath_or_name, **kwargs):
    if os.path.exists(filepath_or_name):
        return pd.read_csv(filepath_or_name, **kwargs)
    print(f"File '{filepath_or_name}' not found in root. Searching subfolders...")
    found_path = find_file_recursive(filepath_or_name)

    if found_path:
        print(f"Found file at: {found_path}")
        return pd.read_csv(found_path, **kwargs)
    else:
        raise FileNotFoundError(f"Could not find '{filepath_or_name}' anywhere in '{os.getcwd()}' or its subfolders.")

def apply_logit(df):
    df_transformed = df.copy()
    if 'Date' not in df_transformed.columns:
        print("WARNING: 'Date' column is missing!")

    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col == 'Date':
            continue
        series = df_transformed[col]

        if series.min() <= 0 or series.max() >=1:
            print(f"Skipping '{col}': Values out of range")
            continue
    
        epsilon = 1e-6
        series_clipped = series.clip(epsilon, 1 - epsilon)
        df_transformed[col] = np.log(series_clipped / (1 - series_clipped))

    return df_transformed

def save_result(df, original_name, output_folder):
    """
    Saves the transformed dataframe to a CSV.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    base_filename = os.path.basename(original_name)
    name_only = os.path.splitext(base_filename)[0]
    
    new_filename = f"{name_only}_logit.csv"
    
    save_path = os.path.join(output_folder, new_filename)
    df.to_csv(save_path, index=False)
    return save_path

def visualize_logit_check(csv_path, output_folder="LOGIT_VISUALS"):
    df = load_csv(csv_path)
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        except Exception as e:
            print(f"Error parsing 'Date': {e}")
            return
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created visualization folder: {output_folder}")

        numeric_cols=df.select_dtypes(include=[np.number]).columns
        count = 0

        print(f"Generating plots for {len(numeric_cols)} columns")

        # 1. INSERT THIS BLOCK BEFORE THE 'for col in numeric_cols:' LOOP:

        global_logit_min = df[numeric_cols].min().min()
        global_logit_max = df[numeric_cols].max().max()
        logit_padding = (global_logit_max - global_logit_min) * 0.05
        logit_ylim = (global_logit_min - logit_padding, global_logit_max + logit_padding)

        df_pd = 1 / (1 + np.exp(-df[numeric_cols]))
        global_pd_min = df_pd.min().min()
        global_pd_max = df_pd.max().max()
        pd_padding = (global_pd_max - global_pd_min) * 0.05
        pd_ylim = (max(0, global_pd_min - pd_padding), min(1, global_pd_max + pd_padding))

        for col in numeric_cols:
            series =df[col].dropna()
            reconstructed_series = 1 / (1 + np.exp(-series))

            fig, (ax1,ax2) =plt.subplots(2,1, figsize=(12,10), sharex=True)

            # 2. INSERT THESE TWO LINES INSIDE THE LOOP (to replace adaptive limits with the global ones):

            ax1.set_ylim(logit_ylim)  # Add under ax1.set_ylabel(...)

            ax2.set_ylim(pd_ylim)     # Add under ax2.set_ylabel(...)

            ax1.plot(series.index, series, color='salmon', linewidth=1, label='Logit')
            ax1.set_title(f"Transzformált: {col} (Log-Odds)", fontsize=12, fontweight='bold')
            ax1.set_ylabel('Logit érték')
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.plot(reconstructed_series.index, reconstructed_series, color='black', linewidth=1, label='PD')
            ax2.set_title(f'Eredeti idősor: {col} (Nemteljesítési valószínűség)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Valószínűség')
            ax2.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()

            safe_col_name = "".join([c if c.isalnum() else "_" for c in col])
            save_path = os.path.join(output_folder, f"{safe_col_name}.png")
            plt.savefig(save_path)
            plt.close(fig)
            count += 1
             
        print(f"DONE! Saved {count} images to {output_folder}/'")

def perform_garch(df, data_folder, vis_folder, burn_in=0):

    for folder in [data_folder, vis_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created: {folder}")

    work_df = df.copy()
    if 'Date' in work_df.columns:
        work_df['Date'] = pd.to_datetime(work_df['Date'])
        work_df.set_index('Date', inplace=True)

    numeric_cols = work_df.select_dtypes(include=[np.number]).columns

    vol_dict = {}
    resid_dict = {}
    params_list = []
    diag_list = []

    passed_count = 0

    print(f"Starting GARCH Pipeline on {len(numeric_cols)} series...")
    for col in numeric_cols:
        series = work_df[col].dropna()
        try:
            model = arch_model(series, mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='ged', rescale=True)
            res = model.fit(disp='off', show_warning=False)

            model_name = "AR(1)-GARCH(1,1)"

            ar_key = next((k for k in res.params.index if 'y' in k or 'L1' in k), None)
            if ar_key and res.pvalues[ar_key] > 0.05:
                model = arch_model(series, mean='constant', vol='GARCH', p=1, q=1, dist='ged', rescale=True)
                res =model.fit(disp='off', show_warning=False)
                model_name = "Constant-GARCH(1,1)"
            
            stable_resid = res.std_resid.iloc[burn_in:]
            stable_vol = res.conditional_volatility.iloc[burn_in:]

            alpha = res.params.get('alpha[1]', 0)
            beta = res.params.get('beta[1]', 0)
            persistence = alpha + beta

            clean_residuals = stable_resid.dropna()

            lb = acorr_ljungbox(clean_residuals**2, lags=[10], return_df=True)
            lb_p = lb['lb_pvalue'].iloc[0]

            is_independent = lb_p > 0.05

            print(f"{col[:33]:<35} {persistence:>8.4f}")

            if not is_independent:
                print(f"SKIP {col}: Failed Ljung-box (p={lb_p:.4f})")
                continue

            passed_count += 1
            print(f"  [PASS] {col}: p={lb_p:.4f}")

            vol_dict[col] = stable_vol
            resid_dict[col] = stable_resid

            p_dict = res.params.to_dict()
            p_dict.update({
                'Column': col, 
                'Model': model_name, 
                'Scale': res.scale,
                'Burn_in': burn_in})
                
            params_list.append(p_dict)

            diag_list.append({
                'Column': col, 'LB_pvalue': lb_p,
                'Independent': True, 'Model': model_name
            })

            

        except Exception as e:
            print(f"Error on {col}: {e}")

    print(f"\n---  GARCH Complete: {passed_count}/{len(numeric_cols)} passed IID check ---")

    if passed_count > 0:

        vol_df = pd.DataFrame(vol_dict)
        resid_df = pd.DataFrame(resid_dict)

        vol_df.dropna(how='all', inplace=True)
        resid_df.dropna(how='all', inplace=True)

        vol_min, vol_max = vol_df.min().min(), vol_df.max().max()
        vol_pad = (vol_max - vol_min) * 0.05
        vol_ylim = (max(0, vol_min - vol_pad), vol_max + vol_pad)

        resid_min, resid_max = resid_df.min().min(), resid_df.max().max()
        resid_pad = (resid_max - resid_min) * 0.05
        resid_ylim = (resid_min - resid_pad, resid_max + resid_pad)

        print("Generating globally scaled plots...")
        for col in vol_df.columns:
            stable_vol = vol_df[col].dropna()
            stable_resid = resid_df[col].dropna()
            lb_p = next(d['LB_pvalue'] for d in diag_list if d['Column'] == col)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            ax1.plot(stable_vol, color='teal', label='Volatilitás', alpha=0.7)
            ax1.set_title(f"{col} (Ljung-Box teszt p-értéke: {lb_p:.3f})", fontsize=10)
            ax1.set_ylim(vol_ylim)  # Applied global y-axis
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            ax2.scatter(stable_resid.index, stable_resid, color='slategray', alpha=0.6, label='Std Residuals', marker='.')
            ax2.set_title("Sztenderd Reziduálisok", fontsize=10)
            ax2.axhline(0, color='black', alpha=0.4)
            ax2.set_ylim(resid_ylim) # Applied global y-axis
            ax2.grid(True, alpha=0.3)

            safe_name = "".join([c if c.isalnum() else "_" for c in col])
            plt.tight_layout()
            plt.savefig(os.path.join(vis_folder, f"{safe_name}_garch.png"))
            plt.close(fig)
        
        vol_df.to_csv(os.path.join(data_folder, "vol_results.csv"))
        resid_df.to_csv(os.path.join(data_folder, "resid_results.csv"))
        pd.DataFrame(params_list).to_csv(os.path.join(data_folder, "garch_param.csv"), index=False)
        pd.DataFrame(diag_list).to_csv(os.path.join(data_folder, "garch_diagnostics.csv"), index=False)
        print(f"Saved results to {data_folder}")
    else:
        print("WARNING: No columns passed the independence test. No files saved.")

def plot_residual_pdf(csv_path, output_folder="RESID_PDF_PLOTS"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    try:
        df = load_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Generating PDF plots for {len(numeric_cols)} series...")

    for col in numeric_cols:
        data = df[col].dropna()
        
        # Calculate outliers beyond +/- 6
        outliers_mask = (data < -6) | (data > 6)
        outliers_count = outliers_mask.sum()
        outliers_pct = (outliers_count / len(data)) * 100
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot empirical histogram
        sns.histplot(data, bins='auto', stat='density', color='slategray', alpha=0.5, 
                     label='Empirikus Hisztogram', ax=ax)
        
        # Plot non-parametric KDE
        sns.kdeplot(data, color='darkblue', linewidth=2, 
                    label='Empirikus Sűrűségfüggvény (KDE)', ax=ax)
        
        # Overlay Standard Normal for tail comparison
        x_grid = np.linspace(-6, 6, 200)
        ax.plot(x_grid, stats.norm.pdf(x_grid, loc=0, scale=1), 
                color='firebrick', linestyle='--', linewidth=2, 
                label='Sztenderd Normális Eloszlás')
        
        ax.set_title(f"Sztenderdizált Reziduálisok Sűrűségfüggvénye: {col}", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("Sztenderdizált Reziduális")
        ax.set_ylabel("Sűrűség")
        
        # Restrict the x-axis to visually focus on the relevant tail mass
        ax.set_xlim(-6, 6)
        
        # Add notification about truncated data if any exists
        if outliers_count > 0:
            info_text = f"Ábra limitálva $\pm6$-ra\nKihagyott extrém értékek: {outliers_count} db ({outliers_pct:.2f}%)"
            ax.text(0.02, 0.5, info_text, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.6, edgecolor='gray'))

        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        safe_name = "".join([c if c.isalnum() else "_" for c in col])
        plt.savefig(os.path.join(output_folder, f"{safe_name}_resid_pdf.png"), dpi=300)
        plt.close(fig)

    print(f"Complete. PDF plots saved to '{output_folder}/'")

def mean_residual_life(csv_path, output_folder="MRL_PLOTS"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    try: 
        df = load_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Generating MRL plots for {len(numeric_cols)} series...")

    def _calc_mrl(series_data):
        percentiles = np.linspace(0.80, 0.995, 60)
        u_vals, mrl_vals = [], []

        for p in percentiles: 
            u=series_data.quantile(p)
            exceedences = series_data[series_data > u] - u
            n_u = len(exceedences)

            if n_u > 1:
                u_vals.append(u)
                mrl_vals.append(exceedences.mean())

        u_95 = series_data.quantile(0.95)

        return np.array(u_vals), np.array(mrl_vals), u_95
    
    for col in numeric_cols:

        z = df[col].dropna()
        rt_u, rt_mrl, rt_u95 = _calc_mrl(z)

        z_neg = -z
        lt_u, lt_mrl, lt_u95 = _calc_mrl(z_neg)

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))

        ax1.plot(lt_u, lt_mrl, color='firebrick', linewidth=2, marker='o', markersize=3)
        ax1.axvline(lt_u95, color='red', linestyle='--', label=f"95% percentilis (u={lt_u95:.3f})")
        ax1.set_title(f"Baloldali Farokeloszlás MRL: {col}", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Küszöbérték (u) [Negatív sokkok abszolút értéke]")
        ax1.set_ylabel("Átlagos többlet $E(u)$")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(rt_u, rt_mrl, color='forestgreen', linewidth=2, marker='o', markersize=3)
        ax2.axvline(rt_u95, color='red', linestyle='--', label=f"95% percentilis (u={lt_u95:.3f})")
        ax2.set_title(f"Jobboldali Farokeloszlás MRL: {col}", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Küszöbérték (u) [Pozitív sokkok]")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = "".join([c if c.isalnum() else "_" for c in col])
        plt.savefig(os.path.join(output_folder, f"{safe_name}_clean_MRL.png"))
        plt.close(fig)

    print(f"Complete. Clean MRL plots saved to '{output_folder}_clean_MRL.png")

def robust_gpd_fit(exceedences, max_shape=0.4):
    if len(exceedences) < 2:
        return 0.1, 0, max(0.001 ,np.mean(exceedences) if len(exceedences) == 1 else 1.0)
    
    exceedences = np.maximum(exceedences, 1e-8)

    if np.var(exceedences) < 1e-8:
        return 0.1, 0, exceedences.mean()
    
    def neg_log_lik(params):
        c, scale = params
        if scale <= 0 or c < 0.001 or c > max_shape:
            return np.inf
        ll = genpareto.logpdf(exceedences, c=c, loc=0, scale=scale)
        if np.any(np.isinf(ll)) or np.any(np.isnan(ll)):
            return np.inf
        return -np.sum(ll)
      
    mean_ex = exceedences.mean()
    variance = np.var(exceedences)

    initial_guess = [0.1, mean_ex]
    res = minimize(neg_log_lik, initial_guess, method='Nelder-Mead')

    if res.success and res.x[1] > 0 and 0 < res.x[0] <=max_shape:
        return res.x[0], 0, res.x[1]
    
    if variance > (mean_ex**2):
        shape_mom = 0.5 * (1-((mean_ex**2)/variance))
        scale_mom = 0.5 * mean_ex * (((mean_ex**2)/variance)+1)
        return min(max(shape_mom, 0.001), max_shape), 0, max(scale_mom, 0.001)
    else:
        return 0.1, 0, mean_ex

def fit_semiparametric_distribution(df_residuals, data_folder="EVT_DATA", vis_folder="EVT_VISUALS", threshold_dict=None):

    for folder in [data_folder, vis_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    df = df_residuals.copy()

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    default_t = {'left': 0.05, 'right': 0.95}
    model_results = {}
    summary_list = []
    u_data_dict = {}

    print(f"Starting EVT Fitting for {len(df_residuals.columns)} columns...")

    for col in numeric_cols:
        series = df[col].dropna()
        data = series.values
        n_total = len(data)

        t = threshold_dict.get(col, default_t) if threshold_dict else default_t
        u_l = np.percentile(data, t['left']*100)
        u_r = np.percentile(data, t['right']*100)

        mask_L = data < u_l
        mask_R = data > u_r
        mask_B = (data >= u_l) & (data <= u_r)

        phi_L = np.sum(mask_L) / n_total
        phi_R = np.sum(mask_R) / n_total
        phi_B = np.sum(mask_B) / n_total

        shape_r, _, scale_r = robust_gpd_fit(data[mask_R]-u_r)
        shape_l, _, scale_l = robust_gpd_fit(u_l- data[mask_L])
        kde = gaussian_kde(data[mask_B], bw_method='scott') if np.sum(mask_B) > 0 else None

        u_values = np.empty(n_total)
        u_values[:] = np.nan

        if np.sum(mask_L) > 0:
            u_values[mask_L] = phi_L * genpareto.sf(u_l - data[mask_L], shape_l, scale=scale_l)

        if np.sum(mask_R) > 0:
            u_values[mask_R] = (1-phi_R) + phi_R * genpareto.cdf(data[mask_R] - u_r, shape_r, scale=scale_r)

        if kde is not None:
            grid_pts = np.linspace(u_l, u_r, 1000)
            pdf_vals = kde(grid_pts)
            cdf_grid = np.cumsum(pdf_vals)
            cdf_grid = cdf_grid /cdf_grid[-1]
            body_u_local = np.interp(data[mask_B], grid_pts, cdf_grid)
            u_values[mask_B] = phi_L + body_u_local * phi_B

        u_data_dict[col] = pd.Series(u_values, index=series.index)

        z_theo = stats.norm.ppf(u_values)

        z_theo_clipped = np.nan_to_num(z_theo, posinf=5, neginf=-5)

        fig, ax = plt.subplots(figsize=(6,5))
        sm.qqplot(z_theo_clipped, line='45', fit=True, ax=ax, markerfacecolor='darkblue', markeredgecolor='darkblue', alpha=0.5)
        ax.set_title(f"Analitikus QQ-Plot: {col}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = "".join([c if c.isalnum() else "_" for c in col])
        plt.savefig(os.path.join(vis_folder, f"{safe_name}_EVT_QQ.png"))
        plt.close(fig)

        params = {
            'u_l': u_l, 'shape_l': shape_l, 'scale_l': scale_l, 'p_l': t['left'],
            'u_r': u_r, 'shape_r': shape_r, 'scale_r': scale_r, 'p_r': 1-t['right'],
            'n_total': n_total
        }

        model_results[col] = {'params': params, 'kde': kde}
        summary_list.append({'Orszag': col, **params})

    pd.DataFrame(summary_list).to_csv(os.path.join(data_folder, "evt_params.csv"), index=False)
    pd.DataFrame(u_data_dict).to_csv(os.path.join(data_folder, "copula_u_values.csv"))
    with open(os.path.join(data_folder, "semiparametric_models.pkl"), 'wb') as f:
        pickle.dump(model_results, f)

    print(f"EVT Pipeline Complete. U-Values saved to {data_folder}/copula_u_values.csv")
    return model_results

def fit_rvine_filtered(u_values_csv, start_date=None, end_date=None, data_folder="VINE_DATA", vis_folder="VINE_VISUALS", plot_tree=True):
    for folder in [data_folder, vis_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")

    try:
        df_u = load_csv(u_values_csv, index_col=0, parse_dates = True)
    except Exception as e:
        print(f"Error loading {u_values_csv}: {e}")
        return None
    
    if start_date or end_date:
        start = start_date if start_date else df_u.index.min()
        end = end_date if end_date else df_u.index.max()
        df_filtered = df_u.loc[start:end].copy()
        print(f"Filtering data from {start} to {end}. Observations: {len(df_filtered)}")
    else:
        df_filtered = df_u.copy()
        print(f"Using full dataset. Observations: {len(df_filtered)}")

    if len(df_filtered) < 30:
        print("WARNING: Very small sample for copula fitting.")

    u_matrix = df_filtered.values.astype(np.float64)
    variable_names = list(df_filtered.columns)

    print("Fitting R_Vine Copula")

    controls = pv.FitControlsVinecop(
        family_set=pv.parametric,
        selection_criterion="aic",
        select_threshold=True,
        threshold=0.05,
        #allow_rotations=False
        show_trace=False
    )

    copula = pv.Vinecop(d=u_matrix.shape[1])

    copula.select(data=u_matrix, controls=controls)

    d_dim = u_matrix.shape[1]
    matrix_array = np.asarray(copula.matrix)

    summary_list = []
    for t in range(d_dim-1):
        for e in range(d_dim-1-t):
            pc = copula.get_pair_copula(t,e)

            var1_idx = matrix_array[d_dim-1-e, e]
            var2_idx = matrix_array[d_dim-2-t-e, e]

            summary_list.append({
                'tree': t+1,
                'var1': str(int(var1_idx)),
                'var2': str(int(var2_idx)),
                'family': pc.family.name,
                'tau': np.round(pc.tau, 4)
            })

    summary_df = pd.DataFrame(summary_list)

    name_map = {str(i+1): name for i, name in enumerate(variable_names)}
    summary_df['var1'] = summary_df['var1'].astype(str).map(name_map).fillna(summary_df['var1'])
    summary_df['var2'] = summary_df['var2'].astype(str).map(name_map).fillna(summary_df['var2'])

    safe_start = str(start_date).replace("-", "") if start_date else "start"
    safe_end = str(end_date).replace("-", "") if end_date else "end"
    filename_base = f"rvine_stress_{safe_start}_{safe_end}"

    summary_df.to_csv(os.path.join(data_folder, f"{filename_base}_summary.csv"), index=False)

    json_model = copula.to_json()
    with open(os.path.join(data_folder, f"{filename_base}_model.json"),"w") as f:
        f.write(json_model)

    print(f"R-Vine fitting complete. Data saved to {data_folder}/")

    if plot_tree:
        try:
            tree1_df = summary_df[summary_df['tree'] == 1].copy()
            G = nx.Graph()

            for _, row in tree1_df.iterrows():
                u, v = row['var1'], row['var2']
                family = row['family']
                tau = row['tau']

                label = f"{family}\ntau={tau:.2f}"
                weight = abs(tau) * 5

                G.add_edge(u, v, weight=weight, label=label, tau=tau)

            fig, ax =plt.subplots(figsize=(12,10))
            pos = nx.spring_layout(G, k=0.8, seed=42)

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightsteelblue', node_size=3000, edgecolors='darkblue', linewidths=1.5)

            weights = [G[u][v]['weight'] for u, v in G.edges()]
            edge_colors = ['darkred' if G[u][v]['tau'] < 0 else 'darkgreen' for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, ax=ax, width=weights, edge_color=edge_colors, alpha=0.6)

            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')

            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            plt.title(f"R-Vine Kopula Struktúra \n{start_date or 'Alapeset'} - {end_date or 'Alapeset'}", fontsize=14, fontweight='bold')
            plt.tight_layout()

            plot_path = os.path.join(vis_folder, f"{filename_base}_network.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Network graph saved to {plot_path}")

        except Exception as e:
            print(f"Failed to generate plot: {e}")

    return copula, summary_df

def plot_empirical_copula_pairs(u_values_csv, vis_folder="VINE_VISUALS", start_date=None, end_date=None):
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
        print(f"Created directory: {vis_folder}")

    try:
        df_u = load_csv(u_values_csv, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading {u_values_csv}: {e}")
        return
    
    if start_date or end_date:
        start = start_date if start_date else df_u.index.min()
        end = end_date if end_date else df_u.index.max()
        df_filtered = df_u.loc[start:end].copy()
        time_label = f"{str(start).replace('-', '')[:8]}_{str(end).replace('-', '')[:8]}"
    else:
        df_filtered = df_u.copy()
        time_label = "teljes minta"

    print(f"Generating pairwise scatter matrix for {len(df_filtered.columns)} variables. This may take a minute...")

    sns.set_theme(style='whitegrid')

    g=sns.PairGrid(df_filtered, corner=True, diag_sharey=False)

    g.map_lower(sns.scatterplot, s=15, alpha=0.5, color='darkblue', edgecolor='k', linewidth=0.2)

    g.map_diag(sns.histplot, bins=20, color='gray', alpha=0.6, stat='density')

    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)

            ax.set_xticks([])
            ax.set_yticks([])

    fig = g.fig
    fig.suptitle(f"Kopula Párok\nIdőintervallum: {start_date or 'Alapeset'} - {end_date or 'Alapeset'}",
                 fontsize=16, fontweight='bold', y=1.02)
    
    plot_path = os.path.join(vis_folder, f"copula_pairs_{time_label}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Pairwise plot saved successfully to {plot_path}")

def plot_copula_contours(u_values_csv, summary_df, vis_folder="VINE__VISUALS", start_date=None, end_date=None):
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    try:
        df_u = load_csv(u_values_csv, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading {u_values_csv}: {e}")
        return
    
    if start_date or end_date:
        start = start_date if start_date else df_u.index.min()
        end = end_date if end_date else df_u.index.max()
        df_filtered = df_u.loc[start:end].copy()
        time_label = f"{str(start).replace('-','')[:8]}_{str(end).replace('_','')[:8]}"
    else:
        df_filtered = df_u.copy()
        time_label = "baseline"

    tree1_df = summary_df[summary_df['tree'] == 1].copy()

    if tree1_df.empty:
        print("No Tree 1 connections found to plot.")
        return

    print(f"Generating {len(tree1_df)} contour plots...")

    for _, row in tree1_df.iterrows():
        var1, var2 = row['var1'], row['var2']
        tau, family = row['tau'], row['family']

        if var1 not in df_filtered.columns or var2 not in df_filtered.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.kdeplot(
            x=df_filtered[var1], y=df_filtered[var2],
            fill=True, cmap="Spectral_r", levels=20, thresh=0.02,
            ax=ax, clip=((0, 1), (0, 1))
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{var1} - {var2}\nTau: {tau:.3f} | Kopula család: {family}", 
                     fontsize=12, fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        safe_v1 = "".join([c if c.isalnum() else "_" for c in var1])
        safe_v2 = "".join([c if c.isalnum() else "_" for c in var2])
        plot_path = os.path.join(vis_folder, f"contour_{safe_v1}_{safe_v2}_{time_label}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

def simulate_1y_terminal_pd(
        copula_json_path, marginals_pkl_path, garch_params_csv,
        logit_csv, vol_csv, resid_csv,
        output_folder="MC_SIMULATION", n_simulations=10000, steps_per_year=252, confidence_level=0.999,
        vol_multiplier=1.0
        ):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Initializing MC engine: {n_simulations} simulations for {steps_per_year} steps per year...")
    
    try:
        copula = pv.Vinecop.from_file(copula_json_path)
        with open(marginals_pkl_path, 'rb') as f:
            marginals = pickle.load(f)
            
        garch_params = pd.read_csv(garch_params_csv).set_index('Column')
        df_logit = load_csv(logit_csv, index_col=0, parse_dates=True)
        df_vol = load_csv(vol_csv, index_col=0, parse_dates=True)
        df_resid = load_csv(resid_csv, index_col=0, parse_dates=True)

    except Exception as e:
        print(f"Error loading simulation inputs: {e}")
        return None
    
    countries = list(marginals.keys())
    d_dim = len(countries)

    total_draws = n_simulations *steps_per_year
    print(f"Pre-drawing {total_draws:,} vectors from the R-Vine Copula...")
    U_all = copula.simulate(n=total_draws)
    Z_all = np.zeros_like(U_all)

    def apply_inverse_evt(u_array, params, kde):
        z_array = np.zeros_like(u_array)

        mask_L = u_array < params['p_l']
        if np.any(mask_L):
            sf_val = u_array[mask_L] / params['p_l']
            z_array[mask_L] = params['u_l'] - genpareto.isf(sf_val, c=params['shape_l'], scale=params['scale_l']) 

        mask_R = u_array > (1-params['p_r'])
        if np.any(mask_R):
            cdf_val = (u_array[mask_R] - (1-params['p_r'])) / params['p_r']
            z_array[mask_R] = params['u_r'] + genpareto.ppf(cdf_val, c=params['shape_r'], scale=params['scale_r'])

        mask_B = ~(mask_L | mask_R)
        if np.any(mask_B):
            grid_pts = np.linspace(params['u_l'], params['u_r'], 2000)
            cdf_grid = np.cumsum(kde(grid_pts))
            cdf_grid = cdf_grid / cdf_grid[-1]
            local_u = (u_array[mask_B] - params['p_l']) / (1 - params['p_l'] - params['p_r'])
            z_array[mask_B] = np.interp(local_u, cdf_grid, grid_pts)

        return z_array

    print("Pushing data through Inverse EVT/KDE functions...")

    for i, col in enumerate(countries):
        Z_all[:, i] = apply_inverse_evt(U_all[:, i], marginals[col]['params'], marginals[col]['kde'])

    Z_all = np.clip(Z_all, -6.0, 6.0)

    Z_matrix = Z_all.reshape((steps_per_year, n_simulations, d_dim))

    current_Y_scaled = np.zeros((n_simulations, d_dim))
    current_vol = np.zeros((n_simulations, d_dim))
    current_eps = np.zeros((n_simulations, d_dim))

    omega_arr = np.zeros(d_dim)
    alpha_arr = np.zeros(d_dim)
    beta_arr = np.zeros(d_dim)
    const_arr = np.zeros(d_dim)
    phi_arr = np.zeros(d_dim)
    is_ar = np.zeros(d_dim, dtype=bool)

    for i, col in enumerate(countries):
        scale = garch_params.loc[col, 'Scale']
        current_Y_scaled[:, i] = df_logit[col].iloc[-1] * scale

        current_vol[:, i] = df_vol[col].iloc[-1]
        current_eps[:, i] = df_vol[col].iloc[-1] * df_resid[col].iloc[-1]

        omega_arr[i] = garch_params.loc[col, 'omega'] * (vol_multiplier ** 2)
        alpha_arr[i] = garch_params.loc[col, 'alpha[1]']
        beta_arr[i] = garch_params.loc[col, 'beta[1]']

        c_term = garch_params.loc[col, 'Const'] if 'Const' in garch_params.columns and not pd.isna(garch_params.loc[col, 'Const']) else 0
        if pd.isna(c_term) and 'mu' in garch_params.columns:
            c_term = garch_params.loc[col, 'mu']
        const_arr[i] = c_term

        if "AR" in str(garch_params.loc[col, 'Model']):
            phi_arr[i] = garch_params.loc[col, 'y[1]'] if 'y[1]' in garch_params.columns else 0
            is_ar[i] = True

    print("Executing sequantial GARCH arithmetic...")

    for step in range(steps_per_year):
        Z_t = Z_matrix[step, :, :]

        var_t = omega_arr + alpha_arr * (current_eps**2) + beta_arr * (current_vol**2)
        sigma_t = np.sqrt(var_t)
        eps_t = sigma_t * Z_t

        Y_t = const_arr + eps_t
        ar_mask = is_ar
        if np.any(ar_mask):
            Y_t[:, ar_mask] += phi_arr[ar_mask] * current_Y_scaled[:, ar_mask]

        current_Y_scaled = Y_t
        current_vol = sigma_t
        current_eps = eps_t

    print("Reconstructing final Probabilities of Default...")
    final_pd_dict = {}
    var_results = []

    for i, col in enumerate(countries):
        scale = garch_params.loc[col, 'Scale']
        final_logit = current_Y_scaled[:, i] / scale
        final_pds = 1 / (1 + np.exp(-final_logit))
        final_pd_dict[col] = final_pds

        var_threshold = np.percentile(final_pds, confidence_level * 100)
        var_results.append({'Country': col, f'VaR_{confidence_level* 100}': var_threshold})

    df_final_sims = pd.DataFrame(final_pd_dict)
    df_var = pd.DataFrame(var_results)

    file_prefix = os.path.basename(copula_json_path).replace("_model.json", "")
    df_final_sims.to_csv(os.path.join(output_folder, f"{file_prefix}_terminal_PDs.csv"), index=False)

    return df_final_sims, df_var