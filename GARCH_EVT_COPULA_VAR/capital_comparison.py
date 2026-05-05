import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm 
import os 
import glob


def calculate_basel_capital(expected_pd, LGD=0.40, EAD=100_000_000, M=0.5):
    expected_pd = np.clip(expected_pd, 1e-6, 0.999)
    r_multiplier = (1 - np.exp(-50*expected_pd)) / (1 - np.exp(-50))
    R = 0.12 * r_multiplier + 0.24 * (1-r_multiplier)
    b = (0.11852 - 0.05478 * np.log(expected_pd)) ** 2
    maturity_adj = (1 + (M - 2.5) * b) / (1 - 1.5 * b) 
    RW = norm.cdf((norm.ppf(expected_pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)) 
    K_bracket = (RW - expected_pd) * LGD

    return K_bracket * maturity_adj * 1.06 *EAD

def generate_capital_waterfall():
    results_dir = "MC_SIMULATION_RESULTS"

    all_files = glob.glob(os.path.join(results_dir, "*_terminal_PDs.csv"))
    base_file = None
    stress_file = None

    for f in all_files:
        if "start_end" in f:
            base_file = f
        elif "rvine_stress" in f:
            stress_file = f

    if not base_file or not stress_file:
        print("Waiting for simulations to finish... Could not findboth baseline and stressed files ")
        return
    
    print(f"Loading Baseline File: {os.path.basename(base_file)}")
    print(f"Loading Stressed File: {os.path.basename(stress_file)}")

    df_base = pd.read_csv(base_file)
    df_stress = pd.read_csv(stress_file)
    countries = df_base.columns

    EAD = 100_000_000
    LGD = 0.40
    CONFIDENCE = 99.9

    base_expected_pds = df_base.mean()
    stress_expected_pds = df_stress.mean()

    basel_base_cap = sum(calculate_basel_capital(pd, LGD, EAD) for pd in base_expected_pds) 
    basel_stress_cap = sum(calculate_basel_capital(pd, LGD, EAD) for pd in stress_expected_pds)

    base_expected_loss = sum(base_expected_pds * LGD * EAD)

    base_standalone_var = 0
    base_standalone_es = 0
    for col in countries:
        losses = df_base[col] * LGD * EAD
        var_c = np.percentile(losses, CONFIDENCE)
        es_c = losses[losses >= var_c].mean()
        base_standalone_var += var_c
        base_standalone_es += es_c
        
    base_undiversified = base_standalone_var - base_expected_loss
    base_undiv_es_ul = base_standalone_es - base_expected_loss

    base_portfolio_losses = df_base.sum(axis=1) * LGD * EAD
    base_port_var = np.percentile(base_portfolio_losses, CONFIDENCE)
    base_port_es = base_portfolio_losses[base_portfolio_losses >= base_port_var].mean()
    
    base_diversified = base_port_var - base_expected_loss
    base_div_es_ul = base_port_es - base_expected_loss

    stress_expected_loss = sum(stress_expected_pds * LGD * EAD)
    
    stress_standalone_var = 0
    stress_standalone_es = 0
    for col in countries:
        losses = df_stress[col] * LGD * EAD
        var_c = np.percentile(losses, CONFIDENCE)
        es_c = losses[losses >= var_c].mean()
        stress_standalone_var += var_c
        stress_standalone_es += es_c
        
    stress_undiversified = stress_standalone_var - stress_expected_loss
    stress_undiv_es_ul = stress_standalone_es - stress_expected_loss

    stress_portfolio_losses = df_stress.sum(axis=1) * LGD * EAD
    stress_port_var = np.percentile(stress_portfolio_losses, CONFIDENCE)
    stress_port_es = stress_portfolio_losses[stress_portfolio_losses >= stress_port_var].mean()
    
    stress_diversified = stress_port_var - stress_expected_loss
    stress_div_es_ul = stress_port_es - stress_expected_loss
    
    labels = [
        'Basel ASRF\n(Alapeset)', 
        'Basel ASRF\n(Válságidőszak)', 
        'Kopula Alapeset\n(Nem divezifikált)', 
        'Kopula Alapeset\n(Diverzifikált)',
        'Kopula Stressz\n(Nem diverzifikált)', 
        'Kopula Stressz\n(Diverzifikált)'
    ]
    
    values_var = [
        basel_base_cap, 
        basel_stress_cap, 
        base_undiversified, 
        base_diversified, 
        stress_undiversified, 
        stress_diversified
    ]
    
    values_es = [
        None, 
        None, 
        base_undiv_es_ul, 
        base_div_es_ul,
        stress_undiv_es_ul, 
        stress_div_es_ul
    ]

    values_var_m = [v / 1_000_000 for v in values_var]
    values_es_m = [v / 1_000_000 if v is not None else None for v in values_es]
    
    colors = ['#9e9e9e', '#757575','#64b5f6', '#1976d2', '#e57373', '#d32f2f']

    plt.figure(figsize=(13, 8))
    bars = plt.bar(labels, values_var_m, color=colors, edgecolor='black', alpha=0.8, label='Nem Várható Veszteség (VaR - EL)')

    for i, bar in enumerate(bars):
        yval_var = bar.get_height()
        es_val = values_es_m[i]
        
        if es_val is not None:
            plt.hlines(y=es_val, xmin=bar.get_x() + 0.1, xmax=bar.get_x() + bar.get_width() - 0.1, 
                       colors='black', linestyles='dashed', linewidth=2)
            
            plt.text(bar.get_x() + bar.get_width()/2, es_val + 1.5, 
                     f'UL (ES): €{es_val:.1f}M\nUL (VaR): €{yval_var:.1f}M',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
        else:
            plt.text(bar.get_x() + bar.get_width()/2, yval_var + 1.5, 
                     f'UL (ASRF): €{yval_var:.1f}M',
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.plot([], [], color='black', linestyle='dashed', linewidth=2, label='Nem Várható Veszteség* (ES - EL)')
    
    max_y = max(
        max(values_var_m), 
        max([v for v in values_es_m if v is not None])
    )
    plt.ylim(0, max_y * 1.15)
    
    plt.title(f'Szabályozói Tőkekövetelmény vs. Belső Modell Nem Várható Veszteség (Portfólió EAD: €{len(countries)*100}M)', fontsize=15, pad=20)
    plt.ylabel('Nem Várható Veszteség - UL (Millió €)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    output_path = os.path.join(results_dir, "capital_comparison_6bar.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nChart generated successfully: {output_path}")

if __name__ == "__main__":
    generate_capital_waterfall()