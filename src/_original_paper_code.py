import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import os
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="paper")

@dataclass
class ResearchConfig:
    base_path: str = "/Users/isaiahchoi/Desktop/Junior Winter Term/Mathematical Finance/Final Project/final model"
    predictor_filename: str = "USDJPY.csv"
    equity_filename: str = "JPEquities.xlsx"
    
    equities: dict[str, str] = field(default_factory=lambda: {
        "5713": "Sumitomo Metal Mining",
        "5711": "Mitsubishi Materials", 
        "5706": "Mitsui Mining & Smelting"
    })
    
    @property
    def predictor_path(self): 
        return os.path.join(self.base_path, self.predictor_filename)
    
    @property
    def equity_path(self): 
        return os.path.join(self.base_path, self.equity_filename)
    
    test_size: float = 0.2
    n_estimators: int = 500  
    random_state: int = 42
    max_lag_analysis: int = 5

class DataEngine:
    def __init__(self, cfg: ResearchConfig):
        self.cfg = cfg
        self.fx_daily = None  
        self.equity_cache = {} 
        self._excel_sheets = None 

    def _load_excel_sheets(self):
        if self._excel_sheets is None:
            filepath = self.cfg.equity_path
            logger.info(f"Loading Excel file: {filepath}")
            self._excel_sheets = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
        return self._excel_sheets
    
    def _find_sheet_for_ticker(self, ticker: str) -> pd.DataFrame:
        sheets = self._load_excel_sheets()
        for sheet_name, df in sheets.items():
            if ticker in str(sheet_name):
                return df.copy()
        for sheet_name, df in sheets.items():
            sample_grid = df.iloc[:15, :5].astype(str).values.flatten()
            if any(ticker in cell for cell in sample_grid):
                return df.copy()
        return None

    def _parse_equity_sheet(self, raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = raw_df.copy()
        header_idx = None
        for i in range(min(20, len(df))):
            row_vals = [str(v).strip() for v in df.iloc[i].values]
            if any('Dates' in v or 'PX_LAST' in v for v in row_vals):
                header_idx = i
                break

        if header_idx is None: return None

        df.columns = [str(c).strip() for c in df.iloc[header_idx].values]
        df = df.iloc[header_idx + 1:].reset_index(drop=True)
        df = df[[c for c in df.columns if str(c).lower() not in ('nan', 'none', 'nat', '')]]
        
        if 'Dates' not in df.columns: return None

        if not pd.api.types.is_datetime64_any_dtype(df['Dates']):
            df_numeric = pd.to_numeric(df['Dates'], errors='coerce')
            if df_numeric.notna().mean() > 0.5:
                df['Dates'] = pd.to_datetime(df_numeric, unit='D', origin='1899-12-30')
            else:
                df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
                
        df.dropna(subset=['Dates'], inplace=True)

        required = ['PX_LAST', 'PX_OPEN']
        for col in required:
            if col not in df.columns: return None
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=required, inplace=True)
        return df

    def _inspect_and_load(self, filepath: str, is_5min: bool) -> pd.DataFrame:
        try:
            with open(filepath, 'r') as f:
                head_lines = [next(f) for _ in range(20)]
            header_idx = next((i for i, line in enumerate(head_lines) if "Dates" in line), -1)
            if header_idx == -1: return None
            
            df = pd.read_csv(filepath, skiprows=header_idx)
            df.columns = [str(c).strip() for c in df.columns]

            if pd.api.types.is_numeric_dtype(df['Dates']):
                df['Dates'] = pd.to_datetime(df['Dates'], unit='D', origin='1899-12-30')
            else:
                df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
            df.dropna(subset=['Dates'], inplace=True)
            
            if is_5min:
                cols = ['Open', 'Close', 'High', 'Low']
                if 'High' not in df.columns and len(df.columns) >= 5:
                     df = df.iloc[:, :5]
                     df.columns = ['Dates', 'Open', 'Close', 'High', 'Low']
                for c in cols:
                    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
                df.dropna(subset=['Close'], inplace=True)
                
                if 'High' in df.columns and 'Low' in df.columns:
                    if (df['High'] < df['Low']).mean() > 0.1:
                        df.rename(columns={'High': 'Low', 'Low': 'High'}, inplace=True)
            return df
        except Exception as e:
            return None
    
    def load_fx(self):
        df_5min = self._inspect_and_load(self.cfg.predictor_path, True)
        if df_5min is None: return None
        
        if df_5min['Dates'].dt.tz is None:
            df_5min['Dates'] = df_5min['Dates'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')
        else:
            df_5min['Dates'] = df_5min['Dates'].dt.tz_convert('Asia/Tokyo')
            
        df_5min.sort_values('Dates', inplace=True)
        df_5min['Base_Date'] = df_5min['Dates'].dt.normalize().dt.tz_localize(None)
        df_5min['Hour'] = df_5min['Dates'].dt.hour
        df_5min['Minute'] = df_5min['Dates'].dt.minute
        
        df_5min['Pred_Date'] = df_5min['Base_Date']
        df_5min.loc[df_5min['Hour'] >= 15, 'Pred_Date'] = df_5min['Base_Date'] + BDay(1)
        df_5min['Pred_Date'] = df_5min['Pred_Date'] + BDay(0)
        
        is_evening = df_5min['Hour'] >= 15
        is_morning = (df_5min['Hour'] < 8) | ((df_5min['Hour'] == 8) & (df_5min['Minute'] < 50))
        df_5min = df_5min[is_evening | is_morning].copy()
        
        df_5min['Log_Ret'] = df_5min.groupby('Pred_Date')['Close'].transform(lambda x: np.log(x / x.shift(1)))
        
        daily = df_5min.groupby('Pred_Date').agg(
            FX_Vol=('Log_Ret', 'std'),
            FX_Ret=('Log_Ret', 'sum'),
            Bars=('Log_Ret', 'count'),
            Last_FX_Bar=('Dates', 'max')  
        ).reset_index()
        
        daily.sort_values('Pred_Date', inplace=True)
        daily['FX_Ret_2D'] = daily['FX_Ret'] + daily['FX_Ret'].shift(1)
        daily['FX_Ret_5D'] = daily['FX_Ret'].rolling(5, min_periods=5).sum()
        
        daily['DoW'] = daily['Pred_Date'].dt.dayofweek
        daily = daily[daily['Bars'] > 50].dropna(subset=['FX_Ret_5D', 'FX_Vol'])
        
        self.fx_daily = daily
        return daily

    def get_data(self, ticker: str):
        if ticker in self.equity_cache:
            return self.equity_cache[ticker].copy()
            
        if self.fx_daily is None:
            self.load_fx()
            
        raw_df = self._find_sheet_for_ticker(ticker)
        if raw_df is None: return None
        
        df_target = self._parse_equity_sheet(raw_df, ticker)
        if df_target is None: return None
            
        df_target.sort_values('Dates', inplace=True)
        df_target['Prev_Close'] = df_target['PX_LAST'].shift(1)
        df_target['Gap'] = (df_target['PX_OPEN'] - df_target['Prev_Close']) / df_target['Prev_Close']
        df_target['Gap'] = np.where(df_target['Gap'].abs() < 1e-5, 0.0, df_target['Gap'])
        
        df_target['Gap_Lag1'] = df_target['Gap'].shift(1)
        df_target['Gap_Lag2'] = df_target['Gap'].shift(2)
        
        df_target.dropna(subset=['Gap', 'Gap_Lag2'], inplace=True)
        
        final = pd.merge(
            df_target[['Dates', 'Gap', 'Gap_Lag1', 'Gap_Lag2']],
            self.fx_daily[['Pred_Date', 'FX_Vol', 'FX_Ret', 'FX_Ret_2D', 'FX_Ret_5D', 'DoW', 'Last_FX_Bar']],
            left_on='Dates',
            right_on='Pred_Date',
            how='inner'
        )
        
        final['DoW'] = pd.Categorical(final['DoW'], categories=[0, 1, 2, 3, 4])
        final = pd.get_dummies(final, columns=['DoW'], drop_first=True, prefix='DoW', dtype=int)
        final.drop_duplicates(subset=['Dates'], inplace=True)
        
        equity_open_time = final['Dates'].dt.tz_localize('Asia/Tokyo') + pd.Timedelta(hours=9)
        violations = final[final['Last_FX_Bar'] >= equity_open_time]
        if not violations.empty:
            max_violation = violations['Last_FX_Bar'].max()
            raise ValueError(f"CRITICAL LOOKAHEAD: FX Bar recorded at {max_violation} used to predict equity open.")
        
        final.drop(columns=['Pred_Date', 'Last_FX_Bar'], inplace=True, errors='ignore')
        
        final['Equity'] = ticker
        final['Target_Class'] = (final['Gap'] > 0).astype(int)
        final.set_index('Dates', inplace=True)
        
        self.equity_cache[ticker] = final
        return final.copy()

def combine_datasets(eng: DataEngine, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    dfs = []
    for t in tickers:
        df = eng.get_data(t)
        if df is not None: dfs.append(df)
            
    if not dfs: return None, []
    combined = pd.concat(dfs)
    
    features = ['FX_Vol', 'FX_Ret', 'FX_Ret_2D', 'FX_Ret_5D', 'Gap_Lag1', 'Gap_Lag2']
    dow_cols = sorted([c for c in combined.columns if c.startswith('DoW_')])
    features.extend(dow_cols)
    
    if len(tickers) > 1:
        dummies = pd.get_dummies(combined['Equity'], prefix='Eq', drop_first=False)
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        combined = pd.concat([combined, dummies], axis=1)
        features.extend(dummies.columns.tolist())
    
    combined.sort_index(inplace=True)
    return combined, features

def split_chronologically(df: pd.DataFrame, test_size: float = 0.2):
    unique_dates = np.sort(df.index.unique())
    split_idx = int(len(unique_dates) * (1 - test_size))
    split_date = unique_dates[split_idx]
    
    train = df[df.index < split_date]
    embargo_cutoff = train.index.max() + BDay(2)
    test = df[df.index > embargo_cutoff]
    
    assert len(train) > 0 and len(test) > 0, "Split failed: Train or Test is empty."
    return train, test

class ReportingEngine:
    def __init__(self, model_reg, model_clf, X_test, y_test_reg, y_test_clf, fx_features_test, experiment_name):
        self.model_reg = model_reg
        self.model_clf = model_clf
        self.X_test = X_test
        self.y_true_reg = y_test_reg.values
        self.y_true_clf = y_test_clf.values
        self.fx_features_test = fx_features_test.values
        self.experiment_name = experiment_name
        
        self.preds_reg = model_reg.predict(X_test)
        self.preds_clf = model_clf.predict(X_test)

    def compute_econometric_metrics(self):
        # 1. Out-of-Sample R2
        oos_r2 = r2_score(self.y_true_reg, self.preds_reg)
        
        # 2. Mutual Information. Calculates and shows whether there is statistical dependence. 
        fx_reshaped = self.fx_features_test.reshape(-1, 1) if self.fx_features_test.ndim == 1 else self.fx_features_test
        mi_score = mutual_info_regression(fx_reshaped, self.y_true_reg, random_state=42)[0]
        
        # 3. Diebold-Mariano Test (vs Naive Zero Forecast). Basically checks whether model is better than just not predicting a gap. 
        naive_pred = np.zeros_like(self.y_true_reg)
        e_model = self.y_true_reg - self.preds_reg
        e_naive = self.y_true_reg - naive_pred
        
        d = (e_naive ** 2) - (e_model ** 2)
        d_mean = np.mean(d)
        gamma_0 = np.var(d, ddof=1)
        
        if gamma_0 == 0:
            dm_stat, p_value = 0.0, 1.0
        else:
            dm_stat = d_mean / np.sqrt(gamma_0 / len(d))
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat))) 
            
        # 4. Directional Accuracy (Secondary Evidence of Exploitability). Binomial test to make sure accuracy is higher than 50%. 
        actual_dir = np.sign(self.y_true_reg)
        pred_dir = np.where(self.preds_clf == 1, 1, -1)
        traded_mask = actual_dir != 0
        
        n_eval = int(traded_mask.sum())
        if n_eval > 0:
            correct_preds = (pred_dir[traded_mask] == actual_dir[traded_mask]).sum()
            acc = correct_preds / n_eval
            binom_p_val = stats.binomtest(int(correct_preds), n_eval, p=0.5, alternative='greater').pvalue
        else:
            acc, binom_p_val = 0.0, 1.0
            
        return {
            'OOS_R2': oos_r2,
            'Mutual_Info': mi_score,
            'DM_Stat': dm_stat,
            'DM_P_Val': p_value,
            'Accuracy': acc,
            'Binom_P_Val': binom_p_val,
            'N_Test': len(self.y_true_reg)
        }

def compute_rolling_r2(df_std: pd.DataFrame, fx_col: str, gap_col: str, window: int = 252):
    """Calculates expanding window Out-of-Sample R2."""
    df_clean = df_std[[fx_col, gap_col]].dropna()
    X = sm.add_constant(df_clean[fx_col])
    y = df_clean[gap_col]
    
    dates = df_clean.index[window:]
    rolling_r2 = []
    
    for i in range(window, len(df_clean)):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        X_test, y_test = X.iloc[i:i+1], y.iloc[i:i+1]
        
        model = sm.OLS(y_train, X_train).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        pred = model.predict(X_test)
        
        mse_model = (y_test.values[0] - pred.values[0]) ** 2
        mse_naive = (y_test.values[0] - 0) ** 2
        loc_r2 = max(-0.5, 1 - (mse_model / (mse_naive + 1e-8)))
        rolling_r2.append(loc_r2)
        
    return pd.Series(rolling_r2, index=dates).rolling(21).mean()

def run_all_experiments():
    cfg = ResearchConfig()
    eng = DataEngine(cfg)
    
    experiments = [
        ["5713"], ["5711"], ["5706"],
        ["5711", "5706"], ["5711", "5713"], ["5706", "5713"],
        ["5711", "5706", "5713"]
    ]
    
    results = []
    ts_data_cache = {} 
    all_importances = {}  
    
    for tickers in experiments:
        exp_name = " + ".join(tickers)
        logger.info(f"Running Inference on: {exp_name}")
        
        df, features = combine_datasets(eng, tickers)
        if df is None or len(df) < 20: continue
        
        # 1. Standardize features to extract Standardized Betas (Effect Sizes)
        cols_to_std = ['Gap', 'FX_Vol', 'FX_Ret', 'FX_Ret_2D', 'FX_Ret_5D', 'Gap_Lag1', 'Gap_Lag2']
        df_std = df.copy()
        for c in cols_to_std:
            if c in df_std.columns:
                df_std[c] = (df_std[c] - df_std[c].mean()) / df_std[c].std()
                
        # Cache single-ticker standard data for time-series plots
        if len(tickers) == 1:
            ts_data_cache[exp_name] = df_std
            
        train, test = split_chronologically(df_std, cfg.test_size)
        train_nonzero = train[train['Gap'] != 0.0]
        
        # 2. Parametric Inference (HAC OLS)
        nw_lags = int(4 * (len(train_nonzero) / 100) ** (2/9))
        
        try:
            sm_core = sm.add_constant(train_nonzero[['FX_Vol', 'FX_Ret']])
            model_core = sm.OLS(train_nonzero['Gap'], sm_core).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags, 'use_correction': True})
            beta_vol_core = model_core.params.get('FX_Vol', np.nan)
            p_vol_core = model_core.pvalues.get('FX_Vol', np.nan)
            beta_ret_core = model_core.params.get('FX_Ret', np.nan)
            p_ret_core = model_core.pvalues.get('FX_Ret', np.nan)
        except Exception:
            beta_vol_core, p_vol_core, beta_ret_core, p_ret_core = [np.nan]*4
            
        try:
            sm_full = sm.add_constant(train_nonzero[['FX_Vol', 'FX_Ret', 'FX_Ret_2D', 'FX_Ret_5D']])
            model_full = sm.OLS(train_nonzero['Gap'], sm_full).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags, 'use_correction': True})
            beta_vol_full = model_full.params.get('FX_Vol', np.nan)
            p_vol_full = model_full.pvalues.get('FX_Vol', np.nan)
            beta_ret_full = model_full.params.get('FX_Ret', np.nan)
            p_ret_full = model_full.pvalues.get('FX_Ret', np.nan)
        except Exception:
            beta_vol_full, p_vol_full, beta_ret_full, p_ret_full = [np.nan]*4

        # 3. Non-Parametric Inference (Random Forest)
        rf_reg = RandomForestRegressor(n_estimators=cfg.n_estimators, random_state=cfg.random_state)
        rf_reg.fit(train_nonzero[features], train_nonzero['Gap'])
        
        rf_clf = RandomForestClassifier(n_estimators=cfg.n_estimators, class_weight='balanced', random_state=cfg.random_state)
        rf_clf.fit(train_nonzero[features], train_nonzero['Target_Class'])
        
        importances = dict(zip(features, rf_reg.feature_importances_))
        all_importances[exp_name] = importances
        
        # 4. Out-of-Sample Evaluation
        reporter = ReportingEngine(rf_reg, rf_clf, test[features], test['Gap'], test['Target_Class'], test['FX_Ret'], exp_name)
        metrics = reporter.compute_econometric_metrics()
        
        metrics.update({
            'Experiment': exp_name,
            'Beta_Vol_Core': beta_vol_core, 'P_Vol_Core': p_vol_core,
            'Beta_Ret_Core': beta_ret_core, 'P_Ret_Core': p_ret_core,
            'Beta_Vol_Full': beta_vol_full, 'P_Vol_Full': p_vol_full,
            'Beta_Ret_Full': beta_ret_full, 'P_Ret_Full': p_ret_full
        })
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    # Holm-Bonferroni correction for multiple hypothesis testing on the DM test
    _, pvals_corrected, _, _ = multipletests(df_results['DM_P_Val'], alpha=0.05, method='holm')
    df_results['DM_Adj_P_Val'] = pvals_corrected

    print("\n" + "="*140)
    print("ECONOMETRIC LEAD-LAG EVIDENCE: USD/JPY OVERNIGHT RETURNS vs NEXT-DAY EQUITY GAP")
    print("="*140)
    
    print(df_results[['Experiment', 'N_Test', 'Beta_Ret_Core', 'P_Ret_Core', 'OOS_R2', 'Mutual_Info', 'DM_Stat', 'DM_Adj_P_Val', 'Accuracy']].to_string(
        index=False, formatters={
            'Beta_Ret_Core': '{:.3f}'.format, 'P_Ret_Core': '{:.4f}'.format,
            'OOS_R2': '{:.4f}'.format, 'Mutual_Info': '{:.4f}'.format,
            'DM_Stat': '{:.3f}'.format, 'DM_Adj_P_Val': '{:.4f}'.format, 'Accuracy': '{:.3f}'.format
        }))
    
    return df_results, ts_data_cache, all_importances, cfg

def plot_all_results(df_results, ts_data_cache, all_importances, cfg):
    """Generates 6 publication-quality econometric visualizations."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    # PLOT 1: HAC OLS Standardized Betas & Significance. Standardizes betas before OLS regression
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
    df_ols = df_results.melt(id_vars=['Experiment'], 
                             value_vars=['Beta_Vol_Core', 'Beta_Ret_Core'], var_name='Feature', value_name='Std_Beta')
    df_pvals = df_results.melt(id_vars=['Experiment'], 
                               value_vars=['P_Vol_Core', 'P_Ret_Core'], var_name='Feature', value_name='P_Value')
    
    df_ols['P_Value'] = df_pvals['P_Value']
    df_ols['Feature'] = df_ols['Feature'].replace({'Beta_Vol_Core': 'FX Volatility', 'Beta_Ret_Core': 'FX Return'})
    df_ols['Significant'] = df_ols['P_Value'] < 0.05
    
    sns.barplot(data=df_ols, x='Experiment', y='Std_Beta', hue='Feature', ax=axes1[0], palette='viridis')
    axes1[0].set_title(r"Standardized Effect Sizes (HAC OLS)", fontweight='bold')
    axes1[0].set_ylabel(r"Standardized Beta ($\beta$)")
    axes1[0].tick_params(axis='x', rotation=45)
    
    # HAC p-values correct for serial correlation and time volatility
    sns.scatterplot(data=df_ols, x='Experiment', y='P_Value', hue='Feature', style='Significant', 
                    s=150, ax=axes1[1], palette='viridis', markers={True: '*', False: 'o'})
    axes1[1].axhline(0.05, color='darkred', linestyle='--', label=r'$\alpha = 0.05$')
    axes1[1].set_title(r"Statistical Significance ($H_0: \beta = 0$)", fontweight='bold')
    axes1[1].set_ylabel(r"HAC P-Value")
    axes1[1].tick_params(axis='x', rotation=45)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(cfg.base_path, '1_OLS_Effect_Sizes.png'), dpi=300, bbox_inches='tight')

    # Time Series Plots (Require single ticker isolation)
    # Filter to only include the base tickers, ignoring combinations like "5711 + 5706"
    single_tickers = [k for k in ts_data_cache.keys() if " + " not in k]
    num_tickers = len(single_tickers)
    
    if num_tickers > 0:
        
        # PLOT 2: Cross-Correlation Function (CCF). 95% confidence level, computes the FX return vs gap
        fig2, axes2 = plt.subplots(1, num_tickers, figsize=(5 * num_tickers, 5), squeeze=False)
        axes2 = axes2.flatten()
        
        for i, ticker in enumerate(single_tickers):
            df_ts = ts_data_cache[ticker]
            ax = axes2[i]
            
            ccf = sm.tsa.stattools.ccf(df_ts['FX_Ret'], df_ts['Gap'], adjusted=False)[:cfg.max_lag_analysis+1]
            lags = np.arange(cfg.max_lag_analysis + 1)
            
            # Removed deprecated use_line_collection
            ax.stem(lags, ccf, basefmt="k-")
            conf_interval = 1.96 / np.sqrt(len(df_ts))
            ax.axhline(conf_interval, color='red', linestyle='--', alpha=0.5, label='95% Significance')
            ax.axhline(-conf_interval, color='red', linestyle='--', alpha=0.5)
            
            company_name = cfg.equities.get(ticker, "Unknown")
            ax.set_title(fr"CCF: FX Returns leading {ticker} ({company_name})" + "\n" + r"$H_0: \rho(FX_{t-k}, Gap_t) = 0$", fontweight='bold', fontsize=10)
            ax.set_xlabel(r"Lag $k$ (Days)")
            if i == 0:
                ax.set_ylabel(r"Pearson Correlation")
                ax.legend()
                
        fig2.tight_layout()
        fig2.savefig(os.path.join(cfg.base_path, '2_Cross_Correlation_All.png'), dpi=300, bbox_inches='tight')

        # PLOT 3: Granger Causality Heatmap (Aggregated into one plot). Answers the question, do past FX returns predict current equity gaps?
        fig3, ax3 = plt.subplots(figsize=(8, 1.5 * num_tickers + 2))
        granger_pvals = {}
        
        for ticker in single_tickers:
            df_ts = ts_data_cache[ticker]
            gc_data = df_ts[['Gap', 'FX_Ret']].dropna()
            gc_res = grangercausalitytests(gc_data, maxlag=cfg.max_lag_analysis, verbose=False)
            
            p_values = [gc_res[i][0]['ssr_ftest'][1] for i in range(1, cfg.max_lag_analysis + 1)]
            company_name = cfg.equities.get(ticker, "Unknown")
            row_label = f"{ticker} ({company_name})"
            granger_pvals[row_label] = p_values
            
        heatmap_data = pd.DataFrame(granger_pvals, index=range(1, cfg.max_lag_analysis + 1)).T
        
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm_r', cbar_kws={'label': r'P-Value'}, vmin=0, vmax=0.1, ax=ax3)
        ax3.set_title(r"Granger Causality: FX Returns $\rightarrow$ Equity Gap" + "\n" + r"$H_0$: FX does not Granger-cause Gap", fontweight='bold')
        ax3.set_xlabel(r"Lag Order")
        ax3.set_ylabel(r"Ticker")
        
        fig3.tight_layout()
        fig3.savefig(os.path.join(cfg.base_path, '3_Granger_Causality_All.png'), dpi=300, bbox_inches='tight')

        # PLOT 4: Impulse Response Function (IRF) - Fixed statsmodels ax hijacking. How does a shock in FX (of one standard deviation) affect equity gap over 5 days? 
        fig4, axes4 = plt.subplots(1, num_tickers, figsize=(6 * num_tickers, 5), squeeze=False)
        axes4 = axes4.flatten()
        
        for i, ticker in enumerate(single_tickers):
            df_ts = ts_data_cache[ticker]
            ax = axes4[i]
            
            var_data = df_ts[['FX_Ret', 'Gap']].dropna()
            model = VAR(var_data)
            results = model.fit(2) 
            irf_res = results.irf(5)
            
            # Manually extract orthogonalized IRF and standard errors
            # Index 1 is Gap (Response), Index 0 is FX_Ret (Impulse)
            point_estimates = irf_res.orth_irfs[:, 1, 0]
            std_errors = irf_res.stderr(orth=True)[:, 1, 0]
            steps = np.arange(len(point_estimates))
            
            # Plot the central response line and 95% confidence intervals
            ax.plot(steps, point_estimates, color='#1f77b4', linewidth=2)
            ax.fill_between(steps, 
                            point_estimates - 1.96 * std_errors, 
                            point_estimates + 1.96 * std_errors, 
                            color='#1f77b4', alpha=0.2)
            ax.axhline(0, color='black', linewidth=1)
            
            company_name = cfg.equities.get(ticker, "Unknown")
            ax.set_title(r"Orthogonalized Impulse Response: " + f"{ticker} ({company_name})\n" + r"Effect of 1-SD FX Shock on Equity Gap", fontweight='bold', fontsize=10)
            ax.set_xlabel(r"Days after shock")
            if i == 0:
                ax.set_ylabel(r"Response in Equity Gap (Z-Score)")
            else:
                ax.set_ylabel("")
                
        fig4.tight_layout()
        fig4.savefig(os.path.join(cfg.base_path, '4_Impulse_Response_All.png'), dpi=300, bbox_inches='tight')

        # PLOT 5: Rolling Out-of-Sample R2. Is predictive power stable over time?
        fig5, axes5 = plt.subplots(1, num_tickers, figsize=(6 * num_tickers, 5), squeeze=False)
        axes5 = axes5.flatten()
        
        for i, ticker in enumerate(single_tickers):
            df_ts = ts_data_cache[ticker]
            ax = axes5[i]
            
            r2_series = compute_rolling_r2(df_ts, 'FX_Ret', 'Gap', window=252)
            ax.plot(r2_series.index, r2_series.values, color='indigo', linewidth=1.5)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            
            company_name = cfg.equities.get(ticker, "Unknown")
            ax.set_title(fr"Rolling Predictive $R^2$: {ticker} ({company_name})" + "\n" + r"$H_0$: Out-of-sample predictive power vs Naive Forecast $\leq$ 0", fontweight='bold', fontsize=10)
            ax.set_xlabel(r"Date")
            ax.tick_params(axis='x', rotation=45)
            
            if i == 0:
                ax.set_ylabel(r"Out-of-Sample $R^2$")
            else:
                ax.set_ylabel("")
                
        fig5.tight_layout()
        fig5.savefig(os.path.join(cfg.base_path, '5_Rolling_R2_All.png'), dpi=300, bbox_inches='tight')

    # PLOT 6: Feature Importance. Which features are the most important for classification?
    df_imp = pd.DataFrame(all_importances).T 
    common_features = df_imp.columns[df_imp.notna().sum() >= 5].tolist()
    df_imp.fillna(0, inplace=True)
    df_imp_common = df_imp[common_features]
    
    df_imp_mean = df_imp_common.mean().sort_values(ascending=False)
    
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.bar(df_imp_mean.index, df_imp_mean.values, color='cadetblue', edgecolor='black', alpha=0.8)
    ax6.set_title(r"Random Forest Non-Linear Feature Importance" + "\n" + r"(Features present in $\geq$ 5/7 configurations)", fontweight='bold')
    ax6.set_ylabel(r"Mean Decrease Impurity")
    ax6.tick_params(axis='x', rotation=45)
    
    fig6.tight_layout()
    fig6.savefig(os.path.join(cfg.base_path, '6_Feature_Importance.png'), dpi=300, bbox_inches='tight')
    
    logger.info("Successfully generated and saved publication visualizations.")

if __name__ == "__main__":
    df_results, ts_cache, importances, cfg = run_all_experiments()
    if not df_results.empty:
        plot_all_results(df_results, ts_cache, importances, cfg)