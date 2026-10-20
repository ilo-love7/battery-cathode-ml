# -*- coding: utf-8 -*-
"""电池正极材料预测工具。

支持使用训练好的模型对新化学式进行预测：
- 带隙预测 (Egap)
- 形成能预测 (Eform)
- 平均电压预测 (Voltage)

示例:
    from predictor import predict_material_properties
    results = predict_material_properties(['LiFePO4', 'LiCoO2', 'LiMn2O4'])
    print(results)
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import featurize_one
from formula import parse_formula
from models import load_pickle


def load_models():
    """加载所有训练好的模型 bundle。"""
    models = {}
    
    # 带隙模型
    gap_pkl = ROOT / "results" / "models_gap.pkl"
    if gap_pkl.exists():
        models['gap'] = load_pickle(str(gap_pkl))
    
    # 形成能模型
    ef_pkl = ROOT / "results" / "models_ef.pkl"
    if ef_pkl.exists():
        models['ef'] = load_pickle(str(ef_pkl))
    
    # 电压模型
    voltage_pkl = ROOT / "results" / "models_voltage.pkl"
    if voltage_pkl.exists():
        models['voltage'] = load_pickle(str(voltage_pkl))
    
    return models


def predict_single(formula: str, models: Dict) -> Dict:
    """预测单个材料的性质。"""
    result = {"formula": formula}
    
    try:
        counts = parse_formula(formula)
        result["parsed"] = counts
        result["n_elements"] = len(counts)
        result["total_atoms"] = sum(counts.values())
        
        # 计算摩尔质量
        from features import ELEMENTS
        molar_mass = sum(counts[el] * ELEMENTS[el]["mass"] for el in counts)
        result["molar_mass_g_mol"] = round(molar_mass, 3)
        
    except Exception as e:
        result["parse_error"] = str(e)
        return result
    
    # Gap prediction
    if 'gap' in models:
        bundle = models['gap']
        element_list = bundle.get("element_list", [])
        
        frac_feats = featurize_one(formula, element_list=element_list)
        stat_feats = featurize_one(formula, element_list=None)
        
        if frac_feats is None or stat_feats is None:
            result["gap_error"] = "Feature extraction failed"
        else:
            # Create DataFrame with all expected columns
            frac_feats = featurize_one(formula, element_list=element_list)
            stat_feats = featurize_one(formula, element_list=None)
            
            if frac_feats is None or stat_feats is None:
                result["gap_error"] = "Feature extraction failed"
                return result
            
            frac_cols = [f"frac_{el}" for el in element_list]
            stat_cols = [k for k in stat_feats.keys() if k.startswith(('Z_', 'mass_', 'EN_', 'radius_', 'group_', 'period_', 'n_elements', 'atoms_per_fu', 'molar_mass'))]
            
            X_frac = pd.DataFrame([frac_feats], columns=frac_cols)
            X_stat = pd.DataFrame([stat_feats], columns=stat_cols)
            X = pd.concat([X_frac, X_stat], axis=1)
            
            import numpy as np
            rf_pred = bundle["models"]["RF"].predict(X.to_numpy())[0]
            hgb_pred = bundle["models"]["HGB"].predict(X.to_numpy())[0]
            ens_pred = 0.5 * (rf_pred + hgb_pred)
            spread = abs(rf_pred - hgb_pred) / 2
            
            result["Egap_RF_eV"] = round(rf_pred, 4)
            result["Egap_HGB_eV"] = round(hgb_pred, 4)
            result["Egap_ensemble_eV"] = round(ens_pred, 4)
            result["Egap_spread"] = round(spread, 4)
            
            # Conductivity assessment
            if ens_pred > 3.0:
                result["conductivity"] = "Insulator (>3 eV)"
            elif ens_pred > 1.0:
                result["conductivity"] = "Semiconductor (1-3 eV)"
            else:
                result["conductivity"] = "Narrow-gap/Metal (<1 eV)"
    
    # Formation energy prediction
    if 'ef' in models:
        bundle = models['ef']
        element_list = bundle.get("element_list", [])
        
        frac_feats = featurize_one(formula, element_list=element_list)
        stat_feats = featurize_one(formula, element_list=None)
        
        if frac_feats is None or stat_feats is None:
            result["ef_error"] = "Feature extraction failed"
        else:
            frac_cols = [f"frac_{el}" for el in element_list]
            stat_cols = [k for k in stat_feats.keys() if k.startswith(('Z_', 'mass_', 'EN_', 'radius_', 'group_', 'period_', 'n_elements', 'atoms_per_fu', 'molar_mass'))]
            
            X_frac = pd.DataFrame([frac_feats], columns=frac_cols)
            X_stat = pd.DataFrame([stat_feats], columns=stat_cols)
            X = pd.concat([X_frac, X_stat], axis=1)
            
            import numpy as np
            rf_pred = bundle["models"]["RF"].predict(X.to_numpy())[0]
            hgb_pred = bundle["models"]["HGB"].predict(X.to_numpy())[0]
            ens_pred = 0.5 * (rf_pred + hgb_pred)
            
            result["Eform_RF_eV_fu"] = round(rf_pred, 4)
            result["Eform_HGB_eV_fu"] = round(hgb_pred, 4)
            result["Eform_ensemble_eV_fu"] = round(ens_pred, 4)
            
            # Per atom formation energy
            atoms = float(sum(counts.values()))
            result["Eform_per_atom_eV"] = round(ens_pred / atoms, 4)
            
            # Stability assessment
            if ens_pred < -2.0:
                result["stability"] = "Highly stable (<-2 eV/fu)"
            elif ens_pred < 0:
                result["stability"] = "Stable (-2~0 eV/fu)"
            else:
                result["stability"] = "Unstable (>0 eV/fu)"
    
    # Voltage prediction
    if 'voltage' in models:
        bundle = models['voltage']
        element_list = bundle.get("element_list", [])
        
        frac_feats = featurize_one(formula, element_list=element_list)
        stat_feats = featurize_one(formula, element_list=None)
        
        if frac_feats is None or stat_feats is None:
            result["voltage_error"] = "Feature extraction failed"
        else:
            frac_cols = [f"frac_{el}" for el in element_list]
            stat_cols = [k for k in stat_feats.keys() if k.startswith(('Z_', 'mass_', 'EN_', 'radius_', 'group_', 'period_', 'n_elements', 'atoms_per_fu', 'molar_mass'))]
            
            X_frac = pd.DataFrame([frac_feats], columns=frac_cols)
            X_stat = pd.DataFrame([stat_feats], columns=stat_cols)
            X = pd.concat([X_frac, X_stat], axis=1)
            
            import numpy as np
            rf_pred = bundle["models"]["RF"].predict(X.to_numpy())[0]
            hgb_pred = bundle["models"]["HGB"].predict(X.to_numpy())[0]
            ens_pred = 0.5 * (rf_pred + hgb_pred)
            
            result["Voltage_RF_V"] = round(rf_pred, 3)
            result["Voltage_HGB_V"] = round(hgb_pred, 3)
            result["Voltage_ensemble_V"] = round(ens_pred, 3)
            
            # Voltage level assessment
            if ens_pred > 4.0:
                result["voltage_level"] = "High voltage (>4 V)"
            elif ens_pred > 3.5:
                result["voltage_level"] = "Medium voltage (3.5-4 V)"
            else:
                result["voltage_level"] = "Low voltage (<3.5 V)"
    
    return result


def predict_material_properties(formulas: List[str], 
                                 include_details: bool = False) -> Union[pd.DataFrame, List[Dict]]:
    """批量预测材料性质。
    
    Args:
        formulas: 化学式列表，如 ['LiFePO4', 'LiCoO2']
        include_details: 是否返回详细字典（默认返回 DataFrame）
    
    Returns:
        包含预测结果的 DataFrame 或字典列表
    """
    print(f"正在预测 {len(formulas)} 个材料的性质...")
    
    models = load_models()
    missing_models = [k for k in ['gap', 'ef', 'voltage'] if k not in models]
    if missing_models:
        print(f"警告：以下模型未找到，将跳过对应预测：{missing_models}")
    
    results = []
    for formula in formulas:
        pred = predict_single(formula, models)
        results.append(pred)
    
    if include_details:
        return results
    
    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 选择主要列
    main_cols = ["formula"]
    for col in ["Egap_ensemble_eV", "Eform_ensemble_eV_fu", "Voltage_ensemble_V"]:
        if col in df.columns:
            main_cols.append(col)
    
    for col in ["conductivity", "stability", "voltage_level"]:
        if col in df.columns:
            main_cols.append(col)
    
    # 添加可用列
    available_cols = [c for c in df.columns if c not in main_cols and not c.startswith('Egap_') and not c.startswith('Eform_') and not c.startswith('Voltage_')]
    main_cols.extend(available_cols[:5])  # 最多添加 5 个额外列
    
    return df[[c for c in main_cols if c in df.columns]]


def screen_candidates(candidates_file: str = None, 
                      n_top: int = 10,
                      min_voltage: float = 3.0,
                      max_voltage: float = 5.0) -> pd.DataFrame:
    """筛选候选正极材料。
    
    Args:
        candidates_file: 候选材料 CSV 文件路径（默认使用项目自带的 cathode_candidates.csv）
        n_top: 返回 Top N 个最优候选
        min_voltage: 最小电压阈值 (V)
        max_voltage: 最大电压阈值 (V)
    
    Returns:
        排序后的候选材料 DataFrame
    """
    if candidates_file is None:
        candidates_file = ROOT / "data" / "cathode_candidates.csv"
    
    if not Path(candidates_file).exists():
        raise FileNotFoundError(f"找不到候选文件：{candidates_file}")
    
    print(f"读取候选材料：{candidates_file}")
    df_cand = pd.read_csv(candidates_file, encoding='utf-8-sig')
    
    # 预测每个候选的性质
    all_results = []
    for _, row in df_cand.iterrows():
        result = predict_single(row['formula'], load_models())
        result['family'] = row.get('family', 'Unknown')
        result['name'] = row.get('name', row['formula'])
        all_results.append(result)
    
    df_result = pd.DataFrame(all_results)
    
    # 过滤电压范围
    if 'Voltage_ensemble_V' in df_result.columns:
        mask = (df_result['Voltage_ensemble_V'] >= min_voltage) & \
               (df_result['Voltage_ensemble_V'] <= max_voltage)
        df_result = df_result[mask]
        print(f"电压过滤 ({min_voltage}-{max_voltage} V): 剩余 {len(df_result)} 个候选")
    
    # 计算综合评分
    if all(col in df_result.columns for col in ['Egap_ensemble_eV', 'Eform_ensemble_eV_fu']):
        # 稳定性：按每原子形成能排名
        df_result['atoms'] = df_result['total_atoms']
        df_result['Eform_per_atom'] = df_result['Eform_ensemble_eV_fu'] / df_result['atoms']
        df_result['score_stability'] = df_result['Eform_per_atom'].rank(pct=True, ascending=False) * 100
        
        # 导电性：exp(-bandgap)
        df_result['score_conduction'] = np.exp(-df_result['Egap_ensemble_eV'].clip(lower=0)).rank(pct=True) * 100
        
        # 综合评分
        df_result['score_total'] = (0.40 * df_result['score_stability'] + 
                                     0.30 * df_result['score_conduction'])
        
        # 排序
        df_result = df_result.sort_values('score_total', ascending=False)
        
        print(f"\nTop {min(n_top, len(df_result))} 候选材料:")
        top_df = df_result.head(n_top)[['name', 'formula', 'family', 
                                         'Egap_ensemble_eV', 'Eform_ensemble_eV_fu',
                                         'Voltage_ensemble_V', 'score_total']]
        print(top_df.to_string(index=False))
    
    return df_result


if __name__ == "__main__":
    print("=" * 60)
    print("  电池正极材料性质预测工具")
    print("=" * 60)
    
    # 示例 1: 预测几个常见材料
    test_formulas = [
        "LiFePO4",      # 橄榄石
        "LiCoO2",       # 层状氧化物
        "LiMn2O4",      # 尖晶石
        "LiNi0.33Co0.33Mn0.34O2",  # NMC
        "Li3V2(PO4)3",  # 聚阴离子
    ]
    
    print("\n--- 示例 1: 常见材料性质预测 ---")
    results = predict_material_properties(test_formulas)
    print(results.to_string(index=False))
    
    # 示例 2: 筛选候选材料
    print("\n--- 示例 2: 候选材料筛选 (Top 5) ---")
    try:
        screened = screen_candidates(n_top=5, min_voltage=3.0, max_voltage=5.0)
    except FileNotFoundError as e:
        print(f"跳过筛选：{e}")
    
    print("\n" + "=" * 60)
    print("  预测完成!")
    print("=" * 60)
