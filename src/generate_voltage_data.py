# -*- coding: utf-8 -*-
"""生成电池正极材料平均电压训练数据。

由于真实的 Materials Project 电池电压数据需要 API Key，
本脚本生成一个基于物理规律的模拟数据集用于演示和验证流程。

数据包含常见锂电池正极材料的典型电压范围：
- 橄榄石族 (LiFePO4, LiMnPO4 等): ~3.4-4.1 V
- 层状氧化物 (LiCoO2, LiNiO2, NMC 系列): ~3.7-3.9 V
- 尖晶石族 (LiMn2O4, LiNi0.5Mn1.5O4): ~3.7-4.7 V
- 聚阴离子 (Li3V2(PO4)3, LiVPO4F): ~3.6-4.2 V
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "battery_voltage.csv"


def generate_voltage_data(n_samples=500, seed=42):
    """生成模拟的电池正极材料电压数据。
    
    根据化学式中的元素组成，使用经验规则模拟电压值。
    实际应用中应替换为 Materials Project API 拉取的真实数据。
    """
    rng = np.random.default_rng(seed)
    
    # 定义不同材料族的基准电压
    families = {
        'olivine': {'formulas': ['LiFePO4', 'LiMnPO4', 'LiCoPO4', 'LiNiPO4', 'Li2FeSiO4', 'Li2MnSiO4'], 
                    'voltage_base': 3.6, 'voltage_std': 0.2},
        'layered': {'formulas': ['LiCoO2', 'LiNiO2', 'LiMnO2', 'LiNi0.33Co0.33Mn0.34O2', 
                                  'LiNi0.5Co0.2Mn0.3O2', 'LiNi0.8Co0.15Al0.05O2',
                                  'LiNi0.5Mn1.5O4', 'LiMn1.5Ni0.5O4'],
                    'voltage_base': 3.8, 'voltage_std': 0.15},
        'spinel': {'formulas': ['LiMn2O4', 'LiNi0.5Mn1.5O4', 'LiCoMnO4', 'LiFeMnPO4'],
                   'voltage_base': 4.0, 'voltage_std': 0.3},
        'polyanion': {'formulas': ['Li3V2(PO4)3', 'LiVPO4F', 'Li2FeSO4', 'Li2CoSO4'],
                      'voltage_base': 3.9, 'voltage_std': 0.25},
        'vanadium_oxide': {'formulas': ['LiV3O8', 'V2O5', 'LiVO2', 'Li0.5V2O5'],
                           'voltage_base': 3.2, 'voltage_std': 0.2},
    }
    
    data = []
    for family, config in families.items():
        n_family = n_samples // len(families)
        for _ in range(n_family):
            formula = rng.choice(config['formulas'])
            voltage = rng.normal(config['voltage_base'], config['voltage_std'])
            voltage = max(2.0, min(5.5, voltage))  # 限制在合理范围内
            
            # 添加一些额外的噪声
            voltage += rng.normal(0, 0.05)
            
            # 计算理论容量（简化版）
            capacity = rng.uniform(120, 200)
            
            data.append({
                'material_id': f"sim_{len(data):04d}",
                'formula': formula,
                'family': family,
                'average_voltage_V': round(voltage, 3),
                'specific_capacity_mAh_g': round(capacity, 1),
                'energy_density_mWh_g': round(voltage * capacity, 1)
            })
    
    df = pd.DataFrame(data)
    
    # 打乱顺序
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    print("正在生成电池电压训练数据...")
    df = generate_voltage_data(n_samples=500)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print(f"[OK] 生成 {len(df)} 条样本")
    print(f"[OK] 电压范围：{df['average_voltage_V'].min():.2f} - {df['average_voltage_V'].max():.2f} V")
    print(f"[OK] 平均电压：{df['average_voltage_V'].mean():.3f} V")
    print(f"[OK] 保存至：{OUTPUT_PATH}")
    print("\n电压分布统计:")
    print(df.groupby('family')['average_voltage_V'].agg(['mean', 'std', 'count']).round(3))
