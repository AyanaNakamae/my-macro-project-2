import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.patches as mpatches

# パラメータ設定
gamma = 2.0  # 相対的危険回避度
beta = 0.985  # 割引因子
R = 1.025**20  # 利子率(20年間)
tax_rate = 0.30  # 所得税率
pension_per_person = 0.4955  # 一人当たり年金額

# 生産性状態
productivity = np.array([0.8027, 1.0, 1.2457])

# 遷移確率行列
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])

# 定常分布（各生産性状態の人口割合）
eigenvals, eigenvecs = np.linalg.eig(P.T)
stationary_idx = np.argmax(np.real(eigenvals))
stationary_dist = np.real(eigenvecs[:, stationary_idx])
stationary_dist = stationary_dist / np.sum(stationary_dist)

print("定常分布:", stationary_dist)
print("生産性状態:", productivity)

# CRRA効用関数
def utility(c, gamma):
    if gamma == 1:
        return np.log(c)
    else:
        return (c**(1-gamma)) / (1-gamma)

# 年金制度なしの価値関数（問1の結果）
def value_function_no_pension(a2, a1, productivity_idx):
    """
    年金制度なしの価値関数
    a2: 次期の資産
    a1: 現在の資産
    productivity_idx: 生産性状態のインデックス
    """
    prod = productivity[productivity_idx]
    
    # 第1期の消費
    c1 = a1 + prod - a2/R
    if c1 <= 0:
        return -np.inf
    
    # 第2期の期待効用を計算
    expected_utility_2 = 0
    for j in range(3):
        prob = P[productivity_idx, j]
        prod_2 = productivity[j]
        
        # 第2期の最適消費・貯蓄を計算
        def objective_2(a3):
            c2 = a2 + prod_2 - a3/R
            if c2 <= 0:
                return np.inf
            
            # 第3期の期待効用
            expected_utility_3 = 0
            for k in range(3):
                prob_3 = P[j, k]
                prod_3 = productivity[k]
                c3 = a3 + prod_3
                if c3 <= 0:
                    return np.inf
                expected_utility_3 += prob_3 * utility(c3, gamma)
            
            return -(utility(c2, gamma) + beta * expected_utility_3)
        
        # 第2期の最適化
        result = minimize_scalar(objective_2, bounds=(0, (a2 + prod_2) * R), method='bounded')
        if result.success:
            expected_utility_2 += prob * (-result.fun)
        else:
            expected_utility_2 += prob * utility(a2 + prod_2, gamma)  # 貯蓄ゼロの場合
    
    return utility(c1, gamma) + beta * expected_utility_2

# 年金制度ありの価値関数
def value_function_with_pension(a2, a1, productivity_idx):
    """
    年金制度ありの価値関数
    a2: 次期の資産
    a1: 現在の資産
    productivity_idx: 生産性状態のインデックス
    """
    prod = productivity[productivity_idx]
    
    # 第1期の消費（税引き後所得）
    after_tax_income = prod * (1 - tax_rate)
    c1 = a1 + after_tax_income - a2/R
    if c1 <= 0:
        return -np.inf
    
    # 第2期の期待効用を計算
    expected_utility_2 = 0
    for j in range(3):
        prob = P[productivity_idx, j]
        prod_2 = productivity[j]
        after_tax_income_2 = prod_2 * (1 - tax_rate)
        
        # 第2期の最適消費・貯蓄を計算
        def objective_2(a3):
            c2 = a2 + after_tax_income_2 - a3/R
            if c2 <= 0:
                return np.inf
            
            # 第3期の期待効用（年金受給）
            expected_utility_3 = 0
            for k in range(3):
                prob_3 = P[j, k]
                c3 = a3 + pension_per_person  # 年金のみ
                if c3 <= 0:
                    return np.inf
                expected_utility_3 += prob_3 * utility(c3, gamma)
            
            return -(utility(c2, gamma) + beta * expected_utility_3)
        
        # 第2期の最適化
        result = minimize_scalar(objective_2, bounds=(0, (a2 + after_tax_income_2) * R), method='bounded')
        if result.success:
            expected_utility_2 += prob * (-result.fun)
        else:
            expected_utility_2 += prob * utility(a2 + after_tax_income_2, gamma)  # 貯蓄ゼロの場合
    
    return utility(c1, gamma) + beta * expected_utility_2

# 最適貯蓄政策関数を計算
def compute_optimal_savings(a1_grid, value_function):
    """最適貯蓄政策関数を計算"""
    optimal_savings = np.zeros((len(a1_grid), 3))
    
    for i, a1 in enumerate(a1_grid):
        for prod_idx in range(3):
            prod = productivity[prod_idx]
            
            # 制約条件を設定
            if value_function == value_function_with_pension:
                max_savings = (a1 + prod * (1 - tax_rate)) * R
            else:
                max_savings = (a1 + prod) * R
            
            # 最適化
            def objective(a2):
                return -value_function(a2, a1, prod_idx)
            
            result = minimize_scalar(objective, bounds=(0, max_savings), method='bounded')
            
            if result.success:
                optimal_savings[i, prod_idx] = result.x
            else:
                optimal_savings[i, prod_idx] = 0
    
    return optimal_savings

# 資産グリッドを設定
a1_grid = np.linspace(0, 2.0, 100)

# 年金制度なしの最適貯蓄
print("年金制度なしの最適貯蓄を計算中...")
optimal_savings_no_pension = compute_optimal_savings(a1_grid, value_function_no_pension)

# 年金制度ありの最適貯蓄
print("年金制度ありの最適貯蓄を計算中...")
optimal_savings_with_pension = compute_optimal_savings(a1_grid, value_function_with_pension)

# グラフの描画（年金制度ありのみ）
plt.figure(figsize=(10, 6))

# 年金制度ありの政策関数
plt.plot(a1_grid, optimal_savings_with_pension[:, 0], 'b-', label='Low productivity (0.8027)', linewidth=2)
plt.plot(a1_grid, optimal_savings_with_pension[:, 1], 'g-', label='Mid productivity (1.0)', linewidth=2)
plt.plot(a1_grid, optimal_savings_with_pension[:, 2], 'r-', label='High productivity (1.2457)', linewidth=2)

# 45度線
plt.plot(a1_grid, a1_grid, 'k--', label='45 degree line', alpha=0.5)

plt.xlabel('Initial Assets (Excluding Interest)')
plt.ylabel('Next Period Assets (Excluding Interest)')
plt.title('Savings Policy Function (With Pension System)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 貯蓄の差を計算
savings_diff = optimal_savings_no_pension - optimal_savings_with_pension

print("\n=== 分析結果 ===")
print(f"年金額: {pension_per_person:.4f}")
print(f"所得税率: {tax_rate*100:.1f}%")
print(f"利子率: {(R-1)*100:.2f}%")

print("\n=== 貯蓄水準の変化 ===")
for i, prod_name in enumerate(['低生産性', '中生産性', '高生産性']):
    avg_diff = np.mean(savings_diff[:, i])
    print(f"{prod_name}: 平均貯蓄減少額 = {avg_diff:.4f}")

# 特定の初期資産水準での比較
asset_levels = [0.5, 1.0, 1.5]
print("\n=== 特定の初期資産水準での貯蓄比較 ===")
for asset_level in asset_levels:
    idx = np.argmin(np.abs(a1_grid - asset_level))
    print(f"\n初期資産 {asset_level}:")
    for i, prod_name in enumerate(['低生産性', '中生産性', '高生産性']):
        no_pension = optimal_savings_no_pension[idx, i]
        with_pension = optimal_savings_with_pension[idx, i]
        diff = no_pension - with_pension
        print(f"  {prod_name}: {no_pension:.4f} → {with_pension:.4f} (差: {diff:.4f})")
