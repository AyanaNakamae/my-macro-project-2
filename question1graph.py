import numpy as np
import matplotlib.pyplot as plt

# matplotlib設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.alpha'] = 0.7

# utility function
def util(cons, gamma):
    return max(cons, 1e-4)**(1.0-gamma)/(1.0-gamma)

# パラメータ設定
gamma = 2.0  # 相対的危険回避度
beta = 0.985**20  # 時間割引因子
r = 1.025**20 - 1.0  # 利子率
JJ = 3  # 3期間モデル
y = np.array([1.0, 1.2, 0.4])  # 各期の基本所得

# 生産性ショック
l = np.array([0.8027, 1.0, 1.2457])  # 低、中、高生産性
NL = 3  # 生産性状態の数

# 遷移確率行列
prob = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1361],
    [0.0021, 0.2528, 0.7451]
])

# 資産グリッド
a_l = 0.0
a_u = 2.0
NA = 100
a = np.linspace(a_l, a_u, NA)

# 初期化
v = np.zeros((JJ, NA, NL))
iaplus = np.zeros((JJ, NA, NL), dtype=int)
aplus = np.zeros((JJ, NA, NL))

# 後ろ向き帰納法

# 第3期（最終期）: 年金がないので労働所得のみ
for ia in range(NA):
    for il in range(NL):
        # 最終期の効用: 労働所得 + 利子付き資産
        v[JJ-1, ia, il] = util(l[il] + (1.0+r)*a[ia], gamma)

# 第2期
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            # 第2期の効用 + 期待効用
            reward[iap] = util(l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*v[JJ-1, iap, 0]
        iaplus[1, ia, il] = np.argmax(reward)
        aplus[1, ia, il] = a[iaplus[1, ia, il]]
        v[1, ia, il] = reward[iaplus[1, ia, il]]

# 第1期
for il in range(NL):
    for ia in range(NA):
        reward = np.zeros(NA)
        for iap in range(NA):
            # 期待価値の計算
            EV = 0.0
            for ilp in range(NL):
                EV += prob[il, ilp]*v[1, iap, ilp]
            
            reward[iap] = util(l[il] + (1.0+r)*a[ia] - a[iap], gamma) + beta*EV
        
        iaplus[0, ia, il] = np.argmax(reward)
        aplus[0, ia, il] = a[iaplus[0, ia, il]]
        v[0, ia, il] = reward[iaplus[0, ia, il]]

# グラフの作成（課題で指定された1つの図）
plt.figure(figsize=(10, 6))
plt.gca().set_facecolor('white')

# 第1期の政策関数（若年期期初の資産 → 次期の資産）
plt.plot(a, aplus[0, :, 0], marker='o', label='Low productivity (0.8027)', linewidth=2.5, markersize=5)
plt.plot(a, aplus[0, :, 1], marker='s', label='Mid productivity (1.0)', linewidth=2.5, markersize=5)
plt.plot(a, aplus[0, :, 2], marker='^', label='High productivity (1.2457)', linewidth=2.5, markersize=5)
plt.plot(a, a, '--', color='gray', alpha=0.7, label='45 degree line', linewidth=1.5)

plt.title("Savings Policy Function by Productivity Level (No Pension)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Initial Assets (Excluding Interest)", fontsize=14)
plt.ylabel("Next Period Assets (Excluding Interest)", fontsize=14)
plt.grid(True, color='lightgray', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')
plt.xlim(a_l, a_u)
plt.ylim(a_l, a_u)

# 軸の数値を見やすくする
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

# 分析結果の出力
print("=== 分析結果 ===")
print(f"パラメータ設定:")
print(f"  相対的危険回避度 γ = {gamma}")
print(f"  時間割引因子 β = {beta:.4f}")
print(f"  利子率 r = {r:.4f}")
print(f"  生産性: 低={l[0]:.4f}, 中={l[1]:.4f}, 高={l[2]:.4f}")

print("\n=== 各生産性での貯蓄行動の比較 ===")
# 資産水準0.5での貯蓄行動を比較
a_sample = 0.5
ia_sample = np.argmin(np.abs(a - a_sample))

print(f"\n期初資産 {a_sample:.1f} での次期資産:")
for il in range(NL):
    productivity_name = ['低生産性', '中生産性', '高生産性'][il]
    print(f"  {productivity_name}: {aplus[0, ia_sample, il]:.4f}")

# 貯蓄率の計算
print(f"\n期初資産 {a_sample:.1f} での貯蓄率:")
for il in range(NL):
    productivity_name = ['低生産性', '中生産性', '高生産性'][il]
    income = l[il] + (1.0+r)*a_sample
    saving = aplus[0, ia_sample, il]
    saving_rate = saving / income
    print(f"  {productivity_name}: {saving_rate:.4f} ({saving_rate*100:.1f}%)")
