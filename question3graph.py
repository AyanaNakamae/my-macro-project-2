import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import matplotlib.font_manager as fm

# Set up Japanese font for matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

# Parameters
gamma = 2.0  # Risk aversion parameter
beta = 0.985  # Discount factor
r = 0.025  # Interest rate
tax_rate = 0.30  # Income tax rate
productivity_levels = np.array([0.8027, 1.0, 1.2457])
pension_per_person = 0.3124  # From problem 2

# Transition matrix P
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])

# Utility function
def utility(c, gamma):
    if gamma == 1:
        return np.log(c)
    else:
        return (c**(1-gamma)) / (1-gamma)

# Expected utility calculation
def expected_utility_period3(c2, a2, gamma, beta, P, productivity_levels, pension=0):
    """Calculate expected utility for period 3 given consumption in period 2 and assets"""
    eu3 = 0
    for j in range(3):  # Future productivity states
        for i in range(3):  # Current productivity states (for transition probability)
            # Period 3 consumption = assets + pension (no labor income in old age)
            c3 = a2 * (1 + r) + pension
            if c3 > 0:
                eu3 += P[i, j] * utility(c3, gamma)
    return eu3

def solve_individual_problem_with_pension(e1, gamma, beta, r, tax_rate, P, productivity_levels, pension):
    """Solve the individual optimization problem with pension system"""
    
    def objective(a1):
        """Objective function to minimize (negative of expected utility)"""
        # Period 1 consumption
        c1 = e1 - a1
        if c1 <= 0:
            return 1e10
        
        u1 = utility(c1, gamma)
        
        # Expected utility from periods 2 and 3
        eu2_plus_3 = 0
        
        for i in range(3):  # Period 2 productivity states
            prob_i = 1/3  # Stationary probability (equal distribution)
            
            # Period 2 after-tax income
            after_tax_income = productivity_levels[i] * (1 - tax_rate)
            
            # Period 2 resources = assets + after-tax income
            resources_2 = a1 * (1 + r) + after_tax_income
            
            # Optimal period 2 problem: choose a2 to maximize utility
            def period2_objective(a2):
                c2 = resources_2 - a2
                if c2 <= 0:
                    return 1e10
                
                u2 = utility(c2, gamma)
                
                # Expected utility from period 3
                eu3 = 0
                for j in range(3):
                    c3 = a2 * (1 + r) + pension
                    if c3 > 0:
                        eu3 += P[i, j] * utility(c3, gamma)
                
                return -(u2 + beta * eu3)
            
            # Solve for optimal a2
            result = minimize_scalar(period2_objective, bounds=(0, resources_2), method='bounded')
            optimal_a2 = result.x
            max_utility_2_plus_3 = -result.fun
            
            eu2_plus_3 += prob_i * max_utility_2_plus_3
        
        total_utility = u1 + beta * eu2_plus_3
        return -total_utility
    
    # Solve for optimal a1
    result = minimize_scalar(objective, bounds=(0, e1), method='bounded')
    optimal_a1 = result.x
    
    return optimal_a1

def solve_individual_problem_no_pension(e1, gamma, beta, r, P, productivity_levels):
    """Solve the individual optimization problem without pension system"""
    
    def objective(a1):
        """Objective function to minimize (negative of expected utility)"""
        # Period 1 consumption
        c1 = e1 - a1
        if c1 <= 0:
            return 1e10
        
        u1 = utility(c1, gamma)
        
        # Expected utility from periods 2 and 3
        eu2_plus_3 = 0
        
        for i in range(3):  # Period 2 productivity states
            prob_i = 1/3  # Stationary probability (equal distribution)
            
            # Period 2 income (no tax in no-pension case)
            income_2 = productivity_levels[i]
            
            # Period 2 resources = assets + income
            resources_2 = a1 * (1 + r) + income_2
            
            # Optimal period 2 problem: choose a2 to maximize utility
            def period2_objective(a2):
                c2 = resources_2 - a2
                if c2 <= 0:
                    return 1e10
                
                u2 = utility(c2, gamma)
                
                # Expected utility from period 3
                eu3 = 0
                for j in range(3):
                    c3 = a2 * (1 + r)  # No pension
                    if c3 > 0:
                        eu3 += P[i, j] * utility(c3, gamma)
                
                return -(u2 + beta * eu3)
            
            # Solve for optimal a2
            result = minimize_scalar(period2_objective, bounds=(0, resources_2), method='bounded')
            optimal_a2 = result.x
            max_utility_2_plus_3 = -result.fun
            
            eu2_plus_3 += prob_i * max_utility_2_plus_3
        
        total_utility = u1 + beta * eu2_plus_3
        return -total_utility
    
    # Solve for optimal a1
    result = minimize_scalar(objective, bounds=(0, e1), method='bounded')
    optimal_a1 = result.x
    
    return optimal_a1

# Calculate savings policy functions
initial_assets_range = np.linspace(0.01, 2.0, 100)
savings_no_pension = {i: [] for i in range(3)}
savings_with_pension = {i: [] for i in range(3)}

print("Calculating savings policy functions...")
print("Without pension system:")
for i, productivity in enumerate(productivity_levels):
    print(f"  Processing productivity level {i+1}: {productivity}")
    for a in initial_assets_range:
        # Total initial resources = assets + productivity
        total_resources = a + productivity
        optimal_saving_no_pension = solve_individual_problem_no_pension(
            total_resources, gamma, beta, r, P, productivity_levels)
        savings_no_pension[i].append(optimal_saving_no_pension)

print("\nWith pension system:")
for i, productivity in enumerate(productivity_levels):
    print(f"  Processing productivity level {i+1}: {productivity}")
    for a in initial_assets_range:
        # Total initial resources = assets + after-tax productivity
        after_tax_productivity = productivity * (1 - tax_rate)
        total_resources = a + after_tax_productivity
        optimal_saving_with_pension = solve_individual_problem_with_pension(
            total_resources, gamma, beta, r, tax_rate, P, productivity_levels, pension_per_person)
        savings_with_pension[i].append(optimal_saving_with_pension)

# Create single comparison plot with all lines
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

productivity_labels = ['Low (0.8027)', 'Mid (1.0)', 'High (1.2457)']
colors = ['blue', 'green', 'red']

for i in range(3):
    ax.plot(initial_assets_range, savings_no_pension[i], 
           color=colors[i], linewidth=2, label=f'{productivity_labels[i]} - No Pension')
    ax.plot(initial_assets_range, savings_with_pension[i], 
           color=colors[i], linewidth=2, linestyle='--', alpha=0.8,
           label=f'{productivity_labels[i]} - With Pension')

# Add 45-degree line
ax.plot(initial_assets_range, initial_assets_range, 
       'k--', alpha=0.3, label='45° line')

ax.set_xlabel('Initial Assets (Excluding Interest)', fontsize=12)
ax.set_ylabel('Next Period Assets (Excluding Interest)', fontsize=12)
ax.set_title('Savings Policy Function by Productivity Level (Pension vs No Pension)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 2.0)

plt.tight_layout()
plt.savefig('savings_policy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph generated successfully!")
