"""
================================================================================
INVENTORY FLOW & SUPPLY CHAIN TRENDS - COMPLETE ANALYSIS
================================================================================
Capstone Project: Statistical Methods and Essential Machine Learning Models
Student: [Your Name]
Date: February 9, 2026

This script performs:
1. Data loading and initial exploration
2. Data cleaning and feature engineering
3. Exploratory data analysis (EDA)
4. Statistical hypothesis testing (t-test, ANOVA, Chi-Square, Correlation)
5. Machine learning modeling (Regression, Classification, Clustering)
6. Visualization and reporting

Requirements:
pip install pandas numpy scipy scikit-learn matplotlib seaborn
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, classification_report, confusion_matrix,
                             silhouette_score)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("INVENTORY FLOW & SUPPLY CHAIN TRENDS - COMPLETE ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n[PART 1] DATA LOADING AND INITIAL EXPLORATION")
print("-" * 80)

# Load your dataset
# UPDATE THIS PATH to your actual dataset location
df_raw = pd.read_csv('supply_chain_dataset.csv')

print(f"\n✓ Dataset loaded successfully!")
print(f"  Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

print(f"\n📋 Column Names:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\n🔍 First 5 Rows:")
print(df_raw.head())

print(f"\n📊 Data Types:")
print(df_raw.dtypes)

print(f"\n📈 Descriptive Statistics:")
print(df_raw.describe())

print(f"\n⚠️  Missing Values:")
missing = df_raw.isnull().sum()
if missing.sum() > 0:
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': (missing.values / len(df_raw)) * 100
    })
    print(missing_df[missing_df['Missing_Count'] > 0].to_string(index=False))
else:
    print("  No missing values found!")

print(f"\n🔄 Duplicate Rows: {df_raw.duplicated().sum()}")

# ============================================================================
# PART 2: DATA CLEANING AND FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("[PART 2] DATA CLEANING AND FEATURE ENGINEERING")
print("-" * 80)

# Create a copy for cleaning
df = df_raw.copy()

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"\n✓ Removed {initial_rows - len(df)} duplicate rows")

# Convert date columns if they exist
if 'Order_Date' in df.columns and 'Shipment_Date' in df.columns:
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Shipment_Date'] = pd.to_datetime(df['Shipment_Date'])
    print("✓ Converted date columns to datetime")

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute numeric columns with median
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✓ Imputed {col} with median: {median_val:.2f}")

# Impute categorical columns with mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"✓ Imputed {col} with mode: {mode_val}")

# Feature Engineering - Create new calculated features
print("\n🔧 Creating Engineered Features:")

if 'Order_Date' in df.columns and 'Shipment_Date' in df.columns:
    df['Actual_Lead_Time'] = (df['Shipment_Date'] - df['Order_Date']).dt.days
    print("  ✓ Actual_Lead_Time (days between order and shipment)")

if 'Actual_Lead_Time' in df.columns and 'Lead_Time' in df.columns:
    df['Lead_Time_Variance'] = df['Actual_Lead_Time'] - df['Lead_Time']
    print("  ✓ Lead_Time_Variance (actual vs planned)")

if 'Demand_Forecast' in df.columns and 'Inventory_Level' in df.columns:
    df['Demand_vs_Inventory'] = df['Demand_Forecast'] - df['Inventory_Level']
    print("  ✓ Demand_vs_Inventory (forecast minus stock)")

if 'Shipment_Quantity' in df.columns and 'Order_Quantity' in df.columns:
    df['Fulfillment_Rate'] = (df['Shipment_Quantity'] / df['Order_Quantity']) * 100
    print("  ✓ Fulfillment_Rate (% of order fulfilled)")

if 'Inventory_Level' in df.columns and 'Demand_Forecast' in df.columns and 'Stockout_Flag' in df.columns:
    df['Stockout_Risk'] = ((df['Inventory_Level'] < df['Demand_Forecast']) & (df['Stockout_Flag'] == 0)).astype(int)
    print("  ✓ Stockout_Risk (predictive flag)")

print(f"\n✓ Final dataset: {len(df):,} rows × {len(df.columns)} columns")

# Save cleaned data
df.to_csv('cleaned_supply_chain_data.csv', index=False)
print("✓ Saved: cleaned_supply_chain_data.csv")

# ============================================================================
# PART 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("[PART 3] EXPLORATORY DATA ANALYSIS")
print("-" * 80)

# Key Performance Indicators
print("\n📊 SUPPLY CHAIN KEY PERFORMANCE INDICATORS:")
print("-" * 80)

if 'Stockout_Flag' in df.columns:
    stockout_count = df['Stockout_Flag'].sum()
    print(f"Total Orders:              {len(df):,}")
    print(f"Stockout Orders:           {stockout_count:,} ({(stockout_count/len(df)*100):.1f}%)")

if 'Backorder_Flag' in df.columns:
    backorder_count = df['Backorder_Flag'].sum()
    print(f"Backorder Orders:          {backorder_count:,} ({(backorder_count/len(df)*100):.1f}%)")

if 'Order_Priority' in df.columns:
    high_priority = (df['Order_Priority'] == 'High').sum()
    print(f"High Priority Orders:      {high_priority:,} ({(high_priority/len(df)*100):.1f}%)")

if 'Lead_Time' in df.columns:
    print(f"\nAverage Lead Time:         {df['Lead_Time'].mean():.2f} days")

if 'Fulfillment_Rate' in df.columns:
    print(f"Average Fulfillment Rate:  {df['Fulfillment_Rate'].mean():.1f}%")

if 'Order_Quantity' in df.columns:
    print(f"Average Order Quantity:    {df['Order_Quantity'].mean():.0f} units")

if 'Inventory_Level' in df.columns:
    print(f"Average Inventory Level:   {df['Inventory_Level'].mean():.0f} units")

# Performance by Warehouse
if 'Warehouse' in df.columns:
    print("\n📈 PERFORMANCE BY WAREHOUSE:")
    print("-" * 80)
    warehouse_cols = ['Order_Quantity', 'Stockout_Flag', 'Lead_Time']
    available_cols = [col for col in warehouse_cols if col in df.columns]
    if available_cols:
        warehouse_stats = df.groupby('Warehouse')[available_cols].agg({
            available_cols[0]: 'sum' if available_cols[0] == 'Order_Quantity' else 'mean',
            **{col: 'sum' if 'Flag' in col else 'mean' for col in available_cols[1:]}
        }).round(2)
        print(warehouse_stats)

# Performance by Product Category
if 'Product_Category' in df.columns:
    print("\n📦 PERFORMANCE BY PRODUCT CATEGORY:")
    print("-" * 80)
    category_cols = ['Order_Quantity', 'Stockout_Flag', 'Product_Price']
    available_cols = [col for col in category_cols if col in df.columns]
    if available_cols:
        category_stats = df.groupby('Product_Category')[available_cols].agg('mean').round(2)
        print(category_stats)

# Visualizations
print("\n📊 Creating EDA Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis - Supply Chain Dataset', fontsize=16, fontweight='bold')

# Plot 1: Distribution of Order Quantity
if 'Order_Quantity' in df.columns:
    axes[0, 0].hist(df['Order_Quantity'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(df['Order_Quantity'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_title('Distribution of Order Quantity')
    axes[0, 0].set_xlabel('Order Quantity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

# Plot 2: Box plot of Lead Time
if 'Lead_Time' in df.columns:
    axes[0, 1].boxplot(df['Lead_Time'])
    axes[0, 1].set_title('Lead Time Distribution')
    axes[0, 1].set_ylabel('Days')

# Plot 3: Bar chart of categories
if 'Product_Category' in df.columns:
    cat_counts = df['Product_Category'].value_counts()
    axes[1, 0].bar(range(len(cat_counts)), cat_counts.values, color='green', alpha=0.7)
    axes[1, 0].set_xticks(range(len(cat_counts)))
    axes[1, 0].set_xticklabels(cat_counts.index, rotation=45, ha='right')
    axes[1, 0].set_title('Orders by Product Category')
    axes[1, 0].set_ylabel('Count')

# Plot 4: Scatter plot
if 'Demand_Forecast' in df.columns and 'Inventory_Level' in df.columns:
    axes[1, 1].scatter(df['Demand_Forecast'], df['Inventory_Level'], alpha=0.5, color='purple')
    axes[1, 1].set_xlabel('Demand Forecast')
    axes[1, 1].set_ylabel('Inventory Level')
    axes[1, 1].set_title('Demand vs Inventory')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_overview.png")

# Correlation Matrix
print("\n📊 Creating Correlation Matrix...")
numeric_cols_for_corr = [col for col in numeric_cols if col in df.columns][:10]  # First 10 numeric columns
if len(numeric_cols_for_corr) >= 3:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_cols_for_corr].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")

# ============================================================================
# PART 4: STATISTICAL HYPOTHESIS TESTING
# ============================================================================
print("\n" + "="*80)
print("[PART 4] STATISTICAL HYPOTHESIS TESTING")
print("="*80)

# Test 1: Independent t-test
if 'Lead_Time' in df.columns and 'Stockout_Flag' in df.columns:
    print("\n[TEST 1] Independent t-test: Lead Time vs Stockout")
    print("-" * 80)
    
    stockout_lead = df[df['Stockout_Flag'] > 0]['Lead_Time']
    no_stockout_lead = df[df['Stockout_Flag'] == 0]['Lead_Time']
    
    t_stat, p_value_t = ttest_ind(stockout_lead, no_stockout_lead)
    
    print(f"H0: Mean lead time is SAME for stockout and non-stockout orders")
    print(f"H1: Mean lead time is DIFFERENT")
    print(f"\nStockout orders - Mean: {stockout_lead.mean():.2f} days (n={len(stockout_lead)})")
    print(f"Non-stockout orders - Mean: {no_stockout_lead.mean():.2f} days (n={len(no_stockout_lead)})")
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value_t:.4f}")
    print(f"\nConclusion: {'✓ REJECT H0' if p_value_t < 0.05 else '✗ FAIL TO REJECT H0'} (α=0.05)")

# Test 2: ANOVA
if 'Warehouse' in df.columns and 'Inventory_Level' in df.columns:
    print("\n[TEST 2] ANOVA: Inventory Level Across Warehouses")
    print("-" * 80)
    
    warehouse_groups = [df[df['Warehouse'] == wh]['Inventory_Level'] for wh in df['Warehouse'].unique()]
    f_stat, p_value_anova = f_oneway(*warehouse_groups)
    
    print(f"H0: Mean inventory level is SAME across all warehouses")
    print(f"H1: At least one warehouse has different mean")
    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value_anova:.4f}")
    print(f"\nConclusion: {'✓ REJECT H0' if p_value_anova < 0.05 else '✗ FAIL TO REJECT H0'} (α=0.05)")

# Test 3: Chi-Square Test
if 'Order_Priority' in df.columns and 'Backorder_Flag' in df.columns:
    print("\n[TEST 3] Chi-Square Test: Order Priority vs Backorder")
    print("-" * 80)
    
    contingency_table = pd.crosstab(df['Order_Priority'], df['Backorder_Flag'])
    chi2, p_value_chi, dof, expected = chi2_contingency(contingency_table)
    
    print(f"H0: Order priority and backorder are INDEPENDENT")
    print(f"H1: They are DEPENDENT")
    print(f"\nContingency Table:")
    print(contingency_table)
    print(f"\nChi-square: {chi2:.4f}")
    print(f"p-value: {p_value_chi:.4f}")
    print(f"\nConclusion: {'✓ REJECT H0' if p_value_chi < 0.05 else '✗ FAIL TO REJECT H0'} (α=0.05)")

# Test 4: Pearson Correlation
if 'Demand_Forecast' in df.columns and 'Inventory_Level' in df.columns:
    print("\n[TEST 4] Pearson Correlation: Demand vs Inventory")
    print("-" * 80)
    
    corr_coef, p_value_corr = pearsonr(df['Demand_Forecast'], df['Inventory_Level'])
    
    print(f"H0: No correlation (ρ = 0)")
    print(f"H1: Correlation exists (ρ ≠ 0)")
    print(f"\nCorrelation coefficient (r): {corr_coef:.4f}")
    print(f"p-value: {p_value_corr:.4f}")
    print(f"\nConclusion: {'✓ REJECT H0' if p_value_corr < 0.05 else '✗ FAIL TO REJECT H0'} (α=0.05)")

# ============================================================================
# PART 5: MACHINE LEARNING MODELS
# ============================================================================
print("\n" + "="*80)
print("[PART 5] MACHINE LEARNING MODELS")
print("="*80)

# Encode categorical variables
if 'Warehouse' in df.columns:
    le_warehouse = LabelEncoder()
    df['Warehouse_encoded'] = le_warehouse.fit_transform(df['Warehouse'])

if 'Product_Category' in df.columns:
    le_category = LabelEncoder()
    df['Category_encoded'] = le_category.fit_transform(df['Product_Category'])

if 'Order_Priority' in df.columns:
    le_priority = LabelEncoder()
    df['Priority_encoded'] = le_priority.fit_transform(df['Order_Priority'])

# MODEL 1: LINEAR REGRESSION
print("\n[MODEL 1] Linear Regression: Predicting Order Quantity")
print("-" * 80)

if 'Order_Quantity' in df.columns:
    # Select features for regression
    feature_candidates = ['Demand_Forecast', 'Inventory_Level', 'Lead_Time', 
                         'Product_Price', 'Warehouse_encoded', 'Category_encoded']
    features_reg = [f for f in feature_candidates if f in df.columns]
    
    if len(features_reg) >= 2:
        X_reg = df[features_reg]
        y_reg = df['Order_Quantity']
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler_reg.transform(X_test_reg)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_reg_scaled, y_train_reg)
        y_pred_reg = lr_model.predict(X_test_reg_scaled)
        
        r2 = r2_score(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        
        print(f"Features: {', '.join(features_reg)}")
        print(f"\nModel Performance:")
        print(f"  R² Score:  {r2:.4f}")
        print(f"  RMSE:      {rmse:.2f}")
        print(f"  MAE:       {mae:.2f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(y_test_reg, y_pred_reg, alpha=0.5, color='steelblue')
        axes[0].plot([y_test_reg.min(), y_test_reg.max()], 
                    [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Order Quantity')
        axes[0].set_ylabel('Predicted Order Quantity')
        axes[0].set_title(f'Regression: Actual vs Predicted\nR² = {r2:.4f}')
        axes[0].grid(True, alpha=0.3)
        
        residuals = y_test_reg - y_pred_reg
        axes[1].scatter(y_pred_reg, residuals, alpha=0.5, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Order Quantity')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_model.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: regression_model.png")

# MODEL 2: LOGISTIC REGRESSION
print("\n[MODEL 2] Logistic Regression: Predicting Stockout")
print("-" * 80)

if 'Stockout_Flag' in df.columns:
    feature_candidates = ['Demand_Forecast', 'Inventory_Level', 'Lead_Time',
                         'Order_Quantity', 'Warehouse_encoded', 'Priority_encoded']
    features_class = [f for f in feature_candidates if f in df.columns]
    
    if len(features_class) >= 2:
        X_class = df[features_class]
        y_class = df['Stockout_Flag']
        
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
        )
        
        scaler_class = StandardScaler()
        X_train_class_scaled = scaler_class.fit_transform(X_train_class)
        X_test_class_scaled = scaler_class.transform(X_test_class)
        
        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_train_class_scaled, y_train_class)
        y_pred_class = log_model.predict(X_test_class_scaled)
        
        accuracy = accuracy_score(y_test_class, y_pred_class)
        conf_matrix = confusion_matrix(y_test_class, y_pred_class)
        
        print(f"Features: {', '.join(features_class)}")
        print(f"\nModel Performance:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y_test_class, y_pred_class))
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        feature_importance = pd.DataFrame({
            'Feature': features_class,
            'Importance': np.abs(log_model.coef_[0])
        }).sort_values('Importance', ascending=True)
        
        axes[1].barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
        axes[1].set_xlabel('Absolute Coefficient Value')
        axes[1].set_title('Feature Importance')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('classification_model.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: classification_model.png")

# MODEL 3: K-MEANS CLUSTERING
print("\n[MODEL 3] K-Means Clustering: Order Segmentation")
print("-" * 80)

feature_candidates = ['Demand_Forecast', 'Inventory_Level', 'Order_Quantity',
                     'Lead_Time', 'Product_Price']
features_cluster = [f for f in feature_candidates if f in df.columns]

if len(features_cluster) >= 3:
    X_cluster = df[features_cluster].copy()
    
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    # Find optimal k
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_cluster_scaled)
    
    df['Cluster'] = cluster_labels
    
    print(f"Features: {', '.join(features_cluster)}")
    print(f"Optimal clusters: {optimal_k}")
    print(f"Silhouette Score: {silhouette_score(X_cluster_scaled, cluster_labels):.4f}")
    
    print(f"\nCluster Characteristics:")
    for i in range(optimal_k):
        cluster_data = df[df['Cluster'] == i]
        print(f"\nCluster {i} ({len(cluster_data):,} orders - {len(cluster_data)/len(df)*100:.1f}%):")
        for feature in features_cluster:
            print(f"  Avg {feature}: {cluster_data[feature].mean():.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Elbow curve
    axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Silhouette scores
    axes[0, 1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title(f'Silhouette Analysis (Optimal k={optimal_k})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cluster scatter plots
    if len(features_cluster) >= 2:
        scatter1 = axes[1, 0].scatter(df[features_cluster[0]], df[features_cluster[1]], 
                                       c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
        axes[1, 0].set_xlabel(features_cluster[0])
        axes[1, 0].set_ylabel(features_cluster[1])
        axes[1, 0].set_title(f'Clusters (k={optimal_k})')
        plt.colorbar(scatter1, ax=axes[1, 0], label='Cluster')
        
    if len(features_cluster) >= 4:
        scatter2 = axes[1, 1].scatter(df[features_cluster[2]], df[features_cluster[3]], 
                                       c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
        axes[1, 1].set_xlabel(features_cluster[2])
        axes[1, 1].set_ylabel(features_cluster[3])
        axes[1, 1].set_title(f'Clusters (k={optimal_k})')
        plt.colorbar(scatter2, ax=axes[1, 1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig('clustering_model.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: clustering_model.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"""
✅ FILES CREATED:
   • cleaned_supply_chain_data.csv
   • eda_overview.png
   • correlation_matrix.png
   • regression_model.png
   • classification_model.png
   • clustering_model.png

📊 KEY RESULTS:
   • Dataset: {len(df):,} orders analyzed
   • Statistical Tests: 4 hypothesis tests completed
   • ML Models: 3 models built and evaluated
   • Visualizations: 5 professional charts created

🎯 NEXT STEPS:
   1. Review all visualizations
   2. Prepare presentation slides
   3. Practice 3-4 minute pitch
   4. Submit deliverables to mentor

💡 Your capstone project is complete!
""")
print("="*80)
"""
================================================================================
INVENTORY FLOW & SUPPLY CHAIN TRENDS - COMPLETE ANALYSIS
================================================================================
Capstone Project: Statistical Methods and Essential Machine Learning Models
Student: [Your Name]
Date: February 9, 2026

This script performs:
1. Data loading and initial exploration
2. Data cleaning and feature engineering
3. Exploratory data analysis (EDA)
4. Statistical hypothesis testing (t-test, ANOVA, Chi-Square, Correlation)
5. Machine learning modeling (Regression, Classification, Clustering)
6. Visualization and reporting

Requirements:
pip install pandas numpy scipy scikit-learn matplotlib seaborn
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, classification_report, confusion_matrix,
                             silhouette_score)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("INVENTORY FLOW & SUPPLY CHAIN TRENDS - COMPLETE ANALYSIS")
print("="*80)

# ============================================================================
# PART 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n[PART 1] DATA LOADING AND INITIAL EXPLORATION")
print("-" * 80)

# Load your dataset
# UPDATE THIS PATH to your actual dataset location
df_raw = pd.read_csv('supply_chain_dataset.csv')

print(f"\n✓ Dataset loaded successfully!")
print(f"  Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

print(f"\n📋 Column Names:")
for i, col in enumerate(df_raw.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\n🔍 First 5 Rows:")
print(df_raw.head())

print(f"\n📊 Data Types:")
print(df_raw.dtypes)

print(f"\n📈 Descriptive Statistics:")
print(df_raw.describe())

print(f"\n⚠️  Missing Values:")
missing = df_raw.isnull().sum()
if missing.sum() > 0:
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': (missing.values / len(df_raw)) * 100
    })
    print(missing_df[missing_df['Missing_Count'] > 0].to_string(index=False))
else:
    print("  No missing values found!")

print(f"\n🔄 Duplicate Rows: {df_raw.duplicated().sum()}")

# ============================================================================
# PART 2: DATA CLEANING AND FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("[PART 2] DATA CLEANING AND FEATURE ENGINEERING")
print("-" * 80)

# Create a copy for cleaning
df = df_raw.copy()

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"\n✓ Removed {initial_rows - len(df)} duplicate rows")

# Convert date columns if they exist
if 'Order_Date' in df.columns and 'Shipment_Date' in df.columns:
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Shipment_Date'] = pd.to_datetime(df['Shipment_Date'])
    print("✓ Converted date columns to datetime")

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute numeric columns with median
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✓ Imputed {col} with median: {median_val:.2f}")

# Impute categorical columns with mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"✓ Imputed {col} with mode: {mode_val}")

# Feature Engineering - Create new calculated features
print("\n🔧 Creating Engineered Features:")

if 'Order_Date' in df.columns and 'Shipment_Date' in df.columns:
    df['Actual_Lead_Time'] = (df['Shipment_Date'] - df['Order_Date']).dt.days
    print("  ✓ Actual_Lead_Time (days between order and shipment)")

if 'Actual_Lead_Time' in df.columns and 'Lead_Time' in df.columns:
    df['Lead_Time_Variance'] = df['Actual_Lead_Time'] - df['Lead_Time']
    print("  ✓ Lead_Time_Variance (actual vs planned)")

if 'Demand_Forecast' in df.columns and 'Inventory_Level' in df.columns:
    df['Demand_vs_Inventory'] = df['Demand_Forecast'] - df['Inventory_Level']
    print("  ✓ Demand_vs_Inventory (forecast minus stock)")

if 'Shipment_Quantity' in df.columns and 'Order_Quantity' in df.columns:
    df['Fulfillment_Rate'] = (df['Shipment_Quantity'] / df['Order_Quantity']) * 100
    print("  ✓ Fulfillment_Rate (% of order fulfilled)")

if 'Inventory_Level' in df.columns and 'Demand_Forecast' in df.columns and 'Stockout_Flag' in df.columns:
    df['Stockout_Risk'] = ((df['Inventory_Level'] < df['Demand_Forecast']) & (df['Stockout_Flag'] == 0)).astype(int)
    print("  ✓ Stockout_Risk (predictive flag)")

print(f"\n✓ Final dataset: {len(df):,} rows × {len(df.columns)} columns")

# Save cleaned data
df.to_csv('cleaned_supply_chain_data.csv', index=False)
print("✓ Saved: cleaned_supply_chain_data.csv")

# ============================================================================
# PART 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("[PART 3] EXPLORATORY DATA ANALYSIS")
print("-" * 80)

# Key Performance Indicators
print("\n📊 SUPPLY CHAIN KEY PERFORMANCE INDICATORS:")
print("-" * 80)

if 'Stockout_Flag' in df.columns:
    stockout_count = df['Stockout_Flag'].sum()
    print(f"Total Orders:              {len(df):,}")
    print(f"Stockout Orders:           {stockout_count:,} ({(stockout_count/len(df)*100):.1f}%)")

if 'Backorder_Flag' in df.columns:
    backorder_count = df['Backorder_Flag'].sum()
    print(f"Backorder Orders:          {backorder_count:,} ({(backorder_count/len(df)*100):.1f}%)")

if 'Order_Priority' in df.columns:
    high_priority = (df['Order_Priority'] == 'High').sum()
    print(f"High Priority Orders:      {high_priority:,} ({(high_priority/len(df)*100):.1f}%)")

if 'Lead_Time' in df.columns:
    print(f"\nAverage Lead Time:         {df['Lead_Time'].mean():.2f} days")

if 'Fulfillment_Rate' in df.columns:
    print(f"Average Fulfillment Rate:  {df['Fulfillment_Rate'].mean():.1f}%")

if 'Order_Quantity' in df.columns:
    print(f"Average Order Quantity:    {df['Order_Quantity'].mean():.0f} units")

if 'Inventory_Level' in df.columns:
    print(f"Average Inventory Level:   {df['Inventory_Level'].mean():.0f} units")

# Performance by Warehouse
if 'Warehouse' in df.columns:
    print("\n📈 PERFORMANCE BY WAREHOUSE:")
    print("-" * 80)
    warehouse_cols = ['Order_Quantity', 'Stockout_Flag', 'Lead_Time']
    available_cols = [col for col in warehouse_cols if col in df.columns]
    if available_cols:
        warehouse_stats = df.groupby('Warehouse')[available_cols].agg({
            available_cols[0]: 'sum' if available_cols[0] == 'Order_Quantity' else 'mean',
            **{col: 'sum' if 'Flag' in col else 'mean' for col in available_cols[1:]}
        }).round(2)
        print(warehouse_stats)

# Performance by Product Category
if 'Product_Category' in df.columns:
    print("\n📦 PERFORMANCE BY PRODUCT CATEGORY:")
    print("-" * 80)
    category_cols = ['Order_Quantity', 'Stockout_Flag', 'Product_Price']
    available_cols = [col for col in category_cols if col in df.columns]
    if available_cols:
        category_stats = df.groupby('Product_Category')[available_cols].agg('mean').round(2)
        print(category_stats)

# Visualizations
print("\n📊 Creating EDA Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis - Supply Chain Dataset', fontsize=16, fontweight='bold')

# Plot 1: Distribution of Order Quantity
if 'Order_Quantity' in df.columns:
    axes[0, 0].hist(df['Order_Quantity'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(df['Order_Quantity'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_title('Distribution of Order Quantity')
    axes[0, 0].set_xlabel('Order Quantity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()

# Plot 2: Box plot of Lead Time
if 'Lead_Time' in df.columns:
    axes[0, 1].boxplot(df['Lead_Time'])
    axes[0, 1].set_title('Lead Time Distribution')
    axes[0, 1].set_ylabel('Days')

# Plot 3: Bar chart of categories
if 'Product_Category' in df.columns:
    cat_counts = df['Product_Category'].value_counts()
    axes[1, 0].bar(range(len(cat_counts)), cat_counts.values, color='green', alpha=0.7)
    axes[1, 0].set_xticks(range(len(cat_counts)))
    axes[1, 0].set_xticklabels(cat_counts.index, rotation=45, ha='right')
    axes[1, 0].set_title('Orders by Product Category')
    axes[1, 0].set_ylabel('Count')

# Plot 4: Scatter plot
if 'Demand_Forecast' in df.columns and 'Inventory_Level' in df.columns:
    axes[1, 1].scatter(df['Demand_Forecast'], df['Inventory_Level'], alpha=0.5, color='purple')
    axes[1, 1].set_xlabel('Demand Forecast')
    axes[1, 1].set_ylabel('Inventory Level')
    axes[1, 1].set_title('Demand vs Inventory')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_overview.png")

# Correlation Matrix
print("\n📊 Creating Correlation Matrix...")
numeric_cols_for_corr = [col for col in numeric_cols if col in df.columns][:10]  # First 10 numeric columns
if len(numeric_cols_for_corr) >= 3:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_cols_for_corr].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")

# ============================================================================
# PART 4: STATISTICAL HYPOTHESIS TESTING
# ============================================================================
print("\n" + "="*80)
print("[PART 4] STATISTICAL HYPOTHESIS TESTING")
print("="*80)

# Test 1: Independent t-test
if 'Lead_Time' in df.columns and 'Stockout_Flag' in df.columns:
    print("\n[TEST 1] Independent t-test: Lead Time vs Stockout")
    print("-" * 80)
    
    stockout_lead = df[df['Stockout_Flag'] > 0]['Lead_Time']
    no_stockout_lead = df[df['Stockout_Flag'] == 0]['Lead_Time']
    
    t_stat, p_value_t = ttest_ind(stockout_lead, no_stockout_lead)
    
    print(f"H0: Mean lead time is SAME for stockout and non-stockout orders")
    print(f"H1: Mean lead time is DIFFERENT")
    print(f"\nStockout orders - Mean: {stockout_lead.mean():.2f} days (n={len(stockout_lead)})")
    print(f"Non-stockout orders - Mean: {no_stockout_lead.mean():.2f} days (n={len(no_stockout_lead)})")
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value_t:.4f}")
    print(f"\nConclusion: {'✓ REJECT H0' if p_value_t < 0.05 else '✗ FAIL TO REJECT H0'} (α=0.05)")

# Test 2: ANOVA
if 'Warehouse' in df.columns and 'Inventory_Level' in df.columns:
    print("\n[TEST 2] ANOVA: Inventory Level Across Warehouses")
    print("-" * 80)
    
    warehouse_groups = [df[df['Warehouse'] == wh]['Inventory_Level'] for wh in df['Warehouse'].unique()]
    f_stat, p_value_anova = f_oneway(*warehouse_groups)
    
    print(f"H0: Mean inventory level is SAME across all warehouses")
    print(f"H1: At least one warehouse has different mean")
    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value_anova:.4f}")
    print(f"\nConclusion: {'✓ REJECT H0' if p_value_anova < 0.05 else '✗ FAIL TO REJECT H0'} (α=0.05)")

# Test 3: Chi-Square Test
if 'Order_Priority' in df.columns and 'Backorder_Flag' in df.columns:
    print("\n[TEST 3] Chi-Square Test: Order Priority vs Backorder")
    print("-" * 80)
    
    contingency_table = pd.crosstab(df['Order_Priority'], df['Backorder_Flag'])
    chi2, p_value_chi, dof, expected = chi2_contingency(contingency_table)
    
    print(f"H0: Order priority and backorder are INDEPENDENT")
    print(f"H1: They are DEPENDENT")
    print(f"\nContingency Table:")
    print(contingency_table)
    print(f"\nChi-square: {chi2:.4f}")
    print(f"p-value: {p_value_chi:.4f}")
    print(f"\nConclusion: {'✓ REJECT H0' if p_value_chi < 0.05 else '✗ FAIL TO REJECT H0'} (α=0.05)")

# Test 4: Pearson Correlation
if 'Demand_Forecast' in df.columns and 'Inventory_Level' in df.columns:
    print("\n[TEST 4] Pearson Correlation: Demand vs Inventory")
    print("-" * 80)
    
    corr_coef, p_value_corr = pearsonr(df['Demand_Forecast'], df['Inventory_Level'])
    
    print(f"H0: No correlation (ρ = 0)")
    print(f"H1: Correlation exists (ρ ≠ 0)")
    print(f"\nCorrelation coefficient (r): {corr_coef:.4f}")
    print(f"p-value: {p_value_corr:.4f}")
    print(f"\nConclusion: {'✓ REJECT H0' if p_value_corr < 0.05 else '✗ FAIL TO REJECT H0'} (α=0.05)")

# ============================================================================
# PART 5: MACHINE LEARNING MODELS
# ============================================================================
print("\n" + "="*80)
print("[PART 5] MACHINE LEARNING MODELS")
print("="*80)

# Encode categorical variables
if 'Warehouse' in df.columns:
    le_warehouse = LabelEncoder()
    df['Warehouse_encoded'] = le_warehouse.fit_transform(df['Warehouse'])

if 'Product_Category' in df.columns:
    le_category = LabelEncoder()
    df['Category_encoded'] = le_category.fit_transform(df['Product_Category'])

if 'Order_Priority' in df.columns:
    le_priority = LabelEncoder()
    df['Priority_encoded'] = le_priority.fit_transform(df['Order_Priority'])

# MODEL 1: LINEAR REGRESSION
print("\n[MODEL 1] Linear Regression: Predicting Order Quantity")
print("-" * 80)

if 'Order_Quantity' in df.columns:
    # Select features for regression
    feature_candidates = ['Demand_Forecast', 'Inventory_Level', 'Lead_Time', 
                         'Product_Price', 'Warehouse_encoded', 'Category_encoded']
    features_reg = [f for f in feature_candidates if f in df.columns]
    
    if len(features_reg) >= 2:
        X_reg = df[features_reg]
        y_reg = df['Order_Quantity']
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
        X_test_reg_scaled = scaler_reg.transform(X_test_reg)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_reg_scaled, y_train_reg)
        y_pred_reg = lr_model.predict(X_test_reg_scaled)
        
        r2 = r2_score(y_test_reg, y_pred_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        
        print(f"Features: {', '.join(features_reg)}")
        print(f"\nModel Performance:")
        print(f"  R² Score:  {r2:.4f}")
        print(f"  RMSE:      {rmse:.2f}")
        print(f"  MAE:       {mae:.2f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(y_test_reg, y_pred_reg, alpha=0.5, color='steelblue')
        axes[0].plot([y_test_reg.min(), y_test_reg.max()], 
                    [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Order Quantity')
        axes[0].set_ylabel('Predicted Order Quantity')
        axes[0].set_title(f'Regression: Actual vs Predicted\nR² = {r2:.4f}')
        axes[0].grid(True, alpha=0.3)
        
        residuals = y_test_reg - y_pred_reg
        axes[1].scatter(y_pred_reg, residuals, alpha=0.5, color='green')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Order Quantity')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_model.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: regression_model.png")

# MODEL 2: LOGISTIC REGRESSION
print("\n[MODEL 2] Logistic Regression: Predicting Stockout")
print("-" * 80)

if 'Stockout_Flag' in df.columns:
    feature_candidates = ['Demand_Forecast', 'Inventory_Level', 'Lead_Time',
                         'Order_Quantity', 'Warehouse_encoded', 'Priority_encoded']
    features_class = [f for f in feature_candidates if f in df.columns]
    
    if len(features_class) >= 2:
        X_class = df[features_class]
        y_class = df['Stockout_Flag']
        
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.3, random_state=42, stratify=y_class
        )
        
        scaler_class = StandardScaler()
        X_train_class_scaled = scaler_class.fit_transform(X_train_class)
        X_test_class_scaled = scaler_class.transform(X_test_class)
        
        log_model = LogisticRegression(random_state=42, max_iter=1000)
        log_model.fit(X_train_class_scaled, y_train_class)
        y_pred_class = log_model.predict(X_test_class_scaled)
        
        accuracy = accuracy_score(y_test_class, y_pred_class)
        conf_matrix = confusion_matrix(y_test_class, y_pred_class)
        
        print(f"Features: {', '.join(features_class)}")
        print(f"\nModel Performance:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"\nClassification Report:")
        print(classification_report(y_test_class, y_pred_class))
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        feature_importance = pd.DataFrame({
            'Feature': features_class,
            'Importance': np.abs(log_model.coef_[0])
        }).sort_values('Importance', ascending=True)
        
        axes[1].barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
        axes[1].set_xlabel('Absolute Coefficient Value')
        axes[1].set_title('Feature Importance')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('classification_model.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: classification_model.png")

# MODEL 3: K-MEANS CLUSTERING
print("\n[MODEL 3] K-Means Clustering: Order Segmentation")
print("-" * 80)

feature_candidates = ['Demand_Forecast', 'Inventory_Level', 'Order_Quantity',
                     'Lead_Time', 'Product_Price']
features_cluster = [f for f in feature_candidates if f in df.columns]

if len(features_cluster) >= 3:
    X_cluster = df[features_cluster].copy()
    
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    # Find optimal k
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_cluster_scaled)
    
    df['Cluster'] = cluster_labels
    
    print(f"Features: {', '.join(features_cluster)}")
    print(f"Optimal clusters: {optimal_k}")
    print(f"Silhouette Score: {silhouette_score(X_cluster_scaled, cluster_labels):.4f}")
    
    print(f"\nCluster Characteristics:")
    for i in range(optimal_k):
        cluster_data = df[df['Cluster'] == i]
        print(f"\nCluster {i} ({len(cluster_data):,} orders - {len(cluster_data)/len(df)*100:.1f}%):")
        for feature in features_cluster:
            print(f"  Avg {feature}: {cluster_data[feature].mean():.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Elbow curve
    axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Silhouette scores
    axes[0, 1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title(f'Silhouette Analysis (Optimal k={optimal_k})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cluster scatter plots
    if len(features_cluster) >= 2:
        scatter1 = axes[1, 0].scatter(df[features_cluster[0]], df[features_cluster[1]], 
                                       c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
        axes[1, 0].set_xlabel(features_cluster[0])
        axes[1, 0].set_ylabel(features_cluster[1])
        axes[1, 0].set_title(f'Clusters (k={optimal_k})')
        plt.colorbar(scatter1, ax=axes[1, 0], label='Cluster')
        
    if len(features_cluster) >= 4:
        scatter2 = axes[1, 1].scatter(df[features_cluster[2]], df[features_cluster[3]], 
                                       c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
        axes[1, 1].set_xlabel(features_cluster[2])
        axes[1, 1].set_ylabel(features_cluster[3])
        axes[1, 1].set_title(f'Clusters (k={optimal_k})')
        plt.colorbar(scatter2, ax=axes[1, 1], label='Cluster')
    
    plt.tight_layout()
    plt.savefig('clustering_model.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: clustering_model.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"""
✅ FILES CREATED:
   • cleaned_supply_chain_data.csv
   • eda_overview.png
   • correlation_matrix.png
   • regression_model.png
   • classification_model.png
   • clustering_model.png

📊 KEY RESULTS:
   • Dataset: {len(df):,} orders analyzed
   • Statistical Tests: 4 hypothesis tests completed
   • ML Models: 3 models built and evaluated
   • Visualizations: 5 professional charts created

🎯 NEXT STEPS:
   1. Review all visualizations
   2. Prepare presentation slides
   3. Practice 3-4 minute pitch
   4. Submit deliverables to mentor

💡 Your capstone project is complete!
""")
print("="*80)