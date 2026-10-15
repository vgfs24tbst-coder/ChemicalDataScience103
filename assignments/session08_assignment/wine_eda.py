"""
wine_eda.py
-----------
Homework: Exploratory Data Analysis on the Wine dataset.

Goals:
1. Load and inspect the data
2. Perform exploratory analysis
3. Produce some plots for visualisations
4. Apply PCA
5. Apply some simple ML models for classification
6. Summarise results
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# Main analysis function
def main():
    # --- 1. Load data ---
    print("Loading dataset...")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    # add a 'target' column
    df.insert(0, "target", wine.target)


    # print basic info (shape, first rows, summary statistics)
    print(df.info())
    print(df.shape)
    print(df.columns)
    print(df.head(10))
    print(df.describe())

    # --- 2. Basic exploration ---
    # print class distribution and check for missing values as well as duplicates
    print(f"Class distribution : {df.nunique(axis=0)}")

    print(f"Missing values: {df.isna().sum()}")

    if df.duplicated().sum() == 0:
        print("No duplicates")
    else: 
        print(f"Nr of duplicates: {df.duplicated().sum()}")
        print(df[df.duplicated()])

    
    # --- 3. Visualisations ---
    # create and save a pairplot using a few selected features
    features= ["alcohol", "magnesium", "malic_acid", "proline" ]
    sns.pairplot(df[features], diag_kind='kde')
    plt.suptitle('Pairplot of 4 Features in Wine set', y=1.02)
    plt.savefig(fname="pairplot")
    # create and save a boxplot comparing one feature across classes
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x = "target", y = df["magnesium"])
    plt.title('Boxplot of magesium Features across all wine classes')
    plt.savefig(fname="boxplot")
    # plot and save a heatmap (e.g. 'heatmap.png')
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Wine Dataset Correlation Heatmap')
    plt.savefig(fname="heatmap")
    

    # --- 4. Scaling and PCA ---
    # separate features (X) and target (y)
    X = df.drop(["target"], axis=1)
    y = df["target"]
    # scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # apply PCA (2 components)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # create a DataFrame with PCA results and target
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["target"] = y
    # plot and save a scatterplot of the first two components (color by target)
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="target", s=80)
    plt.title("PCA Visualization")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.show()

    # --- 5. Classification models ---
    # Do a random train-test split (20%) 
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale training and test features (fit_transform() on X_train, transform() on X_test)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression, a decision tree and a random forest (default hyperparameters)
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train_scaled, y_train)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)

    rf = RandomForestClassifier(max_depth=3, random_state=42)
    rf.fit(X_train, y_train)


    # Compare the label prediction accuracy of the different models.

    summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree (depth=3)", "RandomForest (Depth=3)"],
    "Train Accuracy": [logreg.score(X_train_scaled, y_train), tree.score(X_train, y_train), rf.score(X_train, y_train)],
    "Test Accuracy": [logreg.score(X_test_scaled, y_test), tree.score(X_test, y_test), rf.score(X_test, y_test)]})
    print(summary)
    # Train and evaluate the three models on the PCA data.
    lg = DecisionTreeClassifier(max_depth=3, random_state=42)
    X_pca_full_scaled = PCA(n_components=2).fit_transform(X_scaled)
    lg.fit(X_pca_full_scaled, y)
    print("LogReg accuracy on 2D PCA features:", cross_val_score(lg, X_pca_full_scaled, y, cv=5).mean())

    tree_pca = DecisionTreeClassifier(max_depth=3, random_state=42)
    X_pca_full = PCA(n_components=2).fit_transform(X)
    tree_pca.fit(X_pca_full, y)
    print("Decision Tree accuracy on 2D PCA features:", cross_val_score(tree_pca, X_pca_full, y, cv=5).mean())

    rf_pca = RandomForestClassifier(max_depth=3, random_state=42)
    rf_pca.fit(X_pca_full, y)
    print("Random Forest accuracy on 2D PCA features:", cross_val_score(rf_pca, X_pca_full, y, cv=5).mean())


# Entry point
if __name__ == "__main__":
    main()