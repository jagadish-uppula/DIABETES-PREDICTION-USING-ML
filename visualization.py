import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distribution(data, feature, target='Outcome'):
    """
    Plot distribution of a feature by target variable
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x=feature, hue=target, kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature} by Diabetes Outcome')
    return fig

def plot_correlation_matrix(data):
    """
    Plot correlation heatmap
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    return fig

def plot_feature_importance(features, importances):
    """
    Plot feature importance
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=features, palette='Blues_d', ax=ax)
    ax.set_title('Feature Importance in Diabetes Prediction')
    return fig