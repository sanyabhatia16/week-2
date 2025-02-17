import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

def numerical_data():
    # Generate random numerical data
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    y = np.random.normal(loc=0, scale=1, size=x.shape) + np.sin(x)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.6)
    plt.title('Numerical Data Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.savefig('numerical_data.png')
    plt.close()

def categorical_data():
    # Sample categorical data
    categories = ['A', 'B', 'C', 'D']
    values = [3, 7, 5, 2]

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=categories, y=values, palette='viridis')
    plt.title('Categorical Data Visualization')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.savefig('categorical_data.png')
    plt.close()

def generalization():
    # Sample data for generalization
    x = np.linspace(0, 10, 100)
    y_true = np.sin(x)
    y_pred = np.sin(x) + np.random.normal(0, 0.1, x.shape)

    # Create a line plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_true, label='True Function', color='green')
    plt.scatter(x, y_pred, label='Predicted Data', color='red', alpha=0.5)
    plt.title('Generalization Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.savefig('generalization.png')
    plt.close()

def overfitting():
    # Sample data for overfitting
    x = np.linspace(0, 10, 10)
    y = np.sin(x)
    y_noisy = y + np.random.normal(0, 0.5, x.shape)

    # Fit a polynomial of degree 9 (overfitting)
    coeffs = np.polyfit(x, y_noisy, 9)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(0, 10, 100)
    y_fit = poly(x_fit)

    # Create a plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y_noisy, label='Noisy Data', color='red')
    plt.plot(x_fit, y_fit, label='Overfitted Polynomial', color='blue')
    plt.title('Overfitting Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.savefig('overfitting.png')
    plt.close()

if __name__ == "__main__":
    numerical_data()
    categorical_data()
    generalization()
    overfitting()