import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self, df):
        """
        Initializes the DataAnalyzer class.
        
        :param df: The pandas DataFrame containing the data.
        """
        self.df = df

    def univariate_analysis(self, column, plot_type='hist', bins=10, color='coolwarm', hue=None):
        """
        Performs univariate analysis on the specified column.
        
        :param column: Column name for univariate analysis.
        :param plot_type: Type of plot ('hist', 'box', 'count', 'violin', 'kde', 'pie').
        :param bins: Number of bins for histogram (if applicable).
        :param color: Color or palette for the plot.
        :param hue: Categorical variable for grouping data by color.
        :param color: Color or palette for the plot. Choose from:
            - 'coolwarm': A gradient of cool (blue) to warm (red).
            - 'viridis': Perceptually uniform colormap, great for continuous data.
            - 'plasma': Bright and vibrant colormap.
            - 'magma': Dark and visually striking colormap.
            - 'cividis': Colorblind-friendly alternative to viridis.
            - 'cubehelix': Linearly increasing brightness, customizable.
            - 'rocket': Subtle dark-to-light colormap.
            - 'flare': Bright, diverging colormap.
            - 'crest': Smooth gradient for elegant plots.
            - 'icefire': Contrasting cool (blue) and warm (orange) colors.
            - 'Others: You can use color like blue,green,red etc.
        """
        plt.figure(figsize=(8, 6))
        plt.title(f"Univariate Analysis of {column}", fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)

        if self.df[column].dtype in ['int64', 'float64']:  # Numerical data
            if plot_type == 'hist':
                sns.histplot(self.df, x=column, bins=bins, hue=hue, palette=color, kde=True)
            elif plot_type == 'box':
                sns.boxplot(x=self.df[column], hue=hue, palette=color)
            elif plot_type == 'violin':
                sns.violinplot(x=self.df[column], hue=hue, palette=color)
            elif plot_type == 'kde':
                sns.kdeplot(data=self.df, x=column, hue=hue, fill=True, palette=color)
            else:
                print(f"Unsupported plot type for numerical data: {plot_type}")
        else:  # Categorical data
            if plot_type == 'count':
                sns.countplot(x=self.df[column], hue=hue, palette=color)
            elif plot_type == 'pie':
                print("Pie chart does not support hue.")
            else:
                print(f"Unsupported plot type for categorical data: {plot_type}")
        
        plt.show()

    def bivariate_analysis(self, column1, column2, plot_type='scatter', color='coolwarm', hue=None):
        """
        Performs bivariate analysis between two columns.
        
        :param column1: First column name.
        :param column2: Second column name.
        :param plot_type: Type of plot ('scatter', 'box', 'bar', 'line', 'heatmap', 'pairplot').
        :param color: Color or palette for the plot.
        :param hue: Categorical variable for grouping data by color.
        :param color: Color or palette for the plot. Choose from:
            - 'coolwarm': A gradient of cool (blue) to warm (red).
            - 'viridis': Perceptually uniform colormap, great for continuous data.
            - 'plasma': Bright and vibrant colormap.
            - 'magma': Dark and visually striking colormap.
            - 'cividis': Colorblind-friendly alternative to viridis.
            - 'cubehelix': Linearly increasing brightness, customizable.
            - 'rocket': Subtle dark-to-light colormap.
            - 'flare': Bright, diverging colormap.
            - 'crest': Smooth gradient for elegant plots.
            - 'icefire': Contrasting cool (blue) and warm (orange) colors.
            - 'Others: You can use color like blue,green,red etc.
        """
        plt.figure(figsize=(8, 6))
        plt.title(f"Bivariate Analysis of {column1} vs {column2}", fontsize=14)
        plt.xlabel(column1, fontsize=12)
        plt.ylabel(column2, fontsize=12)

        if plot_type == 'scatter':  # Scatter plot for numerical vs numerical
            if self.df[column1].dtype in ['int64', 'float64'] and self.df[column2].dtype in ['int64', 'float64']:
                sns.scatterplot(x=self.df[column1], y=self.df[column2], hue=self.df[hue], palette=color)
            else:
                print("Scatter plot requires both columns to be numerical.")
        elif plot_type == 'box':  # Box plot for categorical vs numerical
            if self.df[column1].dtype == 'object' and self.df[column2].dtype in ['int64', 'float64']:
                sns.boxplot(x=self.df[column1], y=self.df[column2], hue=hue, palette=color)
            elif self.df[column2].dtype == 'object' and self.df[column1].dtype in ['int64', 'float64']:
                sns.boxplot(x=self.df[column2], y=self.df[column1], hue=hue, palette=color)
            else:
                print("Box plot requires one numerical and one categorical column.")
        elif plot_type == 'bar':  # Bar plot for categorical vs numerical
            if self.df[column1].dtype == 'object' and self.df[column2].dtype in ['int64', 'float64']:
                sns.barplot(x=self.df[column1], y=self.df[column2], hue=hue, palette=color)
            elif self.df[column2].dtype == 'object' and self.df[column1].dtype in ['int64', 'float64']:
                sns.barplot(x=self.df[column2], y=self.df[column1], hue=hue, palette=color)
            else:
                print("Bar plot requires one numerical and one categorical column.")
        elif plot_type == 'line':  # Line plot for numerical vs numerical
            if self.df[column1].dtype in ['int64', 'float64'] and self.df[column2].dtype in ['int64', 'float64']:
                sns.lineplot(x=self.df[column1], y=self.df[column2], hue=hue, palette=color)
            else:
                print("Line plot requires both columns to be numerical.")
        elif plot_type == 'heatmap':  # Correlation heatmap
            print("Heatmap does not support hue.")
        elif plot_type == 'pairplot':  # Pairplot for a subset of columns
            sns.pairplot(self.df, vars=[column1, column2], hue=hue, palette=color, corner=True, diag_kind="kde")
        else:
            print(f"Unsupported plot type: {plot_type}")
        
        plt.show()
