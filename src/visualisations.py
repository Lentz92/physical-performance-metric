import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def bland_altman_plot(data1, data2, plottitle=None, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.figure(figsize=(12, 8))
    plt.title(plottitle, fontsize=16, fontweight='bold')
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.xlabel('Average of two measures', fontsize=14, fontweight='bold')
    plt.ylabel('Difference between two measures', fontsize=14, fontweight='bold')
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred):
    """
    Plots actual vs predicted values along with a trend line and an ideal fit line.

    Args:
        y_test (array-like): Array of actual values.
        y_pred (array-like): Array of predicted values.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='cornflowerblue', label='Actual vs Predicted Values')

    # Fit a line to the data
    coefficients = np.polyfit(y_test, y_pred, 1)
    trend_line = np.poly1d(coefficients)

    # Plot the trend line
    plt.plot(y_test, trend_line(y_test), color='darkblue', linestyle='--', lw=2, label='Trend Line')

    # Plot the ideal fit line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2,
             label='Ideal Fit')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_performance_heatmap(df, year, team, save=False):
    """
    Plots a performance heatmap for a specified team and year.

    This function generates a heatmap for each position within the specified team and year,
    displaying the performance scores of players across different weeks. The heatmap uses a custom
    colormap to represent different performance levels.

    Args:
        df (pd.DataFrame): DataFrame containing the performance data with columns 'year', 'team_name', 'position', 'player', 'week', and 'performance_score'.
        year (int): The year for which to filter the data.
        team (str): The team name for which to filter the data.
        save (bool, optional): If True, saves the generated heatmaps as PNG files. Default is False.
    """
    # Define a custom colormap
    colors = [
        (0.00, "darkred"),   # Poor performance starts with dark red
        (0.20, "red"),       # Transitions to bright red
        (0.40, "grey"),      # Reasonable performance in grey
        (0.60, "grey"),      # Reasonable performance in grey
        (0.65, "limegreen"), # Good performance in brighter green
        (0.80, "green"),     # Better performance in standard green
        (1.00, "darkgreen")  # Excellent performance in darkest green
    ]


    cmap = LinearSegmentedColormap.from_list("custom_red_yellow_green", colors, N=256)

    # Filter data based on year and team
    df_filtered = df[(df['year'] == year) & (df['team_name'] == team)]

    # Iterate through each position and create a separate plot
    positions = df_filtered['position'].unique()
    sns.set(style="white")
    for position in positions:
        # Prepare data for the heatmap
        position_data = df_filtered[df_filtered['position'] == position].sort_values(by='player')
        performance_matrix = position_data.pivot_table(index='player', columns='week', values='performance_score', aggfunc='mean')

        # Plotting heatmap with the custom colormap and ensuring square cells
        plt.figure(figsize=(15, 12))  # Set figure size
        ax = sns.heatmap(performance_matrix,
                         annot=True,
                         cmap=cmap,
                         fmt=".0f",
                         cbar=True,
                         annot_kws={'color': 'black'},  # Ensure text color is black
                         square=True,
                         cbar_kws={
                             'orientation': 'horizontal',
                             'pad': 0.075,
                             'aspect': 70},
                         vmin=0, vmax=100)  # Make cells square
        ax.set_title(f'Performance Heatmap: \n{position} - {team} - {year}')
        ax.set_ylabel('Player')
        ax.set_xlabel('Week')

        # Save the plot if the save option is enabled
        if save:
            plt.savefig(f'../visualisations/{position} - {team} - {year}.png')

        plt.show()