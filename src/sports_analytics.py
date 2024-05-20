import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class SportsAnalytics:
    def __init__(self, data):
        self.data = data

    def identify_high_risk_instances(self, thresholds):
        # Calculate the sum of acc_3m_s_s_total_efforts and dec_3m_s_s_total_efforts
        self.data = self.data.with_columns(
            (pl.col("acc_3m_s_s_total_efforts") + pl.col("dec_3m_s_s_total_efforts")).alias("total_acc_dec_efforts")
        )

        # Identify high-risk instances
        self.data = self.data.with_columns([
            (pl.col("total_player_load") > thresholds["total_player_load"]).alias("high_risk_load"),
            (pl.col("high_intensity_distance_m_v5_v6_m") > thresholds["high_intensity_distance_m_v5_v6_m"]).alias("high_risk_velocity"),
            (pl.col("total_acc_dec_efforts") > thresholds["total_acc_dec_efforts"]).alias("high_risk_acceleration")
        ])
        return self.data

    def filter_by_date(self, start_date, end_date):
        self.data = self.data.filter(
            (pl.col("date") >= pl.lit(start_date).str.strptime(pl.Date)) &
            (pl.col("date") <= pl.lit(end_date).str.strptime(pl.Date))
        )
        return self

    def get_date_range(self):
        min_date = self.data.select(pl.col("date").min()).item()
        max_date = self.data.select(pl.col("date").max()).item()
        return min_date, max_date

    def plot_high_risk_summary(self, summary_df, title, date_range=None):
        fig, ax = plt.subplots(figsize=(14, 8))
        bottom = np.zeros(len(summary_df))

        for metric, color, label in zip(['high_risk_load_count', 'high_risk_velocity_count', 'high_risk_acceleration_count'],
                                        ['blue', 'red', 'green'],
                                        ['High Risk Load', 'High Risk Velocity', 'High Risk Acceleration']):
            ax.bar(summary_df['player'], summary_df[metric], color=color, alpha=0.6, label=label, bottom=bottom)
            bottom += summary_df[metric]

        ax.set_xlabel('Player')
        ax.set_ylabel('Count of High-Risk Instances')
        ax.set_title(title)

        if date_range:
            subtitle = f'Date Range: {date_range[0]} to {date_range[1]}'
            plt.suptitle(subtitle, y=0.9, fontsize=12, color='gray')

        ax.legend()
        ax.grid(True, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def create_summary_table(self, summary_df):
        summary_df.set_index('player', inplace=True)
        print(summary_df)

    def additional_analytics(self, summary_df):
        pass


class TeamAnalytics(SportsAnalytics):
    def __init__(self, data):
        super().__init__(data)

    def analyze_team(self, team_name):
        team_data = self.data.filter(pl.col("team_name") == team_name)
        team_data = team_data.sort(["position", "player"])
        return team_data

    def plot_team(self, team_name, start_date=None, end_date=None):
        date_range = None
        if start_date and end_date:
            self.filter_by_date(start_date, end_date)
            date_range = self.get_date_range()

        team_data = self.analyze_team(team_name)
        high_risk_summary = team_data.groupby(["position", "player"]).agg([
            pl.col("high_risk_load").sum().alias("high_risk_load_count"),
            pl.col("high_risk_velocity").sum().alias("high_risk_velocity_count"),
            pl.col("high_risk_acceleration").sum().alias("high_risk_acceleration_count")
        ])
        high_risk_summary_df = high_risk_summary.to_pandas()

        fig, ax = plt.subplots(figsize=(14, 8))
        bottom = np.zeros(len(high_risk_summary_df))

        # Plot each metric with different colors
        for metric, color, label in zip(['high_risk_load_count', 'high_risk_velocity_count', 'high_risk_acceleration_count'],
                                        ['blue', 'red', 'green'],
                                        ['High Risk Load', 'High Risk Velocity', 'High Risk Acceleration']):
            ax.bar(high_risk_summary_df['player'], high_risk_summary_df[metric], color=color, alpha=0.6, label=label, bottom=bottom)
            bottom += high_risk_summary_df[metric]

        # Add separators and labels for positions
        unique_positions = sorted(high_risk_summary_df['position'].unique())
        current_pos = 0
        for pos in unique_positions:
            pos_count = sum(high_risk_summary_df['position'] == pos)
            ax.axvline(x=current_pos + pos_count - 0.5, color='black', linewidth=3)
            ax.text(current_pos + pos_count / 2 - 0.5, max(bottom) + 1, pos, ha='center', va='bottom', fontsize=12, weight='bold')
            current_pos += pos_count

        ax.set_xlabel('Player')
        ax.set_ylabel('Count of High-Risk Instances')
        ax.set_title(f'High-Risk Instances for {team_name}')

        if date_range:
            subtitle = f'Date Range: {date_range[0]} to {date_range[1]}'
            plt.suptitle(subtitle, y=0.9, fontsize=12, color='gray')

        ax.legend()
        ax.grid(True, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class PositionAnalytics(SportsAnalytics):
    def __init__(self, data):
        super().__init__(data)

    def analyze_position(self, position):
        position_data = self.data.filter(pl.col("position") == position)
        position_data = position_data.sort(["team_name", "player"])
        return position_data

    def plot_position(self, position_name, start_date=None, end_date=None):
        date_range = None
        if start_date and end_date:
            self.filter_by_date(start_date, end_date)
            date_range = self.get_date_range()

        position_data = self.analyze_position(position_name)
        high_risk_summary = position_data.groupby(["team_name", "player"]).agg([
            pl.col("high_risk_load").sum().alias("high_risk_load_count"),
            pl.col("high_risk_velocity").sum().alias("high_risk_velocity_count"),
            pl.col("high_risk_acceleration").sum().alias("high_risk_acceleration_count")
        ])
        high_risk_summary_df = high_risk_summary.to_pandas()
        self.plot_high_risk_summary(high_risk_summary_df, f'High-Risk Instances for Position {position_name}', date_range)


class PlayerAnalytics(SportsAnalytics):
    def __init__(self, data):
        super().__init__(data)

    def analyze_player(self, player_name):
        player_data = self.data.filter(pl.col("player") == player_name)
        return player_data

    def plot_player(self, player_name, start_date=None, end_date=None):
        date_range = None
        if start_date and end_date:
            self.filter_by_date(start_date, end_date)
            date_range = self.get_date_range()

        player_data = self.analyze_player(player_name)

        # Aggregate the data by date
        player_summary = player_data.groupby("date").agg([
            pl.col("total_player_load").sum().alias("total_player_load"),
            pl.col("total_acc_dec_efforts").sum().alias("total_acc_dec_efforts"),
            pl.col("high_intensity_distance_m_v5_v6_m").max().alias("high_intensity_distance_m_v5_v6_m")
        ])
        player_summary_df = player_summary.to_pandas()
        background_alpha = 0.25
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

        # Bar plot for total player load
        axs[0].bar(player_summary_df['date'], player_summary_df['total_player_load'], color='blue')
        axs[0].set_ylabel('Total Player Load', color='blue')
        axs[0].tick_params(axis='y', labelcolor='blue')
        axs[0].set_title(f'Total Player Load for {player_name}')

        # Shade background for total player load
        axs[0].axhspan(800, max(player_summary_df['total_player_load']), facecolor='none', hatch='////', edgecolor='red', alpha=background_alpha)
        axs[0].axhspan(400, 800, facecolor='none', hatch='\\\\\\\\', edgecolor='yellow', alpha=background_alpha)
        axs[0].axhspan(0, 400, facecolor='none', hatch='....', edgecolor='green', alpha=background_alpha)

        # Bar plot for total accelerations
        axs[1].bar(player_summary_df['date'], player_summary_df['total_acc_dec_efforts'], color='green')
        axs[1].set_ylabel('Total Accelerations', color='green')
        axs[1].tick_params(axis='y', labelcolor='green')
        axs[1].set_title(f'Total Accelerations for {player_name}')

        # Shade background for total accelerations
        axs[1].axhspan(15, max(player_summary_df['total_acc_dec_efforts']), facecolor='none', hatch='////', edgecolor='red', alpha=background_alpha)
        axs[1].axhspan(10, 15, facecolor='none', hatch='\\\\\\\\', edgecolor='yellow', alpha=background_alpha)
        axs[1].axhspan(0, 10, facecolor='none', hatch='....', edgecolor='green', alpha=background_alpha)

        # Bar plot for maximum velocity
        axs[2].bar(player_summary_df['date'], player_summary_df['high_intensity_distance_m_v5_v6_m'], color='red')
        axs[2].set_ylabel('High Intensity Running Distance (m)', color='red')
        axs[2].tick_params(axis='y', labelcolor='red')
        axs[2].set_title(f'Total High Intensity Running Distance for {player_name}')
        axs[2].set_xlabel('Date')

        # Shade background for maximum velocity
        axs[2].axhspan(500, max(player_summary_df['high_intensity_distance_m_v5_v6_m']), facecolor='none', hatch='////', edgecolor='red', alpha=background_alpha)
        axs[2].axhspan(250, 500, facecolor='none', hatch='\\\\\\\\', edgecolor='yellow', alpha=background_alpha)
        axs[2].axhspan(0, 250, facecolor='none', hatch='....', edgecolor='green', alpha=background_alpha)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'Player Performance for {player_name}', fontsize=16)

        if date_range:
            subtitle = f'Date Range: {date_range[0]} to {date_range[1]}'
            plt.figtext(0.5, 0.96, subtitle, ha='center', fontsize=10, color='gray')

        plt.show()

