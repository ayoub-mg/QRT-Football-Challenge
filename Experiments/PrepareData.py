import pandas as pd
import numpy as np

class DataPreparer:
    """
    A class used to load, prepare, and optionally save data for training and testing.

    Attributes
    ----------
    path : str
        The path to the data directory.
    mode : str
        The mode of operation, either 'train' or 'test'.
    save_to_csv : bool
        A flag indicating whether to save the prepared data to an CSV file.
    colstonotconsider : list
        A list of columns to be excluded from the data.
    data_home_team : DataFrame
        The home team statistics data.
    data_home_players : DataFrame
        The home players statistics data.
    data_away_team : DataFrame
        The away team statistics data.
    data_away_players : DataFrame
        The away players statistics data.
    data_match : DataFrame
        The match results data.
    data : DataFrame
        The prepared data.

    Methods
    -------
    load_data():
        Loads the data from CSV files depending on the mode attribute.
    remove_columns():
        Removes columns that are not considered from the data.
    rename_columns(data, prefix='', suffix=''):
        Renames columns of the data with a specified prefix or suffix.
    prepare_player_data(data, prefix):
        Prepares player data by calculating sum, max, min, mean, and median values.
    prepare_data():
        Prepares the data by loading it, removing unwanted columns, renaming columns, and merging datasets.
    save_data():
        Saves the prepared data to parquet and optionally to an CSV file.
    """

    def __init__(self, path, train=True, save_to_csv=True):
        """
        Constructs all the necessary attributes for the DataPreparer object.

        Parameters
        ----------
        path : str
            The path to the data directory.
        mode : str, optional
            The mode of operation, either 'train' or 'test' (default is "train").
        save_to_scv : bool, optional
            A flag indicating whether to save the prepared data to an CSV file (default is True).
        """
        self.train = train
        self.path = path
        self.save_to_csv = save_to_csv
        self.data_home_team = None
        self.data_home_players = None
        self.data_away_team = None
        self.data_away_players = None
        self.data_match = None
        self.data = None
        self.load_data()

    def load_data(self):
        """
        Loads the data from CSV files depending on the mode attribute.
        """
        if self.train:
            data_home_team_path = self.path + "train_home_team_statistics_df.csv"
            data_home_players_path = self.path + "train_home_player_statistics_df.csv"
            data_away_team_path = self.path + "train_away_team_statistics_df.csv"
            data_away_players_path = self.path + "train_away_player_statistics_df.csv"
            data_match_path = self.path + "Y_train_1rknArQ.csv"
            self.data_match = pd.read_csv(data_match_path)
        else:
            data_home_team_path = self.path + "test_home_team_statistics_df.csv"
            data_home_players_path = self.path + "test_home_player_statistics_df.csv"
            data_away_team_path = self.path + "test_away_team_statistics_df.csv"
            data_away_players_path = self.path + "test_away_player_statistics_df.csv"

        self.data_home_team = pd.read_csv(data_home_team_path, index_col=0)
        self.data_home_players = pd.read_csv(data_home_players_path, index_col=0)
        self.data_away_team = pd.read_csv(data_away_team_path, index_col=0)
        self.data_away_players = pd.read_csv(data_away_players_path, index_col=0)

    def remove_columns(self):
        """
        Removes columns that are not considered from the data.
        """
        for dataframe in [self.data_home_team, self.data_home_players, self.data_away_team, self.data_away_players]:
            for col in ['TEAM_NAME', 'LEAGUE', 'PLAYER_NAME', 'POSITION']:
                if col in dataframe.columns:
                    dataframe.drop(col, axis=1, inplace=True)

    def rename_columns(self, data, prefix='', suffix=''):
        """
        Renames columns of the data with a specified prefix or suffix.

        Parameters
        ----------
        data : DataFrame
            The data whose columns need to be renamed.
        prefix : str, optional
            The prefix to add to the columns (default is '').
        suffix : str, optional
            The suffix to add to the columns (default is '').

        Returns
        -------
        DataFrame
            The data with renamed columns.
        """
        if prefix:
            data.columns = prefix + data.columns
        if suffix:
            data.columns = data.columns + suffix
        return data

    def prepare_player_data(self, data):
        """
        Prepares player data by calculating sum, max, min, mean, and median values.

        Parameters
        ----------
        data : DataFrame
            The player data to be prepared.
        prefix : str
            The prefix to add to the columns.

        Returns
        -------
        DataFrame
            The prepared player data.
        """
        aggregations = {}
        for col in data.columns:
            if 'sum' in col:
                aggregations[col] = 'sum'
            elif 'average' in col:
                aggregations[col] = 'mean'
            elif 'std' in col:
                aggregations[col] = lambda x: np.sqrt(np.mean(x**2) - np.mean(x)**2)
            
        return data.groupby(data.index).agg(aggregations)

    def prepare_data(self):
        """
        Prepares the data by loading it, removing unwanted columns, renaming columns, and merging datasets.
        """
        self.remove_columns()

        if self.train == True:
            self.data_match['results'] = self.data_match.apply(lambda x: 0 if x['HOME_WINS'] > 0 else 1 if x['DRAW'] else 2, axis=1)
            self.data_match = self.data_match.drop(['HOME_WINS', 'DRAW', 'AWAY_WINS'], axis=1)

        self.data_home_team = self.rename_columns(self.data_home_team, prefix='HOME_')
        self.data_away_team = self.rename_columns(self.data_away_team, prefix='AWAY_')
        self.data_home_players = self.rename_columns(self.data_home_players, prefix='HOME_')
        self.data_away_players = self.rename_columns(self.data_away_players, prefix='AWAY_')

        data_home_players_prepared = self.prepare_player_data(self.data_home_players)
        data_away_players_prepared = self.prepare_player_data(self.data_away_players)

        if self.train :
            self.data = pd.concat([self.data_match, self.data_home_team, self.data_away_team], axis=1)
        else:
            self.data = pd.concat([self.data_home_team, self.data_away_team], axis=1)

        self.data = pd.concat([self.data, data_home_players_prepared, data_away_players_prepared], axis=1)
        if self.train :
            self.data.set_index('ID', inplace=True)


    def save_data(self):
        """
        Saves the prepared data to parquet and optionally to an CSV file.
        """
        output_prefix = "train" if self.train else "test"
        self.data.to_parquet(f"data/prepared_{output_prefix}_data.parquet", index=True)
        print(f"Saving the prepared data to prepared_{output_prefix}_data.parquet")

        if self.save_to_csv:
            self.data.to_csv(f"data/prepared_{output_prefix}_data.csv", index=True)
            print(f"Saving the prepared data to prepared_{output_prefix}_data.csv")

        print('Data prepared and saved!')

