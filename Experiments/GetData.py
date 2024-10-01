import pandas as pd

class DataPreprocessor:
    """
    A class used to load, prepare, and preprocess dataset for training and testing.

    Attributes
    ----------
    train : bool
        A flag indicating whether the data is for training (True) or testing (False).
    data : DataFrame
        The loaded and prepared dataset.
    feature_columns : list
        A list of feature columns used for training the model.
    target_column : str
        The target column for prediction, default is 'results'.

    Methods
    -------
    load_data():
        Loads the data from parquet files depending on the train attribute.
    preprocess_data():
        Prepares the data by loading it, removing columns with too many missing values or single unique values, 
        and scaling the feature columns.
    get_data():
        Prepares the data and returns the processed dataset, feature columns, and target column.
    """

    def __init__(self, train=True):
        """
        Constructs all the necessary attributes for the DataPreprocessor object.

        Parameters
        ----------
        train : bool, optional
            A flag indicating whether the data is for training (default is True).
        """
        self.train = train
        self.data = None
        self.feature_columns = None
        self.target_column = 'results'

    def load_data(self):
        """
        Loads the data from parquet files depending on the train attribute.
        """
        if self.train:
            self.data = pd.read_parquet("data/prepared_train_data.parquet")
        else:
            self.data = pd.read_parquet("data/prepared_test_data.parquet")

    def preprocess_data(self):
        """
        Preprocesses the data by loading it, removing columns with too many missing values or single unique values, 
        and scaling the feature columns.
        """
        self.load_data()

        self.feature_columns = [col for col in self.data.columns if col != self.target_column and self.data[col].unique().size > 1]

        if self.train:
            # Remove columns that have too many NaNs
            threshold = 0.2
            empty_data = self.data.isna().sum() / len(self.data)
            columns_to_drop = empty_data[empty_data > threshold].index
            print(f"Removed {round(100 * len(columns_to_drop) / len(self.data.columns), 2)}% of columns because they have more than {round(100 * threshold)}% of values missing")
            self.data = self.data.drop(columns=columns_to_drop)

            # Remove columns that only have one value
            unique_data = self.data.nunique()
            columns_to_drop = unique_data[unique_data == 1].index
            print(f"Removed {round(100 * len(columns_to_drop) / len(self.data.columns), 2)}% of columns because they have only one value")
            self.data = self.data.drop(columns=columns_to_drop)

            # Remove rows with missing values
            n = len(self.data)
            self.data = self.data.dropna(how="any").dropna(axis=1)
            print(f"Removed {round(100 * (1 - len(self.data) / n), 2)}% of rows because they have missing values")

            # Update feature_columns
            self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]

    def get_data(self):
        """
        Prepares the data and returns the processed dataset, feature columns, and target column.

        Returns
        -------
        tuple
            A tuple containing the processed DataFrame, the list of feature columns, and the target column name.
        """
        self.preprocess_data()
        return self.data, self.feature_columns, self.target_column
