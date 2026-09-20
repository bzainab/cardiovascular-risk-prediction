"""
Data preprocessing module for Framingham Heart Study dataset.
Handles loading, cleaning, normalization, encoding, and train/test split.
"""

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")

    def load_data(self):
        """Load the CSV dataset."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data shape: {self.df.shape}")
        return self.df

    def validate_data(self):
        """Check for missing values and data quality."""
        logger.info("Validating data...")

        # Check missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")
        else:
            logger.info("[OK] No missing values detected")

        # Check data types
        logger.info(f"Data types:\n{self.df.dtypes}")


    def identify_feature_types(self):
        """Classify features as binary, categorical, or continuous."""
        feature_types = {
            "binary": [],
            "categorical": [],
            "continuous": [],
            "target": "TenYearCHD",
        }

        for col in self.df.columns:
            if col == "TenYearCHD":
                feature_types["target"] = col
            elif self.df[col].nunique() == 2:
                feature_types["binary"].append(col)
            elif self.df[col].nunique() < 10:
                feature_types["categorical"].append(col)
            else:
                feature_types["continuous"].append(col)

        logger.info(f"Binary features: {feature_types['binary']}")
        logger.info(f"Categorical features: {feature_types['categorical']}")
        logger.info(f"Continuous features: {feature_types['continuous']}")

        return feature_types

    def encode_categorical_features(self):
        """Encode categorical variables (one-hot or label encoding)."""
        logger.info("Encoding categorical variables...")

        feature_types = self.identify_feature_types()

        # For 'education', use one-hot encoding since it has 4 classes
        if "education" in feature_types["categorical"]:
            self.df = pd.get_dummies(
                self.df, columns=["education"], prefix="edu", drop_first=False
            )
            logger.info(
                f"One-hot encoded 'education': {[col for col in self.df.columns if col.startswith('edu')]}"
            )

        return self.df

    def normalize_features(self, X_train, X_test=None):
        """Normalize continuous features using StandardScaler."""
        logger.info("Normalizing features...")

        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=X_train.columns, index=X_train.index
        )

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(
                X_test_scaled, columns=X_test.columns, index=X_test.index
            )
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def substitute_missing_values(self, X_train, X_test=None):
        logger.info("Substituting missing values...")

        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_imputed = pd.DataFrame(
            X_train_imputed, columns=X_train.columns, index=X_train.index
        )

        if X_test is not None:
            X_test_imputed = self.imputer.transform(X_test)
            X_test_imputed = pd.DataFrame(
                X_test_imputed, columns=X_test.columns, index=X_test.index
            )
            return X_train_imputed, X_test_imputed

        return X_train_imputed

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train/test with stratification."""
        logger.info(
            f"Splitting data: train={100-test_size*100:.0f}%, test={test_size*100:.0f}%"
        )

        # Encode categorical features first
        self.encode_categorical_features()

        # Separate features and target
        X = self.df.drop("TenYearCHD", axis=1)
        y = self.df["TenYearCHD"]

        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info(
            f"Training set size: {X_train.shape[0]} ({len(y_train[y_train==1])} positive)"
        )
        logger.info(
            f"Test set size: {X_test.shape[0]} ({len(y_test[y_test==1])} positive)"
        )

        X_train, X_test = self.substitute_missing_values(X_train, X_test)

        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        self.feature_names = X_train_scaled.columns.tolist()

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

        return X_train_scaled, X_test_scaled, y_train, y_test

    def get_feature_names(self):
        """Get list of feature names after preprocessing."""
        if self.X_train is not None:
            return self.X_train.columns.tolist()
        return None

    def save_preprocessor(self, output_path=None):
        """Save preprocessing artifacts for use in deployment."""
        if output_path is None:
            output_path = Path(__file__).parent.parent / "models" / "preprocessor.pkl"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessor_bundle = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }
        joblib.dump(preprocessor_bundle, output_path)
        logger.info(f"Preprocessor saved to {output_path}")

    def get_summary(self):
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Original dataset shape: {self.df.shape}")
        print("\nTarget variable distribution:")
        print(self.df["TenYearCHD"].value_counts(normalize=True))
        print(f"\nFinal feature count: {self.X_train.shape[1]}")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print("=" * 60 + "\n")


def preprocess_pipeline(
    data_path, test_size=0.2, random_state=42, preprocessor_output_path=None
):
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.validate_data()
    X_train, X_test, y_train, y_test = preprocessor.split_data(test_size, random_state)
    preprocessor.save_preprocessor(output_path=preprocessor_output_path)
    preprocessor.get_summary()

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "Framingham_Data" / "framingham.csv"
    X_train, X_test, y_train, y_test = preprocess_pipeline(str(data_path))
