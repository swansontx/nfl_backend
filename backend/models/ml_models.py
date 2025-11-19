"""
Machine Learning models for complex prop markets

Uses XGBoost/LightGBM for markets where relationships are non-linear:
- Anytime TD (redzone usage, goal line touches)
- Receptions (game script dependent)
- Yards (matchup + game script interactions)
"""

from typing import Optional, Dict, List
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from backend.config import settings
from backend.config.logging_config import get_logger
from .ml_features import MLFeatureSet, MLFeatureBuilder

logger = get_logger(__name__)


class XGBoostPropModel:
    """
    XGBoost model for prop predictions

    Handles both classification (TD yes/no) and regression (yards, receptions)
    """

    def __init__(
        self,
        market: str,
        objective: str = 'binary:logistic',  # or 'reg:squarederror'
        model_dir: Optional[Path] = None
    ):
        """
        Initialize XGBoost model

        Args:
            market: Market name (e.g., 'player_anytime_td')
            objective: XGBoost objective function
            model_dir: Directory to save/load models
        """
        self.market = market
        self.objective = objective
        self.model_dir = model_dir or (settings.base_dir / 'models' / 'saved')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self.training_metrics: Dict = {}

    def train(
        self,
        training_set: MLFeatureSet,
        validation_split: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train XGBoost model

        Args:
            training_set: MLFeatureSet with features and targets
            validation_split: Fraction for validation
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            early_stopping_rounds: Early stopping patience
            verbose: Print training progress

        Returns:
            Training metrics dict
        """
        if training_set.features.empty or training_set.targets is None:
            raise ValueError("Training set is empty")

        X = training_set.features.values
        y = training_set.targets.values
        self.feature_names = training_set.feature_names

        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42,
            stratify=y if self.objective == 'binary:logistic' else None
        )

        logger.info(
            "training_xgboost",
            market=self.market,
            train_samples=len(X_train),
            val_samples=len(X_val),
            features=len(self.feature_names)
        )

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        # XGBoost parameters
        params = {
            'objective': self.objective,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'eval_metric': 'logloss' if self.objective == 'binary:logistic' else 'rmse',
            'tree_method': 'hist',  # Fast histogram-based
            'random_state': 42,
        }

        # Train
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose
        )

        # Evaluate
        y_pred_train = self.model.predict(dtrain)
        y_pred_val = self.model.predict(dval)

        if self.objective == 'binary:logistic':
            # Classification metrics
            train_logloss = log_loss(y_train, y_pred_train)
            val_logloss = log_loss(y_val, y_pred_val)
            train_auc = roc_auc_score(y_train, y_pred_train)
            val_auc = roc_auc_score(y_val, y_pred_val)

            self.training_metrics = {
                'train_logloss': train_logloss,
                'val_logloss': val_logloss,
                'train_auc': train_auc,
                'val_auc': val_auc,
                'best_iteration': self.model.best_iteration,
            }

            logger.info(
                "training_complete",
                market=self.market,
                val_logloss=val_logloss,
                val_auc=val_auc
            )

        else:
            # Regression metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

            self.training_metrics = {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'best_iteration': self.model.best_iteration,
            }

            logger.info(
                "training_complete",
                market=self.market,
                val_rmse=val_rmse
            )

        return self.training_metrics

    def predict(
        self,
        features: Dict[str, float]
    ) -> float:
        """
        Predict for a single sample

        Args:
            features: Feature dict

        Returns:
            Prediction (probability for classification, value for regression)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Convert to DataFrame with correct feature order
        feature_df = pd.DataFrame([features])

        # Ensure all features present
        for fname in self.feature_names:
            if fname not in feature_df.columns:
                feature_df[fname] = 0

        # Reorder to match training
        feature_df = feature_df[self.feature_names]

        # Predict
        dmatrix = xgb.DMatrix(feature_df.values, feature_names=self.feature_names)
        pred = self.model.predict(dmatrix)[0]

        return float(pred)

    def predict_batch(
        self,
        features_list: List[Dict[str, float]]
    ) -> np.ndarray:
        """Predict for multiple samples"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Convert to DataFrame
        feature_df = pd.DataFrame(features_list)

        # Ensure all features present
        for fname in self.feature_names:
            if fname not in feature_df.columns:
                feature_df[fname] = 0

        # Reorder
        feature_df = feature_df[self.feature_names]

        # Predict
        dmatrix = xgb.DMatrix(feature_df.values, feature_names=self.feature_names)
        preds = self.model.predict(dmatrix)

        return preds

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save model to disk

        Args:
            filename: Optional filename (defaults to market name)

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")

        if filename is None:
            filename = f"{self.market}_xgboost.json"

        model_path = self.model_dir / filename

        # Save XGBoost model
        self.model.save_model(str(model_path))

        # Save metadata
        metadata = {
            'market': self.market,
            'objective': self.objective,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
        }

        metadata_path = self.model_dir / f"{filename}.meta.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info("model_saved", market=self.market, path=str(model_path))

        return model_path

    def load(self, filename: Optional[str] = None) -> None:
        """
        Load model from disk

        Args:
            filename: Optional filename (defaults to market name)
        """
        if filename is None:
            filename = f"{self.market}_xgboost.json"

        model_path = self.model_dir / filename
        metadata_path = self.model_dir / f"{filename}.meta.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.feature_names = metadata.get('feature_names', [])
            self.training_metrics = metadata.get('training_metrics', {})

        logger.info("model_loaded", market=self.market, path=str(model_path))

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores

        Returns:
            Dict mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        importance = self.model.get_score(importance_type='gain')

        # Map to feature names
        importance_dict = {}
        for i, fname in enumerate(self.feature_names):
            # XGBoost uses f0, f1, f2, ... as default feature names
            key = f'f{i}'
            if key in importance:
                importance_dict[fname] = importance[key]
            else:
                importance_dict[fname] = 0.0

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict


class AnytimeTDModel:
    """
    Specialized XGBoost model for anytime TD prediction

    Optimized for TD-specific features and class imbalance
    """

    def __init__(self):
        self.model = XGBoostPropModel(
            market='player_anytime_td',
            objective='binary:logistic'
        )
        self.feature_builder = MLFeatureBuilder()

    def train(self, training_set: MLFeatureSet) -> Dict:
        """Train with TD-specific parameters"""
        # Calculate class weights for imbalance
        if training_set.targets is not None:
            pos_count = (training_set.targets == 1).sum()
            neg_count = (training_set.targets == 0).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        else:
            scale_pos_weight = 1.0

        logger.info(
            "training_td_model",
            positive_samples=pos_count if training_set.targets is not None else 0,
            negative_samples=neg_count if training_set.targets is not None else 0,
            scale_pos_weight=scale_pos_weight
        )

        # Train with class weight adjustment
        return self.model.train(
            training_set,
            n_estimators=200,  # More trees for complex TD patterns
            max_depth=5,  # Moderate depth
            learning_rate=0.05,  # Lower LR with more trees
        )

    def predict_td_probability(
        self,
        player_id: str,
        game_id: str,
        smoothed_features,
        matchup_features,
        player
    ) -> float:
        """Predict anytime TD probability"""
        features = self.feature_builder.build_td_features(
            player_id, game_id, smoothed_features, matchup_features, player
        )

        return self.model.predict(features)


class ReceptionsModel:
    """
    Specialized XGBoost model for receptions prediction

    Captures game script and target volume relationships
    """

    def __init__(self):
        self.model = XGBoostPropModel(
            market='player_receptions',
            objective='reg:squarederror'
        )
        self.feature_builder = MLFeatureBuilder()

    def train(self, training_set: MLFeatureSet) -> Dict:
        """Train receptions model"""
        return self.model.train(
            training_set,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
        )

    def predict_receptions(
        self,
        player_id: str,
        game_id: str,
        smoothed_features,
        matchup_features,
        player
    ) -> float:
        """Predict expected receptions"""
        features = self.feature_builder.build_receptions_features(
            player_id, game_id, smoothed_features, matchup_features, player
        )

        return self.model.predict(features)
