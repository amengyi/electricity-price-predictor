import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class TimeSeriesModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='random_forest', params=None):
        self.model_type = model_type
        self.params = params if params is not None else {}
        self.model = None
        self.scaler = None
    
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.params)
            self.model = RandomForestRegressor(**default_params)
        
        elif self.model_type == 'ridge':
            default_params = {'alpha': 1.0, 'random_state': 42}
            default_params.update(self.params)
            self.model = Ridge(**default_params)
        
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': 0
            }
            default_params.update(self.params)
            self.model = XGBRegressor(**default_params)
        
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': -1
            }
            default_params.update(self.params)
            self.model = LGBMRegressor(**default_params)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names):
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(self.model.feature_importances_, index=feature_names)
            return importances.sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            importances = pd.Series(np.abs(self.model.coef_), index=feature_names)
            return importances.sort_values(ascending=False)
        else:
            return None

class StackingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_models=None, meta_model=None, n_splits=3):
        self.base_models = base_models if base_models is not None else []
        self.meta_model = meta_model if meta_model is not None else RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.n_splits = n_splits
        self.fitted_base_models = []
    
    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        n_samples = X.shape[0]
        oof_preds = np.zeros((n_samples, len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models):
            oof_single = np.zeros(n_samples)
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)
                
                model.fit(X_tr_scaled, y_tr)
                oof_single[val_idx] = model.predict(X_val_scaled)
                self.fitted_base_models.append((name, model, scaler))
            
            oof_preds[:, i] = oof_single
        
        self.meta_model.fit(oof_preds, y)
        return self
    
    def predict(self, X):
        meta_features = []
        for name, model, scaler in self.fitted_base_models:
            X_scaled = scaler.transform(X)
            meta_features.append(model.predict(X_scaled))
        
        meta_features = np.column_stack(meta_features)
        return self.meta_model.predict(meta_features)

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    smape_val = smape(y_true, y_pred)
    return {'RMSE': rmse, 'R²': r2, 'SMAPE': smape_val}

def hyperparameter_tuning(X, y, model_type, param_grid=None):
    tscv = TimeSeriesSplit(n_splits=3)
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    
    elif model_type == 'ridge':
        model = Ridge(random_state=42)
        if param_grid is None:
            param_grid = {'alpha': np.logspace(-3, 3, 10)}
    
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        model = XGBRegressor(random_state=42, verbosity=0)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
    
    elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        model = LGBMRegressor(random_state=42, verbose=-1)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
    
    else:
        raise ValueError(f"Unsupported model type for tuning: {model_type}")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=tscv, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=0
    )
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    grid_search.fit(X_scaled, y)
    
    return grid_search.best_params_

def create_ensemble_models():
    base_models = []
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    base_models.append(('RandomForest', rf_model))
    
    if XGBOOST_AVAILABLE:
        xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
        base_models.append(('XGBoost', xgb_model))
    
    if LIGHTGBM_AVAILABLE:
        lgb_model = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
        base_models.append(('LightGBM', lgb_model))
    
    ensemble = StackingEnsemble(base_models=base_models)
    return ensemble, base_models

def get_available_models():
    models = ['random_forest', 'ridge']
    if XGBOOST_AVAILABLE:
        models.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        models.append('lightgbm')
    return models

def get_model_name(model_type):
    names = {
        'random_forest': '随机森林',
        'ridge': '岭回归',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'ensemble': '集成模型'
    }
    return names.get(model_type, model_type)