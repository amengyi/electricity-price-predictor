import pandas as pd
import numpy as np
from datetime import timedelta

try:
    from sqlalchemy import create_engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

def load_from_database(db_config, table_name='anhui_hourly_spot_cleared_wide', start_date=None, end_date=None):
    """从数据库加载真实数据"""
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("需要安装sqlalchemy以连接数据库")
    
    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    
    query = f"SELECT * FROM {table_name}"
    conditions = []
    
    if start_date:
        conditions.append(f"date_hour >= '{start_date}'")
    if end_date:
        conditions.append(f"date_hour < '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY date_hour"
    
    df = pd.read_sql(query, engine)
    df['date_hour'] = pd.to_datetime(df['date_hour']).dt.tz_localize(None)
    
    engine.dispose()
    return df

CHINA_HOLIDAYS = {
    '2024-01-01': '元旦', '2024-02-10': '春节', '2024-02-11': '春节', '2024-02-12': '春节',
    '2024-02-13': '春节', '2024-02-14': '春节', '2024-02-15': '春节', '2024-02-16': '春节',
    '2024-02-17': '春节', '2024-04-04': '清明节', '2024-04-05': '清明节', '2024-04-06': '清明节',
    '2024-05-01': '劳动节', '2024-05-02': '劳动节', '2024-05-03': '劳动节', '2024-05-04': '劳动节',
    '2024-05-05': '劳动节', '2024-06-10': '端午节', '2024-09-15': '中秋节', '2024-09-16': '中秋节',
    '2024-09-17': '中秋节', '2024-10-01': '国庆节', '2024-10-02': '国庆节', '2024-10-03': '国庆节',
    '2024-10-04': '国庆节', '2024-10-05': '国庆节', '2024-10-06': '国庆节', '2024-10-07': '国庆节',
    '2025-01-01': '元旦', '2025-01-28': '春节', '2025-01-29': '春节', '2025-01-30': '春节',
    '2025-01-31': '春节', '2025-02-01': '春节', '2025-02-02': '春节', '2025-02-03': '春节',
    '2025-04-04': '清明节', '2025-04-05': '清明节', '2025-04-06': '清明节', '2025-05-01': '劳动节',
    '2025-05-02': '劳动节', '2025-05-03': '劳动节', '2025-05-04': '劳动节', '2025-05-05': '劳动节',
    '2025-06-01': '端午节', '2025-10-01': '国庆节', '2025-10-02': '国庆节', '2025-10-03': '国庆节',
    '2025-10-04': '国庆节', '2025-10-05': '国庆节', '2025-10-06': '国庆节', '2025-10-07': '国庆节',
    '2026-01-01': '元旦', '2026-01-28': '春节', '2026-01-29': '春节', '2026-01-30': '春节',
    '2026-01-31': '春节', '2026-02-01': '春节', '2026-02-02': '春节', '2026-02-03': '春节',
    '2026-04-04': '清明节', '2026-04-05': '清明节', '2026-04-06': '清明节', '2026-05-01': '劳动节',
    '2026-05-02': '劳动节', '2026-05-03': '劳动节', '2026-05-04': '劳动节', '2026-05-05': '劳动节',
    '2026-06-20': '端午节', '2026-10-01': '国庆节', '2026-10-02': '国庆节', '2026-10-03': '国庆节',
    '2026-10-04': '国庆节', '2026-10-05': '国庆节', '2026-10-06': '国庆节', '2026-10-07': '国庆节'
}

TIME_FEATURES = {
    'year', 'month', 'dayofmonth', 'dayofyear', 'hour', 'dayofweek', 'week', 'is_weekend',
    'hoursin', 'hourcos', 'dayofweeksin', 'dayofweekcos', 'weeksin', 'weekcos',
    'monthsin', 'monthcos', 'dayofyearsin', 'dayofyearcos', 'cos_time', 'sin_time',
    'season_cos', 'season_sin', 'weekofyear', 'quarter', 'season_num'
}

def load_sample_data():
    """加载示例数据用于演示"""
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq=pd.Timedelta(hours=1))
    df = pd.DataFrame({
        'date_hour': dates,
        'price_day_ahead_avg': np.random.normal(500, 100, len(dates)),
        'total_wind_solar_pre_avg': np.random.normal(1000, 200, len(dates)),
        'solar_radiation_avg': np.random.normal(500, 150, len(dates)),
        'cloud_avg': np.random.normal(50, 20, len(dates)),
        'pre_avg': np.random.normal(1013, 10, len(dates)),
        'tem_avg': np.random.normal(15, 10, len(dates)),
        'wind_speed_avg': np.random.normal(5, 3, len(dates)),
        'hum_avg': np.random.normal(60, 20, len(dates)),
        'load_pre_avg': np.random.normal(5000, 1000, len(dates)),
        'out_pre_avg': np.random.normal(1000, 200, len(dates)),
        'non_market_pre_avg': np.random.normal(500, 100, len(dates)),
        'hydro_pre_avg': np.random.normal(800, 150, len(dates)),
        'power_pre_avg': np.random.normal(7000, 1000, len(dates)),
        'balance_pre_avg': np.random.normal(200, 100, len(dates))
    })
    
    df['price_day_ahead_avg'] = df['price_day_ahead_avg'].clip(lower=0)
    df['total_wind_solar_pre_avg'] = df['total_wind_solar_pre_avg'].clip(lower=0)
    df['solar_radiation_avg'] = df['solar_radiation_avg'].clip(lower=0)
    df['cloud_avg'] = df['cloud_avg'].clip(0, 100)
    df['hum_avg'] = df['hum_avg'].clip(0, 100)
    
    df = add_time_features(df)
    df = add_holiday_features(df)
    
    return df

def add_time_features(df, time_col='date_hour'):
    df = df.copy()
    dt = pd.to_datetime(df[time_col])
    
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['dayofmonth'] = dt.dt.day
    df['dayofyear'] = dt.dt.dayofyear
    df['hour'] = dt.dt.hour
    df['dayofweek'] = dt.dt.dayofweek
    df['week'] = dt.dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['weekofyear'] = dt.dt.isocalendar().week.astype(int)
    df['quarter'] = dt.dt.quarter
    
    def get_season(month):
        if month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        elif month in [9, 10, 11]:
            return 3
        else:
            return 4
    
    df['season_num'] = df['month'].apply(get_season)
    df['season_sin'] = np.sin(2 * np.pi * (df['season_num'] - 1) / 4)
    df['season_cos'] = np.cos(2 * np.pi * (df['season_num'] - 1) / 4)
    
    df['hoursin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hourcos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweeksin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweekcos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['weeksin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['weekcos'] = np.cos(2 * np.pi * df['week'] / 52)
    df['monthsin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['monthcos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyearsin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyearcos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    df['sin_time'] = df['hoursin']
    df['cos_time'] = df['hourcos']
    
    return df

def add_holiday_features(df, time_col='date_hour'):
    df = df.copy()
    df['date'] = pd.to_datetime(df[time_col]).dt.date
    df['date_str'] = df['date'].astype(str)
    
    df['is_holiday'] = df['date_str'].isin(CHINA_HOLIDAYS.keys()).astype(int)
    
    df['is_pre_holiday'] = 0
    for holiday_date in CHINA_HOLIDAYS.keys():
        holiday_dt = pd.to_datetime(holiday_date)
        pre_day = holiday_dt - pd.Timedelta(days=1)
        pre_day_str = pre_day.strftime('%Y-%m-%d')
        df.loc[df['date_str'] == pre_day_str, 'is_pre_holiday'] = 1
    
    df['is_post_holiday'] = 0
    for holiday_date in CHINA_HOLIDAYS.keys():
        holiday_dt = pd.to_datetime(holiday_date)
        post_day = holiday_dt + pd.Timedelta(days=1)
        post_day_str = post_day.strftime('%Y-%m-%d')
        df.loc[df['date_str'] == post_day_str, 'is_post_holiday'] = 1
    
    df['days_to_nearest_holiday'] = 365
    all_holiday_dates = [pd.to_datetime(d).date() for d in CHINA_HOLIDAYS.keys()]
    
    for idx, row in df.iterrows():
        current_date = row['date']
        min_distance = 365
        for holiday_date in all_holiday_dates:
            distance = abs((current_date - holiday_date).days)
            if distance < min_distance:
                min_distance = distance
        df.at[idx, 'days_to_nearest_holiday'] = min_distance
    
    df['is_major_holiday'] = 0
    for date_str, holiday_name in CHINA_HOLIDAYS.items():
        if holiday_name in ['春节', '国庆节']:
            df.loc[df['date_str'] == date_str, 'is_major_holiday'] = 1
    
    df = df.drop(columns=['date', 'date_str'])
    
    return df

def add_statistical_features(df, target_col='price_day_ahead_avg'):
    df = df.copy()
    df = df.sort_values('date_hour')
    
    df[f'{target_col}_same_hour_2d_ago'] = df[target_col].shift(48)
    df[f'{target_col}_same_hour_7d_ago'] = df[target_col].shift(168)
    
    df[f'{target_col}_same_hour_30d_mean'] = df[target_col].rolling(window=24*30, min_periods=1).mean().shift(24)
    df[f'{target_col}_same_hour_30d_std'] = df[target_col].rolling(window=24*30, min_periods=1).std().shift(24)
    df[f'{target_col}_same_hour_30d_min'] = df[target_col].rolling(window=24*30, min_periods=1).min().shift(24)
    df[f'{target_col}_same_hour_30d_max'] = df[target_col].rolling(window=24*30, min_periods=1).max().shift(24)
    
    df['date'] = pd.to_datetime(df['date_hour']).dt.date
    daily_mean = df.groupby('date')[target_col].transform('mean')
    df[f'{target_col}_daily_mean_2d_ago'] = daily_mean.shift(48)
    df[f'{target_col}_daily_mean_7d_ago'] = daily_mean.shift(168)
    df[f'{target_col}_daily_mean_7d_rolling'] = daily_mean.rolling(7, min_periods=1).mean().shift(24)
    
    df[f'{target_col}_hourly_change'] = df[target_col].diff().shift(24)
    df[f'{target_col}_daily_change'] = df[target_col].diff(24).shift(24)
    
    df['hour_of_day'] = pd.to_datetime(df['date_hour']).dt.hour
    key_hours = [8, 9, 10, 11, 12, 18, 19, 20, 21, 22]
    for hour in key_hours:
        hour_mask = df['hour_of_day'] == hour
        df.loc[hour_mask, f'{target_col}_hour{hour}_30d_mean'] = (
            df.loc[hour_mask, target_col].rolling(window=30, min_periods=1).mean().shift(1)
        )
    
    df = df.drop(columns=['date', 'hour_of_day'])
    
    return df

def create_all_features(df, target_col='price_day_ahead_avg'):
    df = df.copy()
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = df.sort_values('date_hour').reset_index(drop=True)
    df = add_statistical_features(df, target_col)
    return df

def clean_data(df):
    df = df.copy()
    df = df.dropna(subset=['price_day_ahead_avg'])
    df = df[(df['price_day_ahead_avg'] >= 0) & (df['price_day_ahead_avg'] < 10000)]
    return df

def split_data(df, train_ratio=0.8, target_col='price_day_ahead_avg'):
    df = df.sort_values('date_hour')
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    X_train = train_df.drop(columns=[target_col, 'date_hour'])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col, 'date_hour'])
    y_test = test_df[target_col]
    
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    return X_train, X_test, y_train, y_test, train_df, test_df

def ensure_numeric_features(df, feature_columns):
    df = df.copy()
    for col in feature_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def create_future_features(df_future, df_history, target_col='price_day_ahead_avg'):
    df_future = df_future.copy()
    df_history = df_history.copy()
    
    df_future = add_time_features(df_future)
    df_future = add_holiday_features(df_future)
    df_history = df_history.sort_values('date_hour')
    
    future_features_list = []
    
    for idx, future_row in df_future.iterrows():
        future_time = future_row['date_hour']
        future_date = pd.to_datetime(future_time).date()
        future_hour = pd.to_datetime(future_time).hour
        
        features = {}
        
        two_days_ago = future_time - pd.Timedelta(days=2)
        seven_days_ago = future_time - pd.Timedelta(days=7)
        
        two_days_value = df_history[df_history['date_hour'] == two_days_ago][target_col].values
        seven_days_value = df_history[df_history['date_hour'] == seven_days_ago][target_col].values
        
        features[f'{target_col}_same_hour_2d_ago'] = two_days_value[0] if len(two_days_value) > 0 else np.nan
        features[f'{target_col}_same_hour_7d_ago'] = seven_days_value[0] if len(seven_days_value) > 0 else np.nan
        
        lookback_start = future_time - pd.Timedelta(days=31)
        historical_data = df_history[
            (df_history['date_hour'] >= lookback_start) & 
            (df_history['date_hour'] < future_time - pd.Timedelta(days=1))
        ]
        
        same_hour_data = historical_data[pd.to_datetime(historical_data['date_hour']).dt.hour == future_hour][target_col]
        
        if len(same_hour_data) > 0:
            features[f'{target_col}_same_hour_30d_mean'] = same_hour_data.mean()
            features[f'{target_col}_same_hour_30d_std'] = same_hour_data.std()
            features[f'{target_col}_same_hour_30d_min'] = same_hour_data.min()
            features[f'{target_col}_same_hour_30d_max'] = same_hour_data.max()
        else:
            features[f'{target_col}_same_hour_30d_mean'] = np.nan
            features[f'{target_col}_same_hour_30d_std'] = np.nan
            features[f'{target_col}_same_hour_30d_min'] = np.nan
            features[f'{target_col}_same_hour_30d_max'] = np.nan
        
        historical_data['date'] = pd.to_datetime(historical_data['date_hour']).dt.date
        daily_stats = historical_data.groupby('date')[target_col].mean()
        
        if len(daily_stats) > 1:
            features[f'{target_col}_daily_mean_2d_ago'] = daily_stats.iloc[-2] if len(daily_stats) >= 2 else np.nan
            features[f'{target_col}_daily_mean_7d_ago'] = daily_stats.iloc[-7] if len(daily_stats) >= 7 else np.nan
            features[f'{target_col}_daily_mean_7d_rolling'] = daily_stats.tail(7).mean()
        else:
            features[f'{target_col}_daily_mean_2d_ago'] = np.nan
            features[f'{target_col}_daily_mean_7d_ago'] = np.nan
            features[f'{target_col}_daily_mean_7d_rolling'] = np.nan
        
        if len(historical_data) > 24:
            features[f'{target_col}_hourly_change'] = historical_data[target_col].iloc[-1] - historical_data[target_col].iloc[-2]
            features[f'{target_col}_daily_change'] = historical_data[target_col].iloc[-1] - historical_data[target_col].iloc[-25]
        else:
            features[f'{target_col}_hourly_change'] = np.nan
            features[f'{target_col}_daily_change'] = np.nan
        
        if future_hour in [8, 9, 10, 11, 12, 18, 19, 20, 21, 22]:
            features[f'{target_col}_hour{future_hour}_30d_mean'] = same_hour_data.mean() if len(same_hour_data) > 0 else np.nan
        
        future_features_list.append(features)
    
    future_features_df = pd.DataFrame(future_features_list, index=df_future.index)
    df_future = pd.concat([df_future, future_features_df], axis=1)
    
    return df_future