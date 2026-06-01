import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel
from sqlalchemy import create_engine, text, inspect
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 配置数据库连接
# ==============================
DB_CONFIG = {
    'host': 'pgm-bp1q60i1ca0xnwv98o.pg.rds.aliyuncs.com',
    'port': 1921,
    'database': 'bopha',
    'user': 'seniverse',
    'password': 'mpr3uOFnnM5bHtl'
}

engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ==============================
# ✅ 明确定义 final_features（基础特征）
# ==============================
base_features = [
    'total_wind_solar_pre_avg',
    'solar_radiation_avg',
    'weekcos',
    'cloud_avg',
    'dayofyearcos',
    'dayofyearsin',
    'hourcos',
    'pre_avg',
    'dayofmonth',
    'week',
    'dayofyear',
    'hoursin',
    'monthsin',
    'tem_avg',
    'dayofweekcos',
    'weeksin',
    'pm10_avg',
    'wind_speed_avg',
    'is_weekend',
    'wind120_avg',
    'wind180_avg',
    'hum_avg',
    'heat_avg',
    'dir_avg',
    'pm2_5_avg',
    'wind10_avg',
    'wind100_avg',
    'sin_time',
    'cos_time',
    'season_cos',
    'season_sin',
    'monthcos',
    'month',
    'weekofyear',
    'quarter',
    'season_num',
    'load_pre_avg',
    'out_pre_avg',
    'non_market_pre_avg',
    'hydro_pre_avg',
    'power_pre_avg',
    'balance_pre_avg'
]

print(f"✅ 使用预定义基础特征，共 {len(base_features)} 个特征")

# ==============================
# 🆕 时间特征集合
# ==============================
TIME_FEATURES = {
    'year', 'month', 'dayofmonth', 'dayofyear', 'hour', 'dayofweek', 'week', 'is_weekend',
    'hoursin', 'hourcos',
    'dayofweeksin', 'dayofweekcos',
    'weeksin', 'weekcos',
    'monthsin', 'monthcos',
    'dayofyearsin', 'dayofyearcos',
    'cos_time', 'sin_time',
    'season_cos', 'season_sin',
    'weekofyear', 'quarter', 'season_num'
}

raw_features = [f for f in base_features if f not in TIME_FEATURES]
print(f"📥 原始特征（从数据库读取）: {len(raw_features)} 个")
print(f"🕒 时间特征（动态生成）: {len(set(base_features) & TIME_FEATURES)} 个")

# ==============================
# 🆕 中国节假日定义（2024-2026年）
# ==============================
CHINA_HOLIDAYS = {
    # 2024年
    '2024-01-01': '元旦',
    '2024-02-10': '春节',
    '2024-02-11': '春节',
    '2024-02-12': '春节',
    '2024-02-13': '春节',
    '2024-02-14': '春节',
    '2024-02-15': '春节',
    '2024-02-16': '春节',
    '2024-02-17': '春节',
    '2024-04-04': '清明节',
    '2024-04-05': '清明节',
    '2024-04-06': '清明节',
    '2024-05-01': '劳动节',
    '2024-05-02': '劳动节',
    '2024-05-03': '劳动节',
    '2024-05-04': '劳动节',
    '2024-05-05': '劳动节',
    '2024-06-10': '端午节',
    '2024-09-15': '中秋节',
    '2024-09-16': '中秋节',
    '2024-09-17': '中秋节',
    '2024-10-01': '国庆节',
    '2024-10-02': '国庆节',
    '2024-10-03': '国庆节',
    '2024-10-04': '国庆节',
    '2024-10-05': '国庆节',
    '2024-10-06': '国庆节',
    '2024-10-07': '国庆节',
    
    # 2025年
    '2025-01-01': '元旦',
    '2025-01-28': '春节',
    '2025-01-29': '春节',
    '2025-01-30': '春节',
    '2025-01-31': '春节',
    '2025-02-01': '春节',
    '2025-02-02': '春节',
    '2025-02-03': '春节',
    '2025-04-04': '清明节',
    '2025-04-05': '清明节',
    '2025-04-06': '清明节',
    '2025-05-01': '劳动节',
    '2025-05-02': '劳动节',
    '2025-05-03': '劳动节',
    '2025-05-04': '劳动节',
    '2025-05-05': '劳动节',
    '2025-06-01': '端午节',
    '2025-10-01': '国庆节',
    '2025-10-02': '国庆节',
    '2025-10-03': '国庆节',
    '2025-10-04': '国庆节',
    '2025-10-05': '国庆节',
    '2025-10-06': '国庆节',
    '2025-10-07': '国庆节',
    
    # 2026年
    '2026-01-01': '元旦',
    '2026-01-28': '春节',
    '2026-01-29': '春节',
    '2026-01-30': '春节',
    '2026-01-31': '春节',
    '2026-02-01': '春节',
    '2026-02-02': '春节',
    '2026-02-03': '春节',
    '2026-04-04': '清明节',
    '2026-04-05': '清明节',
    '2026-04-06': '清明节',
    '2026-05-01': '劳动节',
    '2026-05-02': '劳动节',
    '2026-05-03': '劳动节',
    '2026-05-04': '劳动节',
    '2026-05-05': '劳动节',
    '2026-06-20': '端午节',
    '2026-10-01': '国庆节',
    '2026-10-02': '国庆节',
    '2026-10-03': '国庆节',
    '2026-10-04': '国庆节',
    '2026-10-05': '国庆节',
    '2026-10-06': '国庆节',
    '2026-10-07': '国庆节'
}

# ==============================
# 🆕 时间特征生成函数
# ==============================
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

# ==============================
# 🆕 添加节假日特征
# ==============================
def add_holiday_features(df, time_col='date_hour'):
    """添加节假日相关特征"""
    df = df.copy()
    df['date'] = pd.to_datetime(df[time_col]).dt.date
    df['date_str'] = df['date'].astype(str)
    
    # 1. 是否为节假日
    df['is_holiday'] = df['date_str'].isin(CHINA_HOLIDAYS.keys()).astype(int)
    
    # 2. 是否为节假日前一天（通常会有价格波动）
    df['is_pre_holiday'] = 0
    for holiday_date in CHINA_HOLIDAYS.keys():
        holiday_dt = pd.to_datetime(holiday_date)
        pre_day = holiday_dt - pd.Timedelta(days=1)
        pre_day_str = pre_day.strftime('%Y-%m-%d')
        df.loc[df['date_str'] == pre_day_str, 'is_pre_holiday'] = 1
    
    # 3. 是否为节假日后一天
    df['is_post_holiday'] = 0
    for holiday_date in CHINA_HOLIDAYS.keys():
        holiday_dt = pd.to_datetime(holiday_date)
        post_day = holiday_dt + pd.Timedelta(days=1)
        post_day_str = post_day.strftime('%Y-%m-%d')
        df.loc[df['date_str'] == post_day_str, 'is_post_holiday'] = 1
    
    # 4. 距离最近节假日的天数
    df['days_to_nearest_holiday'] = 365  # 初始化为最大值
    all_holiday_dates = [pd.to_datetime(d).date() for d in CHINA_HOLIDAYS.keys()]
    
    for idx, row in df.iterrows():
        current_date = row['date']
        min_distance = 365
        for holiday_date in all_holiday_dates:
            distance = abs((current_date - holiday_date).days)
            if distance < min_distance:
                min_distance = distance
        df.at[idx, 'days_to_nearest_holiday'] = min_distance
    
    # 5. 是否为重大节日（春节、国庆节）
    df['is_major_holiday'] = 0
    for date_str, holiday_name in CHINA_HOLIDAYS.items():
        if holiday_name in ['春节', '国庆节']:
            df.loc[df['date_str'] == date_str, 'is_major_holiday'] = 1
    
    # 清理临时列
    df = df.drop(columns=['date', 'date_str'])
    
    return df

# ==============================
# 🆕 安全的统计特征（避免数据泄露）
# ==============================
def add_safe_statistical_features(df, target_col='price_day_ahead_avg'):
    """
    添加安全的统计特征，避免数据泄露
    
    规则：
    - 预测第D天的价格时，只能使用第D-2天及之前的数据
    """
    df = df.copy()
    df = df.sort_values('date_hour')
    
    # 1. 关键的历史同期价格
    df[f'{target_col}_same_hour_2d_ago'] = df[target_col].shift(48)  # 48小时前
    df[f'{target_col}_same_hour_7d_ago'] = df[target_col].shift(168)  # 168小时前
    
    # 2. 滚动统计特征（使用足够长的窗口并shift避免数据泄露）
    # 最近30天同小时价格的移动平均（shift 24小时确保安全）
    df[f'{target_col}_same_hour_30d_mean'] = (
        df[target_col].rolling(window=24*30, min_periods=1).mean().shift(24)
    )
    
    # 最近30天同小时价格的标准差
    df[f'{target_col}_same_hour_30d_std'] = (
        df[target_col].rolling(window=24*30, min_periods=1).std().shift(24)
    )
    
    # 最近30天同小时价格的最小值
    df[f'{target_col}_same_hour_30d_min'] = (
        df[target_col].rolling(window=24*30, min_periods=1).min().shift(24)
    )
    
    # 最近30天同小时价格的最大值
    df[f'{target_col}_same_hour_30d_max'] = (
        df[target_col].rolling(window=24*30, min_periods=1).max().shift(24)
    )
    
    # 3. 日度统计特征
    # 计算每日平均价格
    df['date'] = pd.to_datetime(df['date_hour']).dt.date
    daily_mean = df.groupby('date')[target_col].transform('mean')
    
    # 前天的日平均价格
    df[f'{target_col}_daily_mean_2d_ago'] = daily_mean.shift(48)
    
    # 上周同一天的日平均价格
    df[f'{target_col}_daily_mean_7d_ago'] = daily_mean.shift(168)
    
    # 最近7天日平均价格的移动平均
    df[f'{target_col}_daily_mean_7d_rolling'] = daily_mean.rolling(7, min_periods=1).mean().shift(24)
    
    # 4. 价格变化率特征
    df[f'{target_col}_hourly_change'] = df[target_col].diff().shift(24)
    df[f'{target_col}_daily_change'] = df[target_col].diff(24).shift(24)
    
    # 5. 小时级特征（只保留几个关键小时）
    df['hour_of_day'] = pd.to_datetime(df['date_hour']).dt.hour
    
    # 只保留关键小时的特征（高峰小时：8-12, 18-22）
    key_hours = [8, 9, 10, 11, 12, 18, 19, 20, 21, 22]
    for hour in key_hours:
        hour_mask = df['hour_of_day'] == hour
        df.loc[hour_mask, f'{target_col}_hour{hour}_30d_mean'] = (
            df.loc[hour_mask, target_col].rolling(window=30, min_periods=1).mean().shift(1)
        )
    
    # 清理临时列
    df = df.drop(columns=['date', 'hour_of_day'])
    
    return df

# ==============================
# 🆕 完整的特征工程函数
# ==============================
def create_all_features(df, target_col='price_day_ahead_avg'):
    """执行完整的特征工程流程"""
    df = df.copy()
    
    # 1. 添加时间特征
    df = add_time_features(df)
    
    # 2. 添加节假日特征
    df = add_holiday_features(df)
    
    # 3. 添加安全的统计特征（需要先排序）
    df = df.sort_values('date_hour').reset_index(drop=True)
    df = add_safe_statistical_features(df, target_col)
    
    return df

# ==============================
# 🆕 获取最终特征列表（筛选前的完整列表）
# ==============================
def get_all_features(base_features, target_col='price_day_ahead_avg'):
    """生成包含所有特征的完整列表（筛选前）"""
    all_features = base_features.copy()
    
    # 添加节假日特征
    holiday_features = ['is_holiday', 'is_pre_holiday', 'is_post_holiday', 
                       'days_to_nearest_holiday', 'is_major_holiday']
    all_features.extend(holiday_features)
    
    # 添加统计特征
    statistical_features = [
        f'{target_col}_same_hour_2d_ago',
        f'{target_col}_same_hour_7d_ago',
        f'{target_col}_same_hour_30d_mean',
        f'{target_col}_same_hour_30d_std',
        f'{target_col}_same_hour_30d_min',
        f'{target_col}_same_hour_30d_max',
        f'{target_col}_daily_mean_2d_ago',
        f'{target_col}_daily_mean_7d_ago',
        f'{target_col}_daily_mean_7d_rolling',
        f'{target_col}_hourly_change',
        f'{target_col}_daily_change',
    ]
    
    # 添加关键小时级特征
    key_hours = [8, 9, 10, 11, 12, 18, 19, 20, 21, 22]
    for hour in key_hours:
        statistical_features.append(f'{target_col}_hour{hour}_30d_mean')
    
    all_features.extend(statistical_features)
    
    return all_features

# ==============================
# 🆕 特征筛选函数（两阶段筛选）
# ==============================
def feature_selection_pipeline(X, y, target_col='price_day_ahead_avg'):
    """
    两阶段特征筛选流程：
    阶段1：基于相关性筛选（去除高度相关的特征）
    阶段2：基于模型重要性筛选
    """
    print(f"\n🔍 开始特征筛选流程...")
    print(f"筛选前特征数量: {X.shape[1]}")
    
    # 阶段1：相关性筛选
    print("\n📊 阶段1：基于相关性的筛选")
    selected_features_phase1 = correlation_based_selection(X, y, target_col)
    print(f"阶段1后保留特征: {len(selected_features_phase1)} 个")
    
    # 阶段2：模型重要性筛选
    print("\n📊 阶段2：基于模型重要性的筛选")
    selected_features_phase2 = model_based_selection(
        X[selected_features_phase1], y, target_col
    )
    print(f"阶段2后保留特征: {len(selected_features_phase2)} 个")
    
    return selected_features_phase2

def correlation_based_selection(X, y, target_col, corr_threshold=0.95):
    """
    基于相关性筛选特征：
    1. 计算特征之间的相关性矩阵
    2. 去除与目标变量相关性低的特征
    3. 去除高度相关的特征（保留一个）
    """
    # 1. 计算特征与目标变量的相关性
    corr_with_target = X.corrwith(y)
    corr_with_target = corr_with_target.abs().sort_values(ascending=False)
    
    # 保留与目标变量相关性最高的前50%的特征
    keep_ratio = 0.5
    n_keep = int(len(corr_with_target) * keep_ratio)
    high_corr_features = corr_with_target.head(n_keep).index.tolist()
    
    # 2. 去除高度相关的特征
    X_filtered = X[high_corr_features].copy()
    corr_matrix = X_filtered.corr().abs()
    
    # 创建上三角矩阵（不包括对角线）
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 找出相关性高于阈值的特征对
    to_drop = set()
    for col in upper_tri.columns:
        high_corr_cols = upper_tri[col][upper_tri[col] > corr_threshold].index.tolist()
        for col2 in high_corr_cols:
            # 保留与目标变量相关性更高的那个
            if abs(corr_with_target[col]) < abs(corr_with_target[col2]):
                to_drop.add(col)
            else:
                to_drop.add(col2)
    
    # 去除高度相关的特征
    selected_features = [col for col in high_corr_features if col not in to_drop]
    
    print(f"   - 与目标变量相关性: 保留前{keep_ratio*100}%的特征 ({n_keep}个)")
    print(f"   - 去除高度相关特征: 去除{len(to_drop)}个特征")
    print(f"   - 关键特征示例: {selected_features[:10]}")
    
    return selected_features

def model_based_selection(X, y, target_col, importance_threshold='median'):
    """
    基于模型重要性筛选特征：
    1. 使用随机森林计算特征重要性
    2. 使用XGBoost计算特征重要性
    3. 结合两者的重要性选择特征
    """
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. 随机森林特征重要性
    print("   训练随机森林计算特征重要性...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    rf_importance = rf_importance / rf_importance.sum()
    
    # 2. XGBoost特征重要性
    print("   训练XGBoost计算特征重要性...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    xgb_model.fit(X_scaled, y)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    xgb_importance = xgb_importance / xgb_importance.sum()
    
    # 3. 结合两种重要性
    combined_importance = (rf_importance + xgb_importance) / 2
    combined_importance = combined_importance.sort_values(ascending=False)
    
    # 根据阈值选择特征
    if importance_threshold == 'median':
        threshold = combined_importance.median()
    elif importance_threshold == 'mean':
        threshold = combined_importance.mean()
    else:
        threshold = importance_threshold
    
    selected_features = combined_importance[combined_importance > threshold].index.tolist()
    
    print(f"   - 最高重要性特征:")
    for i, (feature, importance) in enumerate(combined_importance.head(10).items()):
        print(f"     {i+1}. {feature}: {importance:.4f}")
    
    return selected_features

# ==============================
# 🆕 为未来数据创建特征（没有目标变量）
# ==============================
def create_future_features(df_future, df_history, target_col='price_day_ahead_avg'):
    """
    为未来数据创建特征（使用历史数据计算统计特征）
    """
    df_future = df_future.copy()
    df_history = df_history.copy()
    
    # 1. 添加时间特征
    df_future = add_time_features(df_future)
    
    # 2. 添加节假日特征
    df_future = add_holiday_features(df_future)
    
    # 3. 从历史数据计算统计特征
    df_history = df_history.sort_values('date_hour')
    
    future_features_list = []
    
    for idx, future_row in df_future.iterrows():
        future_time = future_row['date_hour']
        future_date = pd.to_datetime(future_time).date()
        future_hour = pd.to_datetime(future_time).hour
        
        # 计算基于历史数据的特征
        features = {}
        
        # 历史同期价格
        two_days_ago = future_time - pd.Timedelta(days=2)
        seven_days_ago = future_time - pd.Timedelta(days=7)
        
        # 从历史数据中查找对应的值
        two_days_value = df_history[df_history['date_hour'] == two_days_ago][target_col].values
        seven_days_value = df_history[df_history['date_hour'] == seven_days_ago][target_col].values
        
        features[f'{target_col}_same_hour_2d_ago'] = two_days_value[0] if len(two_days_value) > 0 else np.nan
        features[f'{target_col}_same_hour_7d_ago'] = seven_days_value[0] if len(seven_days_value) > 0 else np.nan
        
        # 计算历史统计特征
        lookback_start = future_time - pd.Timedelta(days=31)
        historical_data = df_history[
            (df_history['date_hour'] >= lookback_start) & 
            (df_history['date_hour'] < future_time - pd.Timedelta(days=1))
        ]
        
        # 相同小时的历史数据
        same_hour_data = historical_data[historical_data['date_hour'].dt.hour == future_hour][target_col]
        
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
        
        # 每日统计数据
        historical_data['date'] = historical_data['date_hour'].dt.date
        daily_stats = historical_data.groupby('date')[target_col].mean()
        
        if len(daily_stats) > 1:
            features[f'{target_col}_daily_mean_2d_ago'] = daily_stats.iloc[-2] if len(daily_stats) >= 2 else np.nan
            features[f'{target_col}_daily_mean_7d_ago'] = daily_stats.iloc[-7] if len(daily_stats) >= 7 else np.nan
            features[f'{target_col}_daily_mean_7d_rolling'] = daily_stats.tail(7).mean()
        else:
            features[f'{target_col}_daily_mean_2d_ago'] = np.nan
            features[f'{target_col}_daily_mean_7d_ago'] = np.nan
            features[f'{target_col}_daily_mean_7d_rolling'] = np.nan
        
        # 价格变化率特征
        if len(historical_data) > 24:
            features[f'{target_col}_hourly_change'] = historical_data[target_col].iloc[-1] - historical_data[target_col].iloc[-2]
            features[f'{target_col}_daily_change'] = historical_data[target_col].iloc[-1] - historical_data[target_col].iloc[-25]
        else:
            features[f'{target_col}_hourly_change'] = np.nan
            features[f'{target_col}_daily_change'] = np.nan
        
        # 关键小时级特征
        if future_hour in [8, 9, 10, 11, 12, 18, 19, 20, 21, 22]:
            features[f'{target_col}_hour{future_hour}_30d_mean'] = same_hour_data.mean() if len(same_hour_data) > 0 else np.nan
        
        future_features_list.append(features)
    
    # 合并特征到未来数据
    future_features_df = pd.DataFrame(future_features_list, index=df_future.index)
    df_future = pd.concat([df_future, future_features_df], axis=1)
    
    return df_future

# ==============================
# 确保特征列为数值型
# ==============================
def ensure_numeric_features(df, feature_columns):
    df = df.copy()
    for col in feature_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"⚠️ 警告：特征列 {col} 不在 DataFrame 中")
    return df

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

# ==============================
# 🆕 一次性调参函数（包含特征筛选）
# ==============================
def tune_all_models_once(engine, tuning_start_date, tuning_end_date, all_features, target_variable='price_day_ahead_avg'):
    print(f"\n🛠️ 一次性调参窗口: {tuning_start_date} → {tuning_end_date}")

    cols_to_query = ['date_hour', target_variable] + raw_features
    query = f"""
        SELECT {', '.join(cols_to_query)}
        FROM anhui_hourly_spot_cleared_wide
        WHERE date_hour >= '{tuning_start_date}' AND date_hour < '{tuning_end_date}'
        ORDER BY date_hour;
    """
    df_tune = pd.read_sql(query, engine)
    df_tune['date_hour'] = pd.to_datetime(df_tune['date_hour']).dt.tz_localize(None)
    
    # 使用新的特征工程函数
    df_tune = create_all_features(df_tune, target_variable)
    df_tune = ensure_numeric_features(df_tune, all_features)

    if len(df_tune) == 0:
        raise ValueError("❌ 调参数据为空！")

    X_tune = df_tune[all_features]
    y_tune = df_tune[target_variable]
    valid_mask = ~pd.isna(y_tune)
    X_tune = X_tune[valid_mask]
    y_tune = y_tune[valid_mask]
    
    # 🆕 执行特征筛选
    selected_features = feature_selection_pipeline(X_tune, y_tune, target_variable)
    
    # 使用筛选后的特征
    X_tune_selected = X_tune[selected_features]
    
    print(f"\n🎯 使用筛选后的 {len(selected_features)} 个特征进行模型调参")
    
    rf_params = tune_rf_params(X_tune_selected, y_tune)
    xgb_params = tune_xgb_params(X_tune_selected, y_tune)
    lgb_params = tune_lgb_params(X_tune_selected, y_tune)

    return {
        'RandomForest': rf_params,
        'XGBoost': xgb_params,
        'LightGBM': lgb_params,
        'selected_features': selected_features  # 🆕 返回筛选后的特征列表
    }

# ==============================
# 调参函数（保持不变，但使用筛选后的特征）
# ==============================
def tune_rf_params(X_train, y_train, n_bayes=8, n_grid=3):
    print("\n🔍 开始调参 RandomForest...")

    def rf_bo(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth) if int(max_depth) > 0 else None,
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            random_state=42,
            n_jobs=-1
        )
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            model.fit(X_tr_scaled, y_tr)
            pred = model.predict(X_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(-rmse)
        return np.mean(scores)

    pbounds = {
        'n_estimators': (50, 300),
        'max_depth': (3, 15),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    }
    optimizer = BayesianOptimization(f=rf_bo, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=2, n_iter=n_bayes)
    best = optimizer.max['params']

    n_est_range = [max(50, int(best['n_estimators']) - 20), int(best['n_estimators']), min(300, int(best['n_estimators']) + 20)]
    max_d_range = [None] if best['max_depth'] < 1 else [
        max(3, int(best['max_depth']) - 1),
        int(best['max_depth']),
        min(15, int(best['max_depth']) + 1)
    ]
    min_ss_range = [max(2, int(best['min_samples_split']) - 2), int(best['min_samples_split']), min(20, int(best['min_samples_split']) + 2)]
    min_sl_range = [max(1, int(best['min_samples_leaf']) - 1), int(best['min_samples_leaf']), min(10, int(best['min_samples_leaf']) + 1)]

    param_grid = {
        'n_estimators': list(set(n_est_range)),
        'max_depth': list(set(max_d_range)),
        'min_samples_split': list(set(min_ss_range)),
        'min_samples_leaf': list(set(min_sl_range))
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_scaled, y_train)
    print(f"✅ RF 最终参数: {grid.best_params_}")
    return grid.best_params_

def tune_xgb_params(X_train, y_train, n_bayes=8):
    print("\n🔍 开始调参 XGBoost...")

    def xgb_bo(learning_rate, max_depth, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda):
        model = XGBRegressor(
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            n_estimators=int(n_estimators),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=0
        )
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            model.fit(X_tr_scaled, y_tr)
            pred = model.predict(X_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(-rmse)
        return np.mean(scores)

    pbounds = {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'n_estimators': (50, 300),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 1)
    }
    optimizer = BayesianOptimization(f=xgb_bo, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=2, n_iter=n_bayes)
    best = optimizer.max['params']

    param_grid = {
        'learning_rate': [best['learning_rate']],
        'max_depth': [int(best['max_depth'])],
        'n_estimators': [int(best['n_estimators'])],
        'subsample': [best['subsample']],
        'colsample_bytree': [best['colsample_bytree']],
        'reg_alpha': [best['reg_alpha']],
        'reg_lambda': [best['reg_lambda']]
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    grid = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_scaled, y_train)
    print(f"✅ XGB 最终参数: {grid.best_params_}")
    return grid.best_params_

def tune_lgb_params(X_train, y_train, n_bayes=8):
    print("\n🔍 开始调参 LightGBM...")

    def lgb_bo(learning_rate, max_depth, n_estimators, num_leaves, subsample, colsample_bytree, reg_alpha, reg_lambda):
        model = LGBMRegressor(
            learning_rate=learning_rate,
            max_depth=int(max_depth),
            n_estimators=int(n_estimators),
            num_leaves=int(num_leaves),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbose=-1
        )
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            model.fit(X_tr_scaled, y_tr)
            pred = model.predict(X_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(-rmse)
        return np.mean(scores)

    pbounds = {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'n_estimators': (50, 300),
        'num_leaves': (20, 100),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'reg_alpha': (0, 1),
        'reg_lambda': (0, 1)
    }
    optimizer = BayesianOptimization(f=lgb_bo, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=2, n_iter=n_bayes)
    best = optimizer.max['params']

    param_grid = {
        'learning_rate': [best['learning_rate']],
        'max_depth': [int(best['max_depth'])],
        'n_estimators': [int(best['n_estimators'])],
        'num_leaves': [int(best['num_leaves'])],
        'subsample': [best['subsample']],
        'colsample_bytree': [best['colsample_bytree']],
        'reg_alpha': [best['reg_alpha']],
        'reg_lambda': [best['reg_lambda']]
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    grid = GridSearchCV(
        LGBMRegressor(random_state=42, verbose=-1),
        param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_scaled, y_train)
    print(f"✅ LGB 最终参数: {grid.best_params_}")
    return grid.best_params_

# ==============================
# Two-Stage Stacking（使用筛选后的特征）
# ==============================
def two_stage_stacking_predict_fixed(X_train, y_train, X_test, model_params, selected_features, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_samples = X_train.shape[0]
    oof_preds = np.zeros((n_samples, 3))
    test_meta = np.zeros((X_test.shape[0], 3))

    models = [
        ('RF', RandomForestRegressor(**model_params['RandomForest'])),
        ('XGB', XGBRegressor(**model_params['XGBoost'])),
        ('LGB', LGBMRegressor(**model_params['LightGBM']))
    ]

    # 确保只使用筛选后的特征
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    for i, (name, model) in enumerate(models):
        oof_single = np.zeros(n_samples)
        test_pred_accum = np.zeros(X_test.shape[0])
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_tr_scaled, y_tr)
            oof_single[val_idx] = model.predict(X_val_scaled)
            test_pred_accum += model.predict(X_test_scaled) / n_splits

        oof_preds[:, i] = oof_single
        test_meta[:, i] = test_pred_accum

    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=3)
    meta_model.fit(oof_preds, y_train)
    pred_stacking = meta_model.predict(test_meta)
    return np.clip(pred_stacking, 0, None)

# ==============================
# 回测评估（使用筛选后的特征）
# ==============================
def backtest_and_evaluate(eval_start_date, eval_end_date, selected_features, model_params, target_variable='price_day_ahead_avg'):
    print(f"\n🔍 开始回测评估: {eval_start_date} → {eval_end_date}")

    with engine.connect() as conn:
        conn.execute(text("""
            DELETE FROM anhui_ahead_price_evaluation_results 
            WHERE date >= :start AND date < :end AND eval_type = 'backtest'
        """), {"start": eval_start_date, "end": eval_end_date})
        conn.commit()

    cols_to_query = ['date_hour', target_variable] + raw_features
    query_hist = f"""
        SELECT {', '.join(cols_to_query)}
        FROM anhui_hourly_spot_cleared_wide
        WHERE date_hour < '{eval_end_date}'
        ORDER BY date_hour;
    """
    df_full = pd.read_sql(query_hist, engine)
    df_full['date_hour'] = pd.to_datetime(df_full['date_hour']).dt.tz_localize(None)
    
    # 创建所有特征
    df_full = create_all_features(df_full, target_variable)
    
    eval_dates = pd.date_range(start=eval_start_date, end=pd.Timestamp(eval_end_date) - pd.Timedelta(days=1), freq='D')
    results = []

    for test_day in eval_dates:
        test_day = pd.Timestamp(test_day).normalize()
        train_start = test_day - pd.Timedelta(days=60)
        train_df = df_full[(df_full['date_hour'] >= train_start) & (df_full['date_hour'] < test_day)].copy()
        test_df = df_full[df_full['date_hour'].dt.normalize() == test_day].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # 确保特征存在
        missing_features = [f for f in selected_features if f not in train_df.columns]
        if missing_features:
            # 为缺失的特征填充NaN
            for f in missing_features:
                train_df[f] = np.nan
                test_df[f] = np.nan

        train_df_clean = ensure_numeric_features(train_df, selected_features)
        test_df_clean = ensure_numeric_features(test_df, selected_features)

        X_train = train_df_clean[selected_features]
        y_train = train_df_clean[target_variable]
        X_test = test_df_clean[selected_features]
        y_test = test_df_clean[target_variable]

        valid_mask = ~pd.isna(y_train)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        # 检查是否有足够的数据
        if len(X_train) < 24:  # 至少需要一天的数据
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf = RandomForestRegressor(**model_params['RandomForest']).fit(X_train_scaled, y_train)
        xgb = XGBRegressor(**model_params['XGBoost']).fit(X_train_scaled, y_train)
        lgb = LGBMRegressor(**model_params['LightGBM']).fit(X_train_scaled, y_train)

        pred_rf = rf.predict(X_test_scaled)
        pred_xgb = xgb.predict(X_test_scaled)
        pred_lgb = lgb.predict(X_test_scaled)
        pred_stacking = two_stage_stacking_predict_fixed(X_train, y_train, X_test, model_params, selected_features, n_splits=3)

        strategies = {
            'RandomForest': pred_rf,
            'XGBoost': pred_xgb,
            'LightGBM': pred_lgb,
            'TwoStage_Stacking': pred_stacking
        }

        for name, y_pred in strategies.items():
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = smape(y_test, y_pred)
            results.append({
                'date': test_day,
                'strategy': name,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'eval_type': 'backtest'
            })

    if results:
        eval_df = pd.DataFrame(results)
        eval_df.to_sql('anhui_ahead_price_evaluation_results', con=engine, if_exists='append', index=False)
        print(f"✅ 回测评估完成，共 {len(eval_df)} 条记录写入表")
    else:
        print("⚠️ 无有效回测数据")

# ==============================
# 在线评估（不变）
# ==============================
def evaluate_existing_predictions():
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    if 'anhui_ahead_price_predictions' not in table_names:
        print("ℹ️ 表 'anhui_ahead_price_predictions' 不存在，跳过在线评估。")
        return

    print("\n🔄 执行在线评估...")

    query = """
        SELECT 
            p.date_hour,
            p.pred_two_stage_stacking AS pred,
            d.price_day_ahead_avg AS actual
        FROM anhui_ahead_price_predictions p
        JOIN anhui_hourly_spot_cleared_wide d ON p.date_hour = d.date_hour
        LEFT JOIN anhui_ahead_price_evaluation_results e 
            ON p.date_hour::date = e.date AND e.eval_type = 'online'
        WHERE e.date IS NULL;
    """
    df_compare = pd.read_sql(query, engine)
    if df_compare.empty:
        print("ℹ️ 无新可评估的预测记录")
        return

    df_compare['date'] = df_compare['date_hour'].dt.normalize()

    def agg_metrics(group):
        y_true = group['actual'].values
        y_pred = group['pred'].values
        mask = ~pd.isna(y_true) & ~pd.isna(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) == 0:
            return pd.Series({'rmse': np.nan, 'mape': np.nan, 'r2': np.nan})
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = smape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return pd.Series({'rmse': rmse, 'mape': mape, 'r2': r2})

    daily_metrics = df_compare.groupby('date').apply(agg_metrics).reset_index()
    daily_metrics = daily_metrics.dropna(subset=['rmse'])
    if daily_metrics.empty:
        print("ℹ️ 所有预测日均无可评估数据")
        return

    daily_metrics['strategy'] = 'TwoStage_Stacking'
    daily_metrics['eval_type'] = 'online'
    daily_metrics.to_sql('anhui_ahead_price_evaluation_results', con=engine, if_exists='append', index=False)
    print(f"✅ 在线评估完成，新增 {len(daily_metrics)} 条记录")

# ==============================
# 主预测函数（使用筛选后的特征）
# ==============================
def predict_and_save_to_db(future_start_date, future_end_date, selected_features, model_params, target_variable='price_day_ahead_avg'):
    print(f"📥 从数据库加载历史数据...")
    cols_hist = ['date_hour', target_variable] + raw_features
    query_hist = f"""
        SELECT {', '.join(cols_hist)}
        FROM anhui_hourly_spot_cleared_wide
        WHERE date_hour < '{future_start_date}'
        ORDER BY date_hour;
    """
    df_hist = pd.read_sql(query_hist, engine)
    df_hist['date_hour'] = pd.to_datetime(df_hist['date_hour']).dt.tz_localize(None)
    df_hist = df_hist.dropna(subset=[target_variable]).reset_index(drop=True)
    
    # 创建所有特征
    df_hist = create_all_features(df_hist, target_variable)

    cols_future = ['date_hour'] + raw_features
    query_future = f"""
        SELECT {', '.join(cols_future)}
        FROM anhui_hourly_spot_cleared_wide
        WHERE date_hour >= '{future_start_date}' AND date_hour < '{future_end_date}'
        ORDER BY date_hour;
    """
    df_future_raw = pd.read_sql(query_future, engine)
    if df_future_raw.empty:
        print("⚠️ 未找到未来特征数据")
        return

    df_future_raw['date_hour'] = pd.to_datetime(df_future_raw['date_hour']).dt.tz_localize(None)
    
    # 为未来数据创建特征
    df_future = create_future_features(df_future_raw, df_hist, target_variable)
    
    future_dates = sorted(df_future['date_hour'].dt.normalize().unique())

    print(f"📅 共需预测 {len(future_dates)} 天: {future_dates[0]} → {future_dates[-1]}")

    all_predictions = []

    # 获取最新最优策略
    best_strategy = get_best_model_from_evaluation_results(engine, metric='rmse', eval_type='backtest')
    best_pred_col = STRATEGY_TO_PRED_COL.get(best_strategy, 'pred_lightgbm')

    for idx, test_day in enumerate(future_dates):
        test_day = pd.Timestamp(test_day).normalize()
        print(f"\n🔄 预测日期: {test_day.strftime('%Y-%m-%d')}")

        train_start = test_day - pd.Timedelta(days=60)
        train_df = df_hist[(df_hist['date_hour'] >= train_start) & (df_hist['date_hour'] < test_day)].copy()
        test_df = df_future[df_future['date_hour'].dt.normalize() == test_day].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            print("⚠️ 数据不足，跳过")
            continue

        # 确保筛选的特征存在
        missing_features = [f for f in selected_features if f not in train_df.columns]
        if missing_features:
            for f in missing_features:
                train_df[f] = np.nan
                test_df[f] = np.nan

        train_df_clean = ensure_numeric_features(train_df, selected_features)
        test_df_clean = ensure_numeric_features(test_df, selected_features)

        X_train = train_df_clean[selected_features]
        y_train = train_df_clean[target_variable]
        X_test = test_df_clean[selected_features]

        valid_mask = ~pd.isna(y_train)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        if len(X_train) < 24:
            print("⚠️ 训练数据不足，跳过")
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf = RandomForestRegressor(**model_params['RandomForest']).fit(X_train_scaled, y_train)
        xgb = XGBRegressor(**model_params['XGBoost']).fit(X_train_scaled, y_train)
        lgb = LGBMRegressor(**model_params['LightGBM']).fit(X_train_scaled, y_train)

        pred_rf = rf.predict(X_test_scaled)
        pred_xgb = xgb.predict(X_test_scaled)
        pred_lgb = lgb.predict(X_test_scaled)
        pred_stacking = two_stage_stacking_predict_fixed(X_train, y_train, X_test, model_params, selected_features, n_splits=3)

        daily_result = test_df_clean[['date_hour']].copy()
        daily_result['pred_randomforest'] = pred_rf
        daily_result['pred_xgboost'] = pred_xgb
        daily_result['pred_lightgbm'] = pred_lgb
        daily_result['pred_two_stage_stacking'] = pred_stacking
        daily_result['pred_best'] = daily_result[best_pred_col]

        all_predictions.append(daily_result)

    if all_predictions:
        final_pred = pd.concat(all_predictions, ignore_index=True)
        final_pred['created_at'] = pd.Timestamp.now()
        final_pred.to_sql('anhui_ahead_price_predictions', con=engine, if_exists='append', index=False, method='multi')
        print(f"\n✅ 成功写入 {len(final_pred)} 条预测结果到 PostgreSQL 表 'anhui_ahead_price_predictions'")
    else:
        print("❌ 无有效预测结果，未写入数据库")

# ==============================
# 辅助函数（不变）
# ==============================
def get_best_model_from_evaluation_results(engine, metric='rmse', eval_type='backtest'):
    query = f"""
        SELECT strategy, AVG({metric}) as avg_metric
        FROM anhui_ahead_price_evaluation_results
        WHERE eval_type = '{eval_type}'
        GROUP BY strategy
        ORDER BY avg_metric ASC
        LIMIT 1;
    """
    try:
        df_best = pd.read_sql(query, engine)
        if not df_best.empty:
            best_strategy = df_best.iloc[0]['strategy']
            print(f"🏆 自动选择最优模型: {best_strategy} (基于 {eval_type} 的平均 {metric})")
            return best_strategy
        else:
            print("⚠️ 无评估记录，回退到默认模型: LightGBM")
            return "LightGBM"
    except Exception as e:
        print(f"❌ 查询最优模型失败: {e}，回退到默认模型: LightGBM")
        return "LightGBM"

STRATEGY_TO_PRED_COL = {
    'RandomForest': 'pred_randomforest',
    'XGBoost': 'pred_xgboost',
    'LightGBM': 'pred_lightgbm',
    'TwoStage_Stacking': 'pred_two_stage_stacking'
}

def create_evaluation_table(engine):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS anhui_ahead_price_evaluation_results (
        date DATE,
        strategy TEXT,
        rmse FLOAT,
        mape FLOAT,
        r2 FLOAT,
        eval_type TEXT
    );
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
    print("✅ 确保评估结果表已存在")

# ==============================
# 🆕 通用评估函数
# ==============================
def evaluate_prediction_column(pred_start='2025-12-31', pred_end='2026-01-06', pred_col='pred_lightgbm'):
    print(f"\n🔍 评估预测列 '{pred_col}' ({pred_start} → {pred_end})...")

    query = f"""
        SELECT 
            p.date_hour,
            p.{pred_col} AS pred,
            d.price_day_ahead_avg AS actual
        FROM anhui_ahead_price_predictions p
        JOIN anhui_hourly_spot_cleared_wide d 
            ON p.date_hour = d.date_hour
        WHERE p.date_hour >= '{pred_start}' 
          AND p.date_hour < '{pred_end}'
          AND d.price_day_ahead_avg IS NOT NULL
          AND p.{pred_col} IS NOT NULL;
    """
    df_eval = pd.read_sql(query, engine)
    if df_eval.empty:
        print(f"⚠️ 无有效数据用于评估列 '{pred_col}'")
        return None

    y_true = df_eval['actual'].values
    y_pred = df_eval['pred'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"🎯 {pred_col} | RMSE: {rmse:.4f}, SMAPE: {smape_val:.4f}%, R²: {r2:.4f}")
    return {'column': pred_col, 'rmse': rmse, 'smape': smape_val, 'r2': r2}

# ==============================
# 主程序入口 —— 已更新为包含特征筛选
# ==============================
if __name__ == "__main__":
    # 🆕 获取包含所有特征的完整列表（筛选前）
    all_features = get_all_features(base_features)
    print(f"✅ 所有特征数量（筛选前）: {len(all_features)} 个")
    
    create_evaluation_table(engine)

    print("🗑️  删除旧预测表以确保列名统一...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS anhui_ahead_price_predictions"))
        conn.commit()

    #----------------------------------------------------------------------------------------------------------------------------------- 🛠️ 一次性调参（包含特征筛选）
    tuning_start = '2025-06-30'
    tuning_end = '2025-12-31'
    tuning_result = tune_all_models_once(engine, tuning_start, tuning_end, all_features)
    
    # 从调参结果中提取模型参数和筛选后的特征
    model_params = {
        'RandomForest': tuning_result['RandomForest'],
        'XGBoost': tuning_result['XGBoost'],
        'LightGBM': tuning_result['LightGBM']
    }
    selected_features = tuning_result['selected_features']
    
    print(f"\n🎯 最终使用 {len(selected_features)} 个筛选后的特征")
    print("📋 筛选后的特征列表:")
    for i, feature in enumerate(selected_features[:20]):  # 只显示前20个
        print(f"  {i+1}. {feature}")
    if len(selected_features) > 20:
        print(f"  ... 和 {len(selected_features) - 20} 个其他特征")

    #-----------------------------------------------------------------------------------------------------------------------------------  1. 回测评估（使用筛选后的特征）
    backtest_and_evaluate(
        eval_start_date='2025-11-17',
        eval_end_date='2026-01-24',
        selected_features=selected_features,
        model_params=model_params
    )

    #-----------------------------------------------------------------------------------------------------------------------------------  2. 预测未来（使用筛选后的特征）
    predict_and_save_to_db(
        future_start_date='2026-01-25',
        future_end_date='2026-02-04',
        selected_features=selected_features,
        model_params=model_params
    )

    # 3. 在线评估（历史未评估的）
    evaluate_existing_predictions()

    # 4. 分别评估各模型列的真实线上表现
    print("\n" + "="*60)
    print("📊 分别评估各模型列的线上表现（2026-01-25 → 2026-02-04）")
    print("="*60)
    
    columns_to_evaluate = [
        'pred_lightgbm',
        'pred_xgboost',
        'pred_randomforest',
        'pred_two_stage_stacking'
    ]
    
    eval_results = []
    for col in columns_to_evaluate:
        result = evaluate_prediction_column('2026-01-25', '2026-02-04', pred_col=col)
        if result:
            eval_results.append(result)
    
    if eval_results:
        df_compare = pd.DataFrame(eval_results)
        print("\n📈 模型线上表现对比:")
        print(df_compare.round(4).to_string(index=False))
    else:
        print("⚠️ 无有效模型列可评估")

    # 5. 最终汇总报告（回测 + 在线）
    report_query = """
        SELECT 
            strategy,
            eval_type,
            AVG(rmse) AS avg_rmse,
            AVG(mape) AS avg_mape,
            AVG(r2) AS avg_r2,
            COUNT(*) AS n_days
        FROM anhui_ahead_price_evaluation_results
        GROUP BY strategy, eval_type
        ORDER BY eval_type, avg_rmse;
    """
    report = pd.read_sql(report_query, engine)
    print("\n📊 最新模型评估报告（回测 + 在线）:")
    print(report.round(4).to_string(index=False))