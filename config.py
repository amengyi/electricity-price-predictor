DB_CONFIG = {
    'host': 'pgm-bp1q60i1ca0xnwv98o.pg.rds.aliyuncs.com',
    'port': 1921,
    'database': 'bopha',
    'user': 'seniverse',
    'password': 'mpr3uOFnnM5bHtl'
}

RAW_FEATURES = [
    'total_wind_solar_pre_avg',
    'solar_radiation_avg',
    'cloud_avg',
    'dir_avg',
    'heat_avg',
    'hum_avg',
    'pm10_avg',
    'pm2_5_avg',
    'pre_avg',
    'tem_avg',
    'wind_speed_avg',
    'wind10_avg',
    'wind100_avg',
    'wind120_avg',
    'wind180_avg',
    'load_pre_avg',
    'out_pre_avg',
    'non_market_pre_avg',
    'hydro_pre_avg',
    'power_pre_avg',
    'balance_pre_avg'
]

TARGET_COLUMN = 'price_day_ahead_avg'