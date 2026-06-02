import os

def get_db_config():
    """从环境变量获取数据库配置"""
    return {
        'host': os.environ.get('DB_HOST', 'pgm-bp1q60i1ca0xnwv98o.pg.rds.aliyuncs.com'),
        'port': os.environ.get('DB_PORT', '1921'),
        'database': os.environ.get('DB_NAME', 'bopha'),
        'user': os.environ.get('DB_USER', 'seniverse'),
        'password': os.environ.get('DB_PASSWORD', '')
    }

def get_source_db_config():
    """从环境变量获取源数据库配置"""
    return {
        'host': os.environ.get('SOURCE_DB_HOST', 'rm-2ze5i2i2u4x7n3q1.mysql.rds.aliyuncs.com'),
        'port': os.environ.get('SOURCE_DB_PORT', '3306'),
        'database': os.environ.get('SOURCE_DB_NAME', 'bopha_anhui'),
        'user': os.environ.get('SOURCE_DB_USER', 'seniverse'),
        'password': os.environ.get('SOURCE_DB_PASSWORD', '')
    }

DB_CONFIG = get_db_config()
SOURCE_DB_CONFIG = get_source_db_config()
