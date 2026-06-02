import os

def get_db_config():
    """从环境变量获取数据库配置"""
    config = {
        'host': os.environ.get('DB_HOST', 'pgm-bp1q60i1ca0xnwv98o.pg.rds.aliyuncs.com'),
        'port': os.environ.get('DB_PORT', '1921'),
        'database': os.environ.get('DB_NAME', 'bopha'),
        'user': os.environ.get('DB_USER', 'seniverse'),
        'password': os.environ.get('DB_PASSWORD', '')
    }
    
    # Streamlit Cloud 使用 st.secrets 的特殊格式
    if 'DB_PASSWORD' not in os.environ:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'DB_PASSWORD' in st.secrets:
                config['password'] = st.secrets['DB_PASSWORD']
                config['user'] = st.secrets.get('DB_USER', config['user'])
                config['host'] = st.secrets.get('DB_HOST', config['host'])
                config['port'] = st.secrets.get('DB_PORT', config['port'])
                config['database'] = st.secrets.get('DB_NAME', config['database'])
        except:
            pass
    
    return config

def get_source_db_config():
    """从环境变量获取源数据库配置"""
    config = {
        'host': os.environ.get('SOURCE_DB_HOST', 'rm-2ze5i2i2u4x7n3q1.mysql.rds.aliyuncs.com'),
        'port': os.environ.get('SOURCE_DB_PORT', '3306'),
        'database': os.environ.get('SOURCE_DB_NAME', 'bopha_anhui'),
        'user': os.environ.get('SOURCE_DB_USER', 'seniverse'),
        'password': os.environ.get('SOURCE_DB_PASSWORD', '')
    }
    
    # Streamlit Cloud 使用 st.secrets
    if 'SOURCE_DB_PASSWORD' not in os.environ:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'SOURCE_DB_PASSWORD' in st.secrets:
                config['password'] = st.secrets['SOURCE_DB_PASSWORD']
                config['user'] = st.secrets.get('SOURCE_DB_USER', config['user'])
                config['host'] = st.secrets.get('SOURCE_DB_HOST', config['host'])
                config['port'] = st.secrets.get('SOURCE_DB_PORT', config['port'])
                config['database'] = st.secrets.get('SOURCE_DB_NAME', config['database'])
        except:
            pass
    
    return config

DB_CONFIG = get_db_config()
SOURCE_DB_CONFIG = get_source_db_config()
