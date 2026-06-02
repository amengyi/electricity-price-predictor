import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO, BytesIO

from data_processor import (
    load_sample_data, load_from_database, create_all_features, clean_data, split_data, 
    create_future_features, ensure_numeric_features
)
from models import (
    TimeSeriesModel, evaluate_model, hyperparameter_tuning, 
    get_available_models, get_model_name
)
from config import DB_CONFIG

st.set_page_config(page_title="电价预测系统", layout="wide")

if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'trained' not in st.session_state:
    st.session_state['trained'] = False
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'evaluation_results' not in st.session_state:
    st.session_state['evaluation_results'] = {}

def load_data(data_source='sample'):
    with st.spinner('正在加载数据...'):
        if data_source == 'database':
            try:
                df = load_from_database(DB_CONFIG)
            except Exception as e:
                st.error(f"数据库连接失败: {str(e)}")
                return
        else:
            df = load_sample_data()
        
        df = clean_data(df)
        df = create_all_features(df)
        st.session_state['data'] = df
        st.session_state['current_data_source'] = data_source
        st.success(f"数据加载完成！共 {len(df)} 条记录")

class EnsembleModel:
    def __init__(self, model_results, method='simpleavg'):
        self.model_results = model_results
        self.method = method
        self.models = [m['model'] for m in model_results]
        self.n_models = len(self.models)
    
    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.method == 'simpleavg':
            return predictions.mean(axis=0)
        elif self.method == 'weighted':
            weights = np.array([1/(m['metrics']['RMSE'] + 1e-6) for m in self.model_results])
            weights = weights / weights.sum()
            if len(weights) != predictions.shape[0]:
                weights = weights[:predictions.shape[0]]
                weights = weights / weights.sum()
            return (predictions * weights.reshape(-1, 1)).sum(axis=0)
        elif self.method == 'median':
            return np.median(predictions, axis=0)
        elif self.method == 'stack':
            meta_features = predictions.T
            from sklearn.linear_model import LinearRegression
            meta_model = LinearRegression()
            meta_model.fit(meta_features, self.y_train_meta)
            return meta_model.predict(meta_features)
        else:
            return predictions.mean(axis=0)
    
    def fit(self, X_train, y_train):
        self.y_train_meta = y_train
    
    def get_feature_importance(self, feature_names):
        importances = []
        for m in self.models:
            if hasattr(m, 'get_feature_importance'):
                imp = m.get_feature_importance(feature_names)
                if imp is not None:
                    importances.append(imp)
        
        if not importances:
            return None
        
        avg_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        return avg_importance

def train_model(model_type, params):
    if st.session_state['data'] is None:
        st.error("请先加载数据！")
        return
    
    df = st.session_state['data']
    X_train, X_test, y_train, y_test, train_df, test_df = split_data(df, train_ratio=0.8)
    
    if model_type == 'auto':
        with st.spinner('🤖 正在训练所有模型并选择最佳模型...'):
            available_models = get_available_models()
            results = []
            
            for m_type in available_models:
                try:
                    model = TimeSeriesModel(model_type=m_type)
                    model.fit(X_train, y_train)
                    test_pred = model.predict(X_test)
                    train_pred = model.predict(X_train)
                    test_metrics = evaluate_model(y_test, test_pred)
                    results.append({
                        'model_type': m_type,
                        'model': model,
                        'metrics': test_metrics,
                        'train_pred': train_pred,
                        'test_pred': test_pred
                    })
                    st.write(f"✓ {get_model_name(m_type)} - RMSE: {test_metrics['RMSE']:.2f}")
                except Exception as e:
                    st.write(f"✗ {get_model_name(m_type)} - 训练失败: {str(e)}")
            
            ensemble_methods = [
                {'name': 'simpleavg', 'label': '简单平均', 'desc': '所有模型预测的算术平均'},
                {'name': 'weighted', 'label': '加权平均', 'desc': '按RMSE倒数加权'},
            ]
            
            for method in ensemble_methods:
                try:
                    ensemble_model = EnsembleModel(results, method=method['name'])
                    ensemble_model.fit(X_train, y_train)
                    ensemble_pred = ensemble_model.predict(X_test)
                    ensemble_metrics = evaluate_model(y_test, ensemble_pred)
                    results.append({
                        'model_type': f'ensemble_{method["name"]}',
                        'model': ensemble_model,
                        'metrics': ensemble_metrics,
                        'method': method
                    })
                    st.write(f"✓ {method['label']} - RMSE: {ensemble_metrics['RMSE']:.2f}")
                except Exception as e:
                    st.write(f"✗ {method['label']} - 训练失败: {str(e)}")
            
            best_result = min(results, key=lambda x: x['metrics']['RMSE'])
            model = best_result['model']
            test_metrics = best_result['metrics']
            
            if 'method' in best_result:
                model_type_name = best_result['method']['label']
            else:
                model_type_name = get_model_name(best_result['model_type'])
            
            train_pred = model.predict(X_train)
            train_metrics = evaluate_model(y_train, train_pred)
            
            st.session_state['model'] = model
            st.session_state['trained'] = True
            st.session_state['evaluation_results'] = {
                'train': train_metrics,
                'test': test_metrics,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_pred': train_pred,
                'test_pred': model.predict(X_test),
                'train_df': train_df,
                'test_df': test_df,
                'model_type': best_result['model_type'],
                'model_type_name': model_type_name,
                'all_models': results
            }
            
            st.success(f"🎉 最佳模型: {model_type_name}\n训练集 RMSE: {train_metrics['RMSE']:.2f}\n测试集 RMSE: {test_metrics['RMSE']:.2f}")
    else:
        with st.spinner(f'正在训练 {get_model_name(model_type)}...'):
            model = TimeSeriesModel(model_type=model_type, params=params)
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_metrics = evaluate_model(y_train, train_pred)
            test_metrics = evaluate_model(y_test, test_pred)
            
            st.session_state['model'] = model
            st.session_state['trained'] = True
            st.session_state['evaluation_results'] = {
                'train': train_metrics,
                'test': test_metrics,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_pred': train_pred,
                'test_pred': test_pred,
                'train_df': train_df,
                'test_df': test_df,
                'model_type': model_type
            }
            
            st.success(f"模型训练完成！\n训练集 RMSE: {train_metrics['RMSE']:.2f}\n测试集 RMSE: {test_metrics['RMSE']:.2f}")

def predict_future(days=7):
    if not st.session_state['trained']:
        st.error("请先训练模型！")
        return
    
    df = st.session_state['data']
    model = st.session_state['model']
    
    last_date = df['date_hour'].max()
    future_end = last_date + timedelta(days=days)
    future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=days*24, freq='h')
    
    try:
        df_future_raw = load_from_database(DB_CONFIG, start_date=last_date + timedelta(hours=1), end_date=future_end)
        if len(df_future_raw) == 0:
            raise ValueError("数据库中没有未来数据")
        st.info(f"从数据库加载了 {len(df_future_raw)} 条未来特征数据")
    except Exception as e:
        st.warning(f"无法从数据库获取未来数据: {str(e)}，使用模拟数据")
        df_future_raw = pd.DataFrame({
            'date_hour': future_dates,
            'total_wind_solar_pre_avg': np.random.normal(1000, 200, len(future_dates)),
            'solar_radiation_avg': np.random.normal(500, 150, len(future_dates)),
            'cloud_avg': np.random.normal(50, 20, len(future_dates)),
            'pre_avg': np.random.normal(1013, 10, len(future_dates)),
            'tem_avg': np.random.normal(15, 10, len(future_dates)),
            'wind_speed_avg': np.random.normal(5, 3, len(future_dates)),
            'hum_avg': np.random.normal(60, 20, len(future_dates)),
            'load_pre_avg': np.random.normal(5000, 1000, len(future_dates)),
            'out_pre_avg': np.random.normal(1000, 200, len(future_dates)),
            'non_market_pre_avg': np.random.normal(500, 100, len(future_dates)),
            'hydro_pre_avg': np.random.normal(800, 150, len(future_dates)),
            'power_pre_avg': np.random.normal(7000, 1000, len(future_dates)),
            'balance_pre_avg': np.random.normal(200, 100, len(future_dates))
        })
    
    df_future = create_future_features(df_future_raw, df)
    
    feature_cols = st.session_state['evaluation_results']['X_train'].columns.tolist()
    df_future = ensure_numeric_features(df_future, feature_cols)
    
    X_future = df_future[feature_cols].fillna(0)
    
    with st.spinner('正在进行预测...'):
        predictions = model.predict(X_future)
        predictions = np.clip(predictions, 0, None)
        
        result_df = pd.DataFrame({
            'date_hour': df_future['date_hour'],
            'predicted_price': predictions,
            'day': df_future['date_hour'].dt.date,
            'hour': df_future['date_hour'].dt.hour
        })
        
        df_future_raw['date'] = df_future_raw['date_hour'].dt.date
        feature_availability = []
        key_features = ['tem_avg', 'hum_avg', 'wind_speed_avg', 'cloud_avg',
                       'load_pre_avg', 'total_wind_solar_pre_avg', 'hydro_pre_avg',
                       'solar_radiation_avg', 'pre_avg', 'out_pre_avg', 
                       'non_market_pre_avg', 'power_pre_avg', 'balance_pre_avg']
        
        for date in df_future_raw['date'].unique():
            day_data = df_future_raw[df_future_raw['date'] == date]
            availability = {'日期': date}
            for feat in key_features:
                if feat in day_data.columns:
                    has_data = day_data[feat].notna().sum() > 0
                    availability[feat] = '✓' if has_data else '✗'
                else:
                    availability[feat] = '✗'
            feature_availability.append(availability)
        
        feature_availability_df = pd.DataFrame(feature_availability)
        feature_availability_df.set_index('日期', inplace=True)
        
        feature_name_map = {
            'tem_avg': '温度',
            'hum_avg': '湿度',
            'wind_speed_avg': '风速',
            'cloud_avg': '云量',
            'load_pre_avg': '负荷预测',
            'total_wind_solar_pre_avg': '风光预测',
            'hydro_pre_avg': '水电预测',
            'solar_radiation_avg': '太阳辐射',
            'pre_avg': '气压',
            'out_pre_avg': '外购电预测',
            'non_market_pre_avg': '非市场化电量',
            'power_pre_avg': '发电预测',
            'balance_pre_avg': '平衡预测'
        }
        feature_availability_df.columns = [feature_name_map.get(col, col) for col in feature_availability_df.columns]
        
        st.session_state['predictions'] = result_df
        st.session_state['feature_availability'] = feature_availability_df
        st.session_state['feature_raw'] = df_future_raw
        st.success(f"预测完成！共预测未来 {len(result_df)} 小时")

st.title("⚡ 电价预测系统")

tab1, tab2, tab3, tab4 = st.tabs(["数据概览", "模型训练", "预测结果", "性能评估"])

with tab1:
    st.subheader("数据概览")
    
    with st.form(key='load_data_form'):
        current_source = st.session_state.get('current_data_source', 'sample')
        data_source = st.selectbox(
            "选择数据源", 
            ['sample', 'database'], 
            format_func=lambda x: '示例数据' if x == 'sample' else '真实数据库',
            index=0 if current_source == 'sample' else 1
        )
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            submit_load = st.form_submit_button("加载数据")
        with col2:
            reset_data = st.form_submit_button("重置数据")
        with col3:
            sync_data = st.form_submit_button("🔄 同步最新数据")
    
    if submit_load:
        load_data(data_source)
    if reset_data:
        st.session_state['data'] = None
        st.session_state['trained'] = False
        st.session_state['predictions'] = None
        st.session_state['current_data_source'] = None
        st.rerun()
    if sync_data:
        with st.spinner('正在同步数据库数据...'):
            import subprocess
            result = subprocess.run(['python', '/Users/mengjiantai/Desktop/工具/forecast_system/中间表.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                st.success("数据同步成功！")
                st.code(result.stdout)
            else:
                st.error(f"同步失败: {result.stderr}")
    
    if st.session_state['data'] is not None:
        df = st.session_state['data']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("数据记录数", f"{len(df):,}")
        col2.metric("时间范围", f"{df['date_hour'].min().strftime('%Y-%m-%d')} 至 {df['date_hour'].max().strftime('%Y-%m-%d')}")
        col3.metric("平均电价", f"{df['price_day_ahead_avg'].mean():.2f} 元/MWh")
        
        st.subheader("数据预览")
        st.dataframe(df[['date_hour', 'price_day_ahead_avg', 'total_wind_solar_pre_avg', 'load_pre_avg', 'tem_avg']].head(20))
        
        st.subheader("电价趋势")
        min_date = df['date_hour'].min().to_pydatetime()
        max_date = df['date_hour'].max().to_pydatetime()
        default_end = (min_date + timedelta(days=30)) if (min_date + timedelta(days=30)) < max_date else max_date
        time_range = st.slider("选择时间范围", min_date, max_date, (min_date, default_end))
        filtered_df = df[(df['date_hour'] >= pd.Timestamp(time_range[0])) & (df['date_hour'] <= pd.Timestamp(time_range[1]))]
        
        fig = px.line(filtered_df, x='date_hour', y='price_day_ahead_avg', title='电价变化趋势')
        fig.update_layout(xaxis_title='时间', yaxis_title='电价 (元/MWh)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("电价统计分布")
        fig = px.histogram(df, x='price_day_ahead_avg', nbins=50, title='电价分布直方图')
        fig.update_layout(xaxis_title='电价 (元/MWh)', yaxis_title='频率')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("特征相关性")
        corr_cols = ['price_day_ahead_avg', 'total_wind_solar_pre_avg', 'load_pre_avg', 
                    'tem_avg', 'wind_speed_avg', 'hum_avg']
        corr_matrix = df[corr_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_cols,
            y=corr_cols,
            colorscale='RdBu_r',
            zmin=-1, zmax=1
        ))
        fig.update_layout(title='特征相关性矩阵', xaxis_title='特征', yaxis_title='特征')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("模型训练")
    
    available_models = get_available_models()
    model_options = ['auto'] + available_models
    model_labels = {
        'auto': '🤖 自动选择最佳模型',
        'random_forest': '🌲 随机森林 - 集成学习，适合非线性数据',
        'ridge': '📐 岭回归 - 线性模型，处理多重共线性',
        'xgboost': '🚀 XGBoost - 梯度提升树，高性能',
        'lightgbm': '⚡ LightGBM - 轻量级梯度提升树'
    }
    model_type = st.selectbox("选择模型", model_options, format_func=lambda x: model_labels.get(x, x), key='model_select')
    
    params = {}
    if model_type != 'auto':
        if model_type == 'random_forest':
            params['n_estimators'] = st.slider("决策树数量", 50, 300, 100, 10, key='rf_n_estimators')
            params['max_depth'] = st.slider("最大深度", 3, 20, 10, 1, key='rf_max_depth')
            params['min_samples_split'] = st.slider("最小分割样本数", 2, 20, 2, 1, key='rf_min_samples')
        elif model_type == 'ridge':
            params['alpha'] = st.slider("正则化系数", 0.001, 1000.0, 1.0, 0.001, key='ridge_alpha')
        elif model_type == 'xgboost':
            params['n_estimators'] = st.slider("树数量", 50, 200, 100, 10, key='xgb_n_estimators')
            params['max_depth'] = st.slider("最大深度", 3, 10, 6, 1, key='xgb_max_depth')
            params['learning_rate'] = st.slider("学习率", 0.01, 0.3, 0.1, 0.01, key='xgb_lr')
        elif model_type == 'lightgbm':
            params['n_estimators'] = st.slider("树数量", 50, 200, 100, 10, key='lgb_n_estimators')
            params['max_depth'] = st.slider("最大深度", 3, 10, 6, 1, key='lgb_max_depth')
            params['learning_rate'] = st.slider("学习率", 0.01, 0.3, 0.1, 0.01, key='lgb_lr')
    
    if model_type == 'auto':
        st.info("✅ 自动模式将训练所有可用模型（随机森林、岭回归、XGBoost、LightGBM）及集成方法（简单平均、加权平均），选择效果最好的模型进行预测")
    
    with st.form(key='train_form'):
        submit_train = st.form_submit_button("开始训练")
    
    if submit_train:
        train_model(model_type, params)

with tab3:
    st.subheader("预测结果")
    
    if st.session_state['trained']:
        with st.form(key='predict_form'):
            days = st.slider("预测天数", 1, 14, 7, key='predict_days')
            submit_predict = st.form_submit_button("开始预测")
        
        if submit_predict:
            predict_future(days)
        
        if st.session_state['predictions'] is not None:
            pred_df = st.session_state['predictions']
            
            st.subheader("预测特征说明")
            feature_info = """
            **预测使用的特征（按类型分类）：**
            
            📅 **时间特征**（始终可用）
            - 小时、星期几、月份、季节
            - 是否工作日、是否节假日
            
            🌤️ **天气特征**（通常较长时间可用）
            - 温度、湿度、风速、气压
            - 云量、太阳辐射
            
            🔄 **电力系统特征**（可能仅短期可用）
            - 负荷预测值
            - 风光出力预测
            - 水电预测、非市场化电量预测
            
            💡 **历史统计特征**
            - 历史同期电价均值
            - 滚动统计特征（均值、标准差）
            """
            st.markdown(feature_info)
            
            if 'feature_availability' in st.session_state:
                st.subheader("每日特征可用性")
                st.info("✓ 表示该特征有数据，✗ 表示该特征无数据（使用默认值填充）")
                st.dataframe(st.session_state['feature_availability'])
            
            st.subheader("预测结果预览")
            st.dataframe(pred_df[['date_hour', 'predicted_price']].head(24))
            
            st.subheader("预测趋势图")
            fig = px.line(pred_df, x='date_hour', y='predicted_price', title=f'未来{days}天电价预测')
            fig.update_layout(xaxis_title='时间', yaxis_title='预测电价 (元/MWh)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("按日期统计")
            daily_stats = pred_df.groupby('day')['predicted_price'].agg(['mean', 'min', 'max']).reset_index()
            daily_stats.columns = ['日期', '日均电价', '最低电价', '最高电价']
            st.dataframe(daily_stats.style.format({
                '日均电价': '{:.2f}',
                '最低电价': '{:.2f}',
                '最高电价': '{:.2f}'
            }))
            
            fig = px.bar(daily_stats, x='日期', y=['最低电价', '日均电价', '最高电价'], 
                        title='每日电价统计')
            fig.update_layout(yaxis_title='电价 (元/MWh)')
            st.plotly_chart(fig, use_container_width=True)
            
            csv_data = pred_df.to_csv(index=False)
            st.download_button(
                label="下载预测结果",
                data=csv_data,
                file_name=f'electricity_price_prediction_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )
    else:
        st.info("请先在『模型训练』标签页训练模型")

with tab4:
    st.subheader("性能评估")
    
    if st.session_state['trained']:
        results = st.session_state['evaluation_results']
        
        st.subheader("训练集 vs 测试集指标")
        metrics_df = pd.DataFrame({
            '指标': ['RMSE', 'R²', 'SMAPE'],
            '训练集': [
                f"{results['train']['RMSE']:.2f}",
                f"{results['train']['R²']:.4f}",
                f"{results['train']['SMAPE']:.2f}%"
            ],
            '测试集': [
                f"{results['test']['RMSE']:.2f}",
                f"{results['test']['R²']:.4f}",
                f"{results['test']['SMAPE']:.2f}%"
            ]
        })
        st.table(metrics_df)
        
        st.subheader("预测值 vs 实际值对比")
        compare_df = pd.DataFrame({
            'date_hour': results['test_df']['date_hour'].values,
            '实际值': results['y_test'].values,
            '预测值': results['test_pred']
        })
        compare_df['误差'] = compare_df['实际值'] - compare_df['预测值']
        
        fig = px.line(compare_df, x='date_hour', y=['实际值', '预测值'], 
                    title='测试集预测对比')
        fig.update_layout(xaxis_title='时间', yaxis_title='电价 (元/MWh)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("误差分布")
        fig = px.histogram(compare_df, x='误差', nbins=50, title='预测误差分布')
        fig.update_layout(xaxis_title='误差 (元/MWh)', yaxis_title='频率')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("特征重要性")
        model = st.session_state['model']
        feature_importance = model.get_feature_importance(results['X_train'].columns)
        if feature_importance is not None:
            top_features = feature_importance.head(10)
            fig = px.bar(x=top_features.values, y=top_features.index, 
                        orientation='h', title='Top 10 重要特征')
            fig.update_layout(xaxis_title='重要性', yaxis_title='特征')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("当前模型不支持特征重要性分析")
    else:
        st.info("请先在『模型训练』标签页训练模型")