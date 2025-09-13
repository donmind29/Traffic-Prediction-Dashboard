import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="XGBoost Traffic Speed Prediction Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E86AB;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# ---- ADDED ---- helper to find street column consistently
def find_street_column(df, candidates=None):
    """返回数据中第一个存在的街道列名（或 None）。"""
    if df is None:
        return None
    if candidates is None:
        candidates = ['streetName', 'road_name', 'street', 'segment_id', 'link_id', 'road_id']
    for col in candidates:
        if col in df.columns:
            return col
    return None


# 缓存函数用于加载模型和数据
@st.cache_resource
def load_models():
    """加载所有保存的模型"""
    try:
        models_dict = joblib.load('all_models.pkl')
        preprocessing_info = joblib.load('preprocessing_info.pkl')
        results_df = joblib.load('hyperparameter_tuning_results.pkl')
        return models_dict, preprocessing_info, results_df
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None


@st.cache_data
def load_original_data():
    """加载原始数据"""
    try:
        df = pd.read_csv('Johor_Bahru_hour_with_POI.csv')
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None


@st.cache_data
def get_top_streets(df, top_n=10):
    """获取样本数前N条街道（使用 find_street_column 统一识别列名）"""
    if df is None:
        return []

    street_col = find_street_column(df)
    if street_col:
        street_counts = df[street_col].value_counts()
        return street_counts.head(top_n).index.tolist()
    else:
        # 如果没有街道列，返回前N个唯一标识
        if 'index' in df.columns:
            return df['index'].unique()[:top_n].tolist()
        else:
            return list(range(top_n))


def create_future_prediction_interface(model, scaler, feature_names, original_df=None):
    """创建未来预测界面（Prediction Settings 移到侧边栏）"""

    # 初始化session state
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'shap_results' not in st.session_state:
        st.session_state.shap_results = None
    if 'prediction_params' not in st.session_state:
        st.session_state.prediction_params = None

    # 获取前10条街道
    top_streets = get_top_streets(original_df, top_n=10)

    # === 在侧边栏添加预测设置 ===
    with st.sidebar:
        st.divider()
        st.markdown("### 🔮 Prediction Settings")

        # 选择街道
        selected_street = st.selectbox(
            "Select Street",
            options=top_streets,
            format_func=lambda x: f"Street {x}" if isinstance(x, (int, float)) else str(x),
            help="Select a street to predict (Top 10 by sample count)",
            key="sidebar_street_selector"
        )

        # 选择预测时间范围
        st.markdown("#### Time Range")

        # 开始日期
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date(),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=30),
            key="sidebar_start_date"
        )

        # 预测天数
        prediction_days = st.slider(
            "Prediction Days",
            min_value=1,
            max_value=7,
            value=3,
            help="Select number of days to predict (1-7 days)",
            key="sidebar_prediction_days"
        )

        # 时间粒度
        time_granularity = st.radio(
            "Time Granularity",
            options=["Hourly", "Peak Hours", "Daily Average"],
            index=0,
            help="Select prediction time granularity",
            key="sidebar_time_granularity"
        )

        # 预测按钮
        predict_button = st.button("🚀 Start Prediction", use_container_width=True, type="primary",
                                   key="sidebar_predict_btn")

    # === 主界面只显示预测结果 ===
    # 只有点击预测按钮时才重新计算
    if predict_button:
        with st.spinner("Generating predictions..."):
            try:
                # 生成预测时间序列并同时返回特征矩阵（原始、缩放）
                predictions, feature_df, feature_df_scaled = generate_future_predictions(
                    model, scaler, feature_names, original_df,
                    selected_street, start_date, prediction_days, time_granularity
                )

                # 保存预测结果到session state
                st.session_state.prediction_results = {
                    'predictions': predictions,
                    'feature_df': feature_df,
                    'feature_df_scaled': feature_df_scaled
                }

                # 保存预测参数
                st.session_state.prediction_params = {
                    'selected_street': selected_street,
                    'start_date': start_date,
                    'prediction_days': prediction_days,
                    'time_granularity': time_granularity
                }

                # 计算SHAP值并保存
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(feature_df_scaled)
                    if isinstance(shap_vals, list) and len(shap_vals) == 1:
                        shap_vals = shap_vals[0]

                    st.session_state.shap_results = {
                        'shap_vals': shap_vals,
                        'explainer': explainer
                    }
                except Exception as e:
                    st.session_state.shap_results = None
                    st.warning(f"Unable to calculate SHAP values: {e}")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.session_state.prediction_results = None
                st.session_state.shap_results = None

    # 显示预测结果（如果存在）
    if st.session_state.prediction_results is not None:
        # 从session state获取数据
        predictions = st.session_state.prediction_results['predictions']
        feature_df = st.session_state.prediction_results['feature_df']
        feature_df_scaled = st.session_state.prediction_results['feature_df_scaled']
        params = st.session_state.prediction_params

        # 显示预测结果
        st.markdown("### Prediction Results")

        # 显示当前预测参数
        st.info(f"📊 Current Prediction: Street {params['selected_street']} | "
                f"📅 {params['prediction_days']} days from {params['start_date']} | "
                f"⏰ {params['time_granularity']}")

        # 创建时间序列图
        fig = create_prediction_timeline(predictions, params['selected_street'], params['time_granularity'])
        st.plotly_chart(fig, use_container_width=True)

        # 显示统计信息
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            st.metric("Average Predicted Speed", f"{predictions['speed'].mean():.1f} km/h")
        with col2_2:
            st.metric("Maximum Speed", f"{predictions['speed'].max():.1f} km/h")
        with col2_3:
            st.metric("Minimum Speed", f"{predictions['speed'].min():.1f} km/h")

        # 显示详细数据表
        with st.expander("View Detailed Prediction Data"):
            st.dataframe(
                predictions[['datetime', 'speed', 'confidence']].style.format({
                    'speed': '{:.1f} km/h',
                    'confidence': '{:.1%}'
                }),
                use_container_width=True
            )

        # -------------- SHAP 分析（针对本次预测） --------------
        st.markdown("## 🔎 SHAP Value Analysis for This Prediction")

        if st.session_state.shap_results is None:
            st.info(
                "SHAP value calculation failed or unavailable. Please ensure the model supports SHAP (tree models) and feature order matches training.")
        else:
            shap_vals = st.session_state.shap_results['shap_vals']

            # 时间点选择不会触发页面重新计算
            st.markdown("### Single Point SHAP Decomposition")

            # 格式化时间用于显示
            time_labels = [pd.to_datetime(t).strftime("%Y-%m-%d %H:%M:%S") for t in
                           predictions['datetime'].tolist()]

            if len(time_labels) > 0:
                # 使用unique key避免与其他selectbox冲突
                selected_idx = st.selectbox(
                    "Select Time Point (View SHAP decomposition for this time)",
                    options=list(range(len(time_labels))),
                    format_func=lambda i: time_labels[i],
                    index=0,
                    key="shap_time_selector"  # 添加unique key
                )

                if selected_idx is not None:
                    sample_shap = np.array(shap_vals)[selected_idx]
                    sample_feature_values = feature_df.iloc[selected_idx]

                    shap_df = pd.DataFrame({
                        'feature': feature_names,
                        'feature_value': sample_feature_values.values,
                        'shap_value': sample_shap
                    })
                    shap_df['abs_shap'] = shap_df['shap_value'].abs()
                    shap_df = shap_df.sort_values('abs_shap', ascending=False)

                    # 横向条形图：按贡献大小排序，正负用颜色区分
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df['shap_value'],
                        y=shap_df['feature'],
                        orientation='h',
                        marker=dict(
                            color=shap_df['shap_value'].apply(lambda v: '#4ECDC4' if v >= 0 else '#FF6B6B')
                        ),
                        text=shap_df['shap_value'].round(3),
                        textposition='outside'
                    ))
                    fig_shap.update_layout(
                        title=f"SHAP Decomposition — {time_labels[selected_idx]}",
                        xaxis_title="SHAP Value (Impact on prediction: positive→increases, negative→decreases)",
                        yaxis_title="Feature",
                        height=max(400, len(shap_df) * 18),
                        margin=dict(l=200)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                    # 显示特征值与 SHAP 值表
                    with st.expander("View Feature Values and SHAP Details for This Time Point"):
                        st.dataframe(
                            shap_df[['feature', 'feature_value', 'shap_value']].reset_index(drop=True).style.format(
                                {
                                    'feature_value': '{:.4f}',
                                    'shap_value': '{:.4f}'
                                }),
                            use_container_width=True
                        )

            # 显示所有预测时间点的平均 |SHAP|（帮助把握本次预测区间的整体重要性分布）
            # st.markdown("### Average |SHAP| for This Prediction Period (Sorted by Feature)")
            # mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            # mean_shap_df = pd.DataFrame({
            #     'feature': feature_names,
            #     'mean_abs_shap': mean_abs_shap
            # }).sort_values('mean_abs_shap', ascending=False)
            #
            # fig_mean = go.Figure(go.Bar(
            #     x=mean_shap_df['mean_abs_shap'].head(30)[::-1],
            #     y=mean_shap_df['feature'].head(30)[::-1],
            #     orientation='h',
            #     text=mean_shap_df['mean_abs_shap'].head(30)[::-1].round(3),
            #     textposition='outside'
            # ))
            # fig_mean.update_layout(
            #     title="Average |SHAP| for This Prediction Period",
            #     xaxis_title="Average |SHAP|",
            #     yaxis_title="Feature",
            #     height=max(400, len(mean_shap_df.head(30)) * 20),
            #     margin=dict(l=200)
            # )
            # st.plotly_chart(fig_mean, use_container_width=True)
            # 调用瀑布图函数
            mean_shap_signed = np.mean(shap_vals, axis=0)
            avg_prediction = predictions['speed'].mean()
            try:
                explainer = st.session_state.shap_results['explainer']
                base_value = explainer.expected_value
                if isinstance(base_value, np.ndarray):
                    base_value = float(base_value[0])
                else:
                    base_value = float(base_value)
            except:
                # 如果无法获取基准值，使用训练数据的平均值作为估计
                base_value = avg_prediction - np.sum(mean_shap_signed)
            waterfall_fig = create_shap_waterfall_plot(
                feature_names, mean_shap_signed, base_value, avg_prediction
            )
            st.plotly_chart(waterfall_fig, use_container_width=True)

    else:
        # 如果没有预测结果，显示提示
        st.info(
            "👈 Configure prediction settings in the sidebar and click 'Start Prediction' to generate predictions and SHAP analysis")


def generate_future_predictions(model, scaler, feature_names, original_df,
                                selected_street, start_date, prediction_days, time_granularity):
    """生成未来预测数据并返回：predictions_df, feature_df, feature_df_scaled"""
    predictions = []

    # 生成时间点
    if time_granularity == "Hourly":
        time_points = []
        for day in range(prediction_days):
            current_date = start_date + timedelta(days=day)
            for hour in range(24):
                time_points.append(datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour))
    elif time_granularity == "Peak Hours":
        time_points = []
        for day in range(prediction_days):
            current_date = start_date + timedelta(days=day)
            # 早高峰 7-9点, 晚高峰 17-19点
            for hour in [7, 8, 9, 17, 18, 19]:
                time_points.append(datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour))
    else:  # Daily Average
        time_points = []
        for day in range(prediction_days):
            current_date = start_date + timedelta(days=day)
            time_points.append(datetime.combine(current_date, datetime.min.time()) + timedelta(hours=12))

    # 构建所有时间点的特征矩阵（原始）
    feature_matrix = []
    for time_point in time_points:
        fv = build_feature_vector(
            original_df, selected_street, time_point, feature_names
        )
        feature_matrix.append(fv)

    # 转为 DataFrame（原始特征值）
    if len(feature_matrix) == 0:
        feature_df = pd.DataFrame(columns=feature_names)
    else:
        feature_df = pd.DataFrame(feature_matrix, columns=feature_names)

    # 标准化（尽量一次性转换）
    if scaler is not None and len(feature_df) > 0:
        try:
            feature_matrix_scaled = scaler.transform(feature_df.values)
        except Exception as e:
            # 如果缩放失败，退回原始值并记录警告（上层会捕获）
            feature_matrix_scaled = feature_df.values
    else:
        feature_matrix_scaled = feature_df.values

    # 预测（一次性）
    try:
        preds = model.predict(feature_matrix_scaled)
    except Exception as e:
        # 回退到逐个预测（更鲁棒，但慢）
        preds = []
        for row in feature_matrix_scaled:
            try:
                preds.append(model.predict([row])[0])
            except:
                preds.append(np.nan)
        preds = np.array(preds)

    # 为每个时间点构建预测记录
    for i, time_point in enumerate(time_points):
        pred_speed = float(preds[i]) if i < len(preds) else np.nan

        # 计算置信度（使用之前的函数）
        confidence = calculate_confidence(pred_speed, time_point)

        predictions.append({
            'datetime': time_point,
            'speed': pred_speed,
            'confidence': confidence,
            'street': selected_street
        })

    predictions_df = pd.DataFrame(predictions)
    # 保证 datetime 列为 pd.Timestamp
    if not predictions_df.empty:
        predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])

    # feature_df_scaled 也转换为 DataFrame 方便后续索引（列名与 feature_names 对齐）
    feature_df_scaled = pd.DataFrame(feature_matrix_scaled, columns=feature_names) if len(
        feature_matrix_scaled) > 0 else pd.DataFrame(columns=feature_names)

    return predictions_df, feature_df, feature_df_scaled


def build_feature_vector(original_df, selected_street, time_point, feature_names):
    """构建特征向量（根据 feature_names 顺序）"""
    feature_vector = []

    # 尝试识别街道列
    street_col = find_street_column(original_df)

    for feature in feature_names:
        if 'hour' in feature.lower():
            feature_vector.append(time_point.hour)
        elif 'weekend' in feature.lower():
            feature_vector.append(1 if time_point.weekday() >= 5 else 0)
        elif 'peak' in feature.lower():
            is_peak = time_point.hour in [7, 8, 9, 17, 18, 19]
            feature_vector.append(1 if is_peak else 0)
        elif original_df is not None and feature in original_df.columns:
            # 从历史数据中获取该街道的平均值（按识别到的街道列筛选）
            if street_col is not None:
                street_data = original_df[original_df[street_col] == selected_street]
            else:
                # 如果没有街道列，则尝试使用 index 或空表
                if 'index' in original_df.columns:
                    street_data = original_df[original_df['index'] == selected_street]
                else:
                    street_data = pd.DataFrame()

            if not street_data.empty and feature in street_data.columns:
                # 使用该街道历史上的均值作为默认
                feature_vector.append(street_data[feature].mean())
            else:
                # 当无历史时用 0 填充（可根据需求替换为其他策略）
                feature_vector.append(0.0)
        else:
            # 默认占位
            feature_vector.append(0.0)

    return feature_vector


def calculate_confidence(pred_speed, time_point):
    """计算预测置信度"""
    # 基于时间和速度的简单置信度估算
    base_confidence = 0.85

    # 工作日的置信度更高
    if time_point.weekday() < 5:
        base_confidence += 0.05

    # 高峰时段的置信度较低（更多不确定性）
    if time_point.hour in [7, 8, 9, 17, 18, 19]:
        base_confidence -= 0.1

    # 速度在正常范围内的置信度更高
    if 30 <= pred_speed <= 80:
        base_confidence += 0.05

    return max(0.6, min(0.95, base_confidence))


def create_prediction_timeline(predictions, selected_street, time_granularity):
    """创建预测时间线图"""
    fig = go.Figure()

    # 添加预测线
    fig.add_trace(go.Scatter(
        x=predictions['datetime'],
        y=predictions['speed'],
        mode='lines+markers',
        name='Predicted Speed',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=8, color='#2E86AB'),
        hovertemplate='Time: %{x}<br>Speed: %{y:.1f} km/h<extra></extra>'
    ))

    # 添加置信区间
    upper_bound = predictions['speed'] * (1 + (1 - predictions['confidence']))
    lower_bound = predictions['speed'] * (1 - (1 - predictions['confidence']))

    fig.add_trace(go.Scatter(
        x=predictions['datetime'].tolist() + predictions['datetime'].tolist()[::-1],
        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 标记高峰时段
    if time_granularity == "Hourly":
        for idx, row in predictions.iterrows():
            if row['datetime'].hour in [7, 8, 9, 17, 18, 19]:
                fig.add_vline(
                    x=row['datetime'],
                    line_dash="dot",
                    line_color="orange",
                    opacity=0.3
                )

    fig.update_layout(
        title=f"Street {selected_street} - Speed Prediction for Next {len(predictions)} Time Points",
        xaxis_title="Time",
        yaxis_title="Predicted Speed (km/h)",
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # 添加网格
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    return fig


def create_shap_waterfall_plot(feature_names, mean_shap_values, base_value, final_prediction, top_n=15):
    """
    创建SHAP瀑布图，显示特征如何从基准值逐步贡献到最终预测值

    Parameters:
    - feature_names: 特征名称列表
    - mean_shap_values: 平均SHAP值（保留正负）
    - base_value: 模型基准值
    - final_prediction: 最终预测值
    - top_n: 显示前N个最重要的特征
    """

    # 创建特征贡献DataFrame并按绝对值排序
    contrib_df = pd.DataFrame({
        'feature': feature_names,
        'contribution': mean_shap_values,
        'abs_contribution': np.abs(mean_shap_values)
    }).sort_values('abs_contribution', ascending=False)

    # 只取前N个最重要的特征，其余合并为"Other Features"
    if len(contrib_df) > top_n:
        top_features = contrib_df.head(top_n)
        other_contribution = contrib_df.tail(len(contrib_df) - top_n)['contribution'].sum()

        # 添加"Other Features"项
        other_row = pd.DataFrame({
            'feature': ['Other Features'],
            'contribution': [other_contribution],
            'abs_contribution': [abs(other_contribution)]
        })
        waterfall_data = pd.concat([top_features, other_row], ignore_index=True)
    else:
        waterfall_data = contrib_df

    # 计算瀑布图的累积值
    cumulative_values = [base_value]
    for contrib in waterfall_data['contribution']:
        cumulative_values.append(cumulative_values[-1] + contrib)

    # 创建瀑布图
    fig = go.Figure()

    # 添加基准值
    fig.add_trace(go.Bar(
        name='Baseline',
        x=['Baseline'],
        y=[base_value],
        marker_color='lightgray',
        text=[f'{base_value:.2f}'],
        textposition='outside',
        showlegend=True
    ))

    # 添加每个特征的贡献
    x_labels = ['Baseline']
    colors = []
    y_values = [base_value]

    for i, (_, row) in enumerate(waterfall_data.iterrows()):
        feature = row['feature']
        contrib = row['contribution']

        # 截断过长的特征名
        display_name = feature if len(feature) <= 20 else feature[:17] + '...'
        x_labels.append(display_name)

        # 确定颜色：正贡献为绿色，负贡献为红色
        color = '#2E8B57' if contrib > 0 else '#DC143C'  # 深绿 vs 深红
        colors.append(color)

        # 对于瀑布图，需要显示增量条形图
        if contrib > 0:
            # 正贡献：从前一个值开始向上
            base = cumulative_values[i]
            height = contrib
        else:
            # 负贡献：从前一个值+贡献开始向上（贡献为负值）
            base = cumulative_values[i + 1]
            height = -contrib  # 转为正值用于显示

        fig.add_trace(go.Bar(
            name=display_name,
            x=[display_name],
            y=[height],
            base=base,
            marker_color=color,
            text=[f'{contrib:+.2f}'],
            textposition='outside',
            showlegend=False,
            hovertemplate=f'<b>{feature}</b><br>Contribution: {contrib:+.3f}<br>Cumulative: {cumulative_values[i + 1]:.3f}<extra></extra>'
        ))

    # 添加最终预测值
    fig.add_trace(go.Bar(
        name='Final Prediction',
        x=['Final Prediction'],
        y=[final_prediction],
        marker_color='#2E86AB',
        text=[f'{final_prediction:.2f}'],
        textposition='outside',
        showlegend=True
    ))

    # 添加连接线显示累积效果
    x_positions = list(range(len(x_labels) + 1))
    fig.add_trace(go.Scatter(
        x=x_positions,
        y=cumulative_values,
        mode='lines+markers',
        line=dict(color='black', width=1, dash='dot'),
        marker=dict(color='black', size=4),
        name='Cumulative Value',
        showlegend=True,
        hovertemplate='Cumulative: %{y:.3f}<extra></extra>'
    ))

    # 更新布局
    fig.update_layout(
        title=f"SHAP Waterfall Plot - How Features Build Up the Prediction",
        xaxis_title="Features (ordered by importance)",
        yaxis_title="Predicted Speed (km/h)",
        height=500,
        barmode='overlay',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x=0,
                y=base_value + max(abs(base_value) * 0.1, 2),
                text=f"Baseline: {base_value:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="gray",
                font=dict(color="gray", size=10)
            ),
            dict(
                x=len(x_labels),
                y=final_prediction + max(abs(final_prediction) * 0.1, 2),
                text=f"Final: {final_prediction:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#2E86AB",
                font=dict(color="#2E86AB", size=10)
            )
        ]
    )

    # 调整x轴标签角度以适应长特征名
    fig.update_xaxes(tickangle=45)

    return fig


def create_feature_importance_plot(model, feature_names, show_all=True):
    """创建特征重要性图（显示全部特征）"""
    # 获取特征重要性
    importances = model.feature_importances_

    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 如果不显示全部，只显示前N个
    if not show_all and len(importance_df) > 15:
        importance_df = importance_df.head(15)

    # 创建条形图
    fig = go.Figure(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker=dict(
            color=importance_df['importance'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=importance_df['importance'].round(4),
        textposition='outside'
    ))

    fig.update_layout(
        title="Feature Importance Analysis (All Features)" if show_all else "Feature Importance Analysis (Top 15)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(400, len(importance_df) * 25),  # 根据特征数量调整高度
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=150)  # 增加左边距以容纳特征名称
    )

    return fig


def create_shap_summary_with_all_features(shap_values, X_sample):
    """创建显示所有特征的SHAP摘要图"""
    fig = go.Figure()

    # 计算每个特征的平均绝对SHAP值
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    # 显示所有特征
    for idx, feature in enumerate(feature_importance['feature']):
        feature_idx = X_sample.columns.get_loc(feature)

        # 获取该特征的SHAP值和特征值
        shap_vals = shap_values[:, feature_idx]
        feature_vals = X_sample[feature].values

        # 归一化特征值用于颜色映射
        if feature_vals.std() > 0:
            feature_vals_norm = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
        else:
            feature_vals_norm = np.zeros_like(feature_vals)

        # 添加散点（使用较小的点以容纳更多特征）
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=[idx] * len(shap_vals) + np.random.normal(0, 0.05, len(shap_vals)),
            mode='markers',
            marker=dict(
                size=3,
                color=feature_vals_norm,
                colorscale='RdBu_r',
                showscale=(idx == 0),
                colorbar=dict(
                    title="Feature Value<br>(Normalized)",
                    x=1.02
                ),
                opacity=0.6
            ),
            name=feature,
            hovertemplate=f'{feature}<br>SHAP Value: %{{x:.3f}}<br>Feature Value: %{{marker.color:.3f}}<extra></extra>'
        ))

    fig.update_layout(
        title=f"SHAP Value Summary Plot (All {len(feature_importance)} Features)",
        xaxis_title="SHAP Value (Impact on Model Output)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(feature_importance))),
            ticktext=feature_importance['feature'].tolist()[::-1],
            title=""
        ),
        height=max(500, len(feature_importance) * 20),  # 根据特征数量调整高度
        showlegend=False,
        hovermode='closest',
        margin=dict(l=150)  # 增加左边距
    )

    return fig


# 计算特征交互
def calculate_feature_interactions(shap_values, X_sample, feature_names, n_top_interactions=10):
    """
    计算特征间的交互效应

    Parameters:
    - shap_values: SHAP values array
    - X_sample: sample data
    - feature_names: list of feature names
    - n_top_interactions: number of top interactions to return

    Returns:
    - DataFrame with interaction scores
    """
    interaction_matrix = np.zeros((len(feature_names), len(feature_names)))

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            try:
                # 计算特征值之间的相关性
                feature_corr = np.corrcoef(X_sample.iloc[:, i], X_sample.iloc[:, j])[0, 1]

                # 计算SHAP值之间的相关性
                shap_corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]

                # 交互强度 = |特征相关性 * SHAP相关性|
                interaction_score = np.abs(feature_corr * shap_corr)

                interaction_matrix[i, j] = interaction_score
                interaction_matrix[j, i] = interaction_score

            except:
                interaction_matrix[i, j] = 0
                interaction_matrix[j, i] = 0

    # 提取上三角矩阵的交互对
    interactions = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            interactions.append({
                'Feature1': feature_names[i],
                'Feature2': feature_names[j],
                'Interaction_Score': interaction_matrix[i, j]
            })

    # 按交互强度排序
    interactions_df = pd.DataFrame(interactions).sort_values(
        'Interaction_Score', ascending=False
    ).head(n_top_interactions)

    return interactions_df, interaction_matrix


def main():
    # 标题
    st.markdown('<h1 class="main-header">🚗 Traffic Speed Prediction Analysis Platform</h1>',
                unsafe_allow_html=True)

    # 加载模型和数据
    with st.spinner('Loading the model and data...'):
        models_dict, preprocessing_info, results_df = load_models()
        original_df = load_original_data()

    if models_dict is None or preprocessing_info is None:
        st.error(
            "The necessary model files cannot be loaded. Please ensure that all files are in the correct location.")
        return

    # 使用最佳模型
    best_model = models_dict['best_model']
    X_test = preprocessing_info['X_test']
    y_test = preprocessing_info['y_test']
    feature_names = preprocessing_info['feature_names']
    scaler = preprocessing_info['scaler']

    # 侧边栏配置
    with st.sidebar:
        st.title("⚙️ Model Information")

        # 显示模型基本信息
        st.subheader("📊 Model Performance")
        try:
            y_pred = best_model.predict(X_test)
            current_r2 = r2_score(y_test, y_pred)
            current_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            current_mae = mean_absolute_error(y_test, y_pred)

            # st.metric("模型类型", "XGBoost 最佳模型")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{current_r2:.4f}")
                st.metric("RMSE", f"{current_rmse:.4f}")
            with col2:
                st.metric("MAE", f"{current_mae:.4f}")
                # st.metric("测试样本数", f"{len(X_test)}")
        except Exception as e:
            st.error(f"无法计算模型性能: {str(e)}")

        st.divider()

        # SHAP分析配置
        # st.subheader("分析配置")
        shap_sample_size = 1000
        # shap_sample_size = st.slider(
        #     "SHAP分析样本数",
        #     min_value=50,
        #     max_value=500,
        #     value=100,
        #     step=50,
        #     help="更多样本会提供更准确的分析，但计算时间更长"
        # )
        show_all_features = True
        # show_all_features = st.checkbox(
        #     "显示所有特征",
        #     value=True,
        #     help="勾选以显示所有特征的重要性"
        # )

    # 主界面：未来预测
    st.header("🔮 Prediction of Future Traffic Speeds")
    try:
        create_future_prediction_interface(best_model, scaler, feature_names, original_df)
    except Exception as e:
        st.error(f"预测界面初始化失败: {str(e)}")


if __name__ == "__main__":
    main()