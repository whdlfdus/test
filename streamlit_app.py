import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # For numeric type checking / conversion
import uuid # For unique keys for filter/sort rules
from google.cloud import bigquery # BigQuery 클라이언트 라이브러리
from google.oauth2 import service_account # 서비스 계정 인증용 (선택 사항)

# NumPy bool8 호환성 문제 해결을 위한 코드 추가
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_ # np.bool_을 np.bool8로 별칭 지정

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="고급 데이터 시각화 도구 v2.13") # 버전 업데이트

# --- 초기 세션 상태 설정 ---
def init_session_state():
    defaults = {
        'df': None, 'headers': [], 'numeric_headers': [], 'string_headers': [],
        'chart_type': '막대 (Bar)', 'x_axis': None,
        'y_axis_single': None,
        'y_axis_multiple': [],
        'y_axis_secondary': "None",
        'group_by_col': "None",
        'agg_method': 'Sum',
        'pie_name_col': None, 'pie_value_col': None,
        'last_uploaded_filename': None,
        'data_loaded_success': False,
        'df_raw_uploaded': None, # 원본 데이터 저장용
        # 정제
        'mv_selected_cols': [], 'mv_method': "결측치가 있는 행 전체 제거", 'mv_specific_value': "",
        'ot_selected_cols': [], 'ot_detection_method': "IQR 방식", 'ot_iqr_multiplier': 1.5,
        'ot_zscore_threshold': 3.0, 'ot_treatment_method': "결측치로 처리 (NaN으로 대체)",
        'dd_subset_cols': [], 'dd_keep_method': "첫 번째 행 유지",
        # 변환
        'filter_rules': [],
        'filter_conjunction': 'AND',
        # 정렬
        'sort_rules': [],
        # 구조 변경
        'pivot_index_cols': [], 'pivot_columns_col': None,
        'pivot_values_cols': [], 'pivot_agg_func': 'mean',
        'unpivot_id_vars': [], 'unpivot_value_vars': [],
        'unpivot_var_name': 'variable', 'unpivot_value_name': 'value',
        # 파생 변수 (기존 수식 기반)
        'derived_var_name': '', 'derived_var_formula': '',
        # 고급 파생 변수 (신규)
        'advanced_derived_definitions': {}, 
        'show_adv_derived_var_builder': False,
        'editing_adv_derived_var_name': None,
        'adv_builder_var_name': "",
        'adv_builder_var_type': 'conditional', 
        'adv_builder_conditional_rules': [], 
        'adv_builder_else_value': '',
        'adv_builder_window_func': 'ROW_NUMBER',
        'adv_builder_window_target_col': '',
        'adv_builder_window_partition_by': [],
        'adv_builder_window_order_by_col': '',
        'adv_builder_window_order_by_dir': 'ASC',
        'adv_builder_window_lag_lead_offset': 1,
        'adv_builder_window_lag_lead_default': '',
        # BigQuery
        'bq_query': "SELECT\n    name,\n    SUM(number) AS total_widgets\nFROM\n    `bigquery-public-data.usa_names.usa_1910_current`\nWHERE\n    name LIKE 'A%'\nGROUP BY\n    name\nORDER BY\n    total_widgets DESC\nLIMIT 100;",
        'bq_project_id': '', # 선택적: 사용자가 프로젝트 ID를 직접 입력하게 할 경우
        # 분포 차트
        'hist_bins': None,
        'box_points': "outliers",
        # 관계 차트
        'scatter_x_col': None, 'scatter_y_col': None,
        'scatter_color_col': "None", 'scatter_size_col': "None",
        'scatter_hover_name_col': "None",
        # 밀도 플롯
        'density_value_cols': [], 'density_color_col': "None",
        # 레이더 차트
        'radar_category_col': None, 'radar_value_cols': [],
        # 히트맵
        'heatmap_corr_cols': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if not st.session_state.adv_builder_conditional_rules:
        st.session_state.adv_builder_conditional_rules = [{'id': str(uuid.uuid4()), 'variable1': '', 'operator1': '==', 'value1': '', 'logical_op': '', 'variable2': '', 'operator2': '==', 'value2': '', 'then_value': ''}]

init_session_state()

ADV_COMPARISON_OPERATORS = ['==', '!=', '>', '>=', '<', '<=', 'contains', 'startswith', 'endswith', 'isnull', 'notnull']
ADV_LOGICAL_OPERATORS = ['AND', 'OR']
ADV_WINDOW_FUNCTIONS = ['ROW_NUMBER', 'RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'SUM', 'AVG', 'MIN', 'MAX', 'COUNT']
ADV_SORT_DIRECTIONS = ['ASC', 'DESC']

def get_column_types(df):
    if df is None: return [], []
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    all_cols_for_cat = list(df.columns)
    return numeric_cols, all_cols_for_cat

def get_all_available_variables_for_derived():
    if st.session_state.df is None: return []
    return list(st.session_state.df.columns)

def get_variable_type_for_derived(var_name):
    if st.session_state.df is not None and var_name in st.session_state.df.columns:
        if pd.api.types.is_numeric_dtype(st.session_state.df[var_name]):
            return 'number'
    return 'string'

def apply_advanced_derived_variables(df_input):
    if df_input is None or not st.session_state.advanced_derived_definitions:
        return df_input
    current_df = df_input.copy()
    original_cols_before_adv = list(current_df.columns)
    for var_name, definition in st.session_state.advanced_derived_definitions.items():
        try:
            if definition['type'] == 'conditional':
                conditions = []
                choices = []
                for rule in definition['rules']:
                    s1_series = current_df[rule['variable1']]
                    op1 = rule['operator1']
                    val1_str = str(rule['value1'])
                    val1 = val1_str
                    if get_variable_type_for_derived(rule['variable1']) == 'number' and op1 not in ['isnull', 'notnull', 'contains', 'startswith', 'endswith']:
                        try: val1 = float(val1_str)
                        except ValueError: st.warning(f"규칙 '{var_name}'-'{rule['variable1']}': 값 '{val1_str}' 숫자 변환 실패. 문자열로 비교 시도."); pass
                    cond1 = pd.Series([True] * len(current_df), index=current_df.index)
                    if op1 == '==': cond1 = (s1_series == val1)
                    elif op1 == '!=': cond1 = (s1_series != val1)
                    elif op1 == '>': cond1 = (s1_series > val1)
                    elif op1 == '>=': cond1 = (s1_series >= val1)
                    elif op1 == '<': cond1 = (s1_series < val1)
                    elif op1 == '<=': cond1 = (s1_series <= val1)
                    elif op1 == 'contains': cond1 = s1_series.astype(str).str.contains(val1_str, case=False, na=False)
                    elif op1 == 'startswith': cond1 = s1_series.astype(str).str.startswith(val1_str, na=False)
                    elif op1 == 'endswith': cond1 = s1_series.astype(str).str.endswith(val1_str, na=False)
                    elif op1 == 'isnull': cond1 = s1_series.isnull()
                    elif op1 == 'notnull': cond1 = s1_series.notnull()
                    final_cond = cond1
                    if rule['logical_op']:
                        s2_series = current_df[rule['variable2']]
                        op2 = rule['operator2']
                        val2_str = str(rule['value2'])
                        val2 = val2_str
                        if get_variable_type_for_derived(rule['variable2']) == 'number' and op2 not in ['isnull', 'notnull', 'contains', 'startswith', 'endswith']:
                            try: val2 = float(val2_str)
                            except ValueError: st.warning(f"규칙 '{var_name}'-'{rule['variable2']}': 값 '{val2_str}' 숫자 변환 실패. 문자열로 비교 시도."); pass
                        cond2 = pd.Series([True] * len(current_df), index=current_df.index)
                        if op2 == '==': cond2 = (s2_series == val2)
                        elif op2 == '!=': cond2 = (s2_series != val2)
                        elif op2 == '>': cond2 = (s2_series > val2)
                        elif op2 == '>=': cond2 = (s2_series >= val2)
                        elif op2 == '<': cond2 = (s2_series < val2)
                        elif op2 == '<=': cond2 = (s2_series <= val2)
                        elif op2 == 'contains': cond2 = s2_series.astype(str).str.contains(val2_str, case=False, na=False)
                        elif op2 == 'startswith': cond2 = s2_series.astype(str).str.startswith(val2_str, na=False)
                        elif op2 == 'endswith': cond2 = s2_series.astype(str).str.endswith(val2_str, na=False)
                        elif op2 == 'isnull': cond2 = s2_series.isnull()
                        elif op2 == 'notnull': cond2 = s2_series.notnull()
                        if rule['logical_op'] == 'AND': final_cond = cond1 & cond2
                        elif rule['logical_op'] == 'OR': final_cond = cond1 | cond2
                    conditions.append(final_cond)
                    choices.append(rule['then_value'])
                current_df[var_name] = np.select(conditions, choices, default=definition['else_value'])
            elif definition['type'] == 'window':
                conf = definition['config']
                target_col = conf['target_col']
                df_for_window = current_df.copy()
                if conf['order_by_col'] and conf['function'] in ['RANK', 'DENSE_RANK', 'LAG', 'LEAD', 'ROW_NUMBER']:
                    ascending_val = conf['order_by_dir'] == 'ASC'
                    sort_by_list = conf['partition_by'] + [conf['order_by_col']] if conf['partition_by'] else [conf['order_by_col']]
                    ascending_list = [True]*len(conf['partition_by']) + [ascending_val] if conf['partition_by'] else [ascending_val]
                    df_for_window = df_for_window.sort_values(by=sort_by_list, ascending=ascending_list)
                grouped_df_for_transform = df_for_window
                if conf['partition_by']:
                    grouped_df_for_transform = df_for_window.groupby(conf['partition_by'], group_keys=False, sort=False)
                result_series = pd.Series(index=current_df.index, dtype=object)
                if conf['function'] == 'ROW_NUMBER':
                    if conf['partition_by']: result_series = grouped_df_for_transform.cumcount() + 1
                    else: result_series = pd.Series(np.arange(len(df_for_window)) + 1, index=df_for_window.index)
                elif conf['function'] == 'RANK':
                    if conf['order_by_col']: result_series = grouped_df_for_transform[conf['order_by_col']].rank(method='min', ascending=(conf['order_by_dir'] == 'ASC'))
                    else: st.warning(f"창 함수 '{var_name}': RANK는 ORDER BY가 필요합니다."); result_series = None
                elif conf['function'] == 'DENSE_RANK':
                    if conf['order_by_col']: result_series = grouped_df_for_transform[conf['order_by_col']].rank(method='dense', ascending=(conf['order_by_dir'] == 'ASC'))
                    else: st.warning(f"창 함수 '{var_name}': DENSE_RANK는 ORDER BY가 필요합니다."); result_series = None
                elif conf['function'] == 'LAG':
                    if target_col and conf['order_by_col']: result_series = grouped_df_for_transform[target_col].shift(periods=conf['offset'], fill_value=conf.get('default_value'))
                    else: st.warning(f"창 함수 '{var_name}': LAG는 대상 변수와 ORDER BY가 필요합니다."); result_series = None
                elif conf['function'] == 'LEAD':
                    if target_col and conf['order_by_col']: result_series = grouped_df_for_transform[target_col].shift(periods=-conf['offset'], fill_value=conf.get('default_value'))
                    else: st.warning(f"창 함수 '{var_name}': LEAD는 대상 변수와 ORDER BY가 필요합니다."); result_series = None
                elif conf['function'] == 'SUM':
                    if target_col: result_series = grouped_df_for_transform[target_col].transform('sum')
                    else: st.warning(f"창 함수 '{var_name}': SUM은 대상 변수가 필요합니다."); result_series = None
                elif conf['function'] == 'AVG':
                    if target_col: result_series = grouped_df_for_transform[target_col].transform('mean')
                    else: st.warning(f"창 함수 '{var_name}': AVG는 대상 변수가 필요합니다."); result_series = None
                elif conf['function'] == 'MIN':
                    if target_col: result_series = grouped_df_for_transform[target_col].transform('min')
                    else: st.warning(f"창 함수 '{var_name}': MIN은 대상 변수가 필요합니다."); result_series = None
                elif conf['function'] == 'MAX':
                    if target_col: result_series = grouped_df_for_transform[target_col].transform('max')
                    else: st.warning(f"창 함수 '{var_name}': MAX는 대상 변수가 필요합니다."); result_series = None
                elif conf['function'] == 'COUNT':
                    col_to_count = target_col if target_col else df_for_window.columns[0]
                    result_series = grouped_df_for_transform[col_to_count].transform('count')
                if result_series is not None: current_df[var_name] = result_series.reindex(current_df.index)
                else: current_df[var_name] = None
            if var_name in current_df.columns:
                try: current_df[var_name] = pd.to_numeric(current_df[var_name], errors='ignore')
                except Exception: pass
        except Exception as e_adv_derived:
            st.error(f"고급 파생 변수 '{var_name}' 적용 중 오류 발생: {e_adv_derived}")
            if var_name in current_df.columns and var_name not in df_input.columns:
                current_df = current_df.drop(columns=[var_name])
    return current_df

def _reset_dependent_states(all_cols, num_cols):
    """ 데이터 로드 후 관련 세션 상태를 초기화하거나 유효한 값으로 설정하는 함수 """
    st.session_state.x_axis = st.session_state.x_axis if st.session_state.x_axis in all_cols else (all_cols[0] if all_cols else None)
    st.session_state.y_axis_single = st.session_state.y_axis_single if st.session_state.y_axis_single in num_cols else (num_cols[0] if num_cols else None)
    st.session_state.y_axis_multiple = [col for col in st.session_state.y_axis_multiple if col in num_cols]
    if not st.session_state.y_axis_multiple and num_cols: st.session_state.y_axis_multiple = [num_cols[0]]

    st.session_state.group_by_col = st.session_state.group_by_col if st.session_state.group_by_col in (["None"] + all_cols) else "None"
    st.session_state.y_axis_secondary = st.session_state.y_axis_secondary if st.session_state.y_axis_secondary in (["None"] + num_cols) else "None"
    st.session_state.pie_name_col = st.session_state.pie_name_col if st.session_state.pie_name_col in all_cols else (all_cols[0] if all_cols else None)
    st.session_state.pie_value_col = st.session_state.pie_value_col if st.session_state.pie_value_col in num_cols else (num_cols[0] if num_cols else None)

    keys_to_clear_or_default = [
        'mv_selected_cols', 'ot_selected_cols', 'dd_subset_cols',
        'filter_rules', 'sort_rules',
        'pivot_index_cols', 'pivot_columns_col', 'pivot_values_cols',
        'unpivot_id_vars', 'unpivot_value_vars',
        'derived_var_name', 'derived_var_formula',
        'advanced_derived_definitions',
        'show_adv_derived_var_builder', 'editing_adv_derived_var_name',
        'hist_bins', 'box_points',
        'scatter_x_col', 'scatter_y_col', 'scatter_color_col', 'scatter_size_col', 'scatter_hover_name_col',
        'density_value_cols', 'density_color_col',
        'radar_category_col', 'radar_value_cols',
        'heatmap_corr_cols'
    ]
    for key in keys_to_clear_or_default:
        if key.endswith('_cols') or key.endswith('_vars') or key.endswith('_rules') or key == 'advanced_derived_definitions':
            st.session_state[key] = [] if key != 'advanced_derived_definitions' else {}
        elif key.endswith('_col') or key.endswith('filename') or key.endswith('_name'):
             st.session_state[key] = None if key not in ['group_by_col', 'y_axis_secondary', 'scatter_color_col', 'scatter_size_col', 'scatter_hover_name_col', 'density_color_col', 'editing_adv_derived_var_name'] else ("None" if key != 'editing_adv_derived_var_name' else None)
        elif key in ['derived_var_formula']:
             st.session_state[key] = ''
        elif key == 'hist_bins': st.session_state[key] = None
        elif key == 'box_points': st.session_state[key] = "outliers"
        elif key == 'show_adv_derived_var_builder': st.session_state[key] = False
    
    st.session_state.filter_rules = []
    st.session_state.scatter_x_col = num_cols[0] if len(num_cols) > 0 else None
    st.session_state.scatter_y_col = num_cols[1] if len(num_cols) > 1 else (num_cols[0] if len(num_cols) > 0 else None)
    st.session_state.density_value_cols = [num_cols[0]] if num_cols else []
    st.session_state.radar_category_col = all_cols[0] if all_cols else None
    st.session_state.radar_value_cols = num_cols[:min(len(num_cols), 3)]
    st.session_state.heatmap_corr_cols = num_cols[:min(len(num_cols), 5)]
    
    st.session_state.adv_builder_conditional_rules = [{'id': str(uuid.uuid4()), 'variable1': all_cols[0] if all_cols else '', 'operator1': '==', 'value1': '', 'logical_op': '', 'variable2': '', 'operator2': '==', 'value2': '', 'then_value': ''}]
    st.session_state.adv_builder_else_value = ''
    st.session_state.adv_builder_window_target_col = num_cols[0] if num_cols else ''


def update_dataframe_states(df_new, source_name="데이터"):
    """새로운 DataFrame으로 세션 상태를 업데이트하는 공통 함수"""
    st.session_state.df_raw_uploaded = df_new.copy()
    st.session_state.df = df_new.copy()
    st.session_state.headers = list(df_new.columns)
    st.session_state.original_cols = list(df_new.columns)
    numeric_cols, string_cols = get_column_types(df_new)
    st.session_state.numeric_headers = numeric_cols
    st.session_state.string_headers = string_cols
    st.session_state.data_loaded_success = True
    st.session_state.last_uploaded_filename = source_name # 파일명 대신 소스명으로

    _reset_dependent_states(st.session_state.headers, st.session_state.numeric_headers)
    
    st.success(f"{source_name}가 성공적으로 로드 및 업데이트되었습니다! 모든 정제/변환/구조변경/파생변수 및 차트 설정이 일부 초기화됩니다.")
    
    # 데이터 로드 후 즉시 고급 파생 변수 적용 (만약 정의된 것이 있다면)
    st.session_state.df = apply_advanced_derived_variables(st.session_state.df)
    st.session_state.headers = list(st.session_state.df.columns) # 파생 변수 적용 후 헤더 업데이트
    st.session_state.numeric_headers, st.session_state.string_headers = get_column_types(st.session_state.df)


def load_data_from_csv(uploaded_file):
    try:
        df_new = pd.read_csv(uploaded_file)
        update_dataframe_states(df_new, source_name=uploaded_file.name)
        return True
    except Exception as e:
        st.error(f"CSV 데이터 로드 중 오류 발생: {e}")
        st.session_state.data_loaded_success = False
        st.session_state.df = None; st.session_state.headers = []; st.session_state.numeric_headers = []; st.session_state.string_headers = []
        return False

def load_data_from_bigquery(query, project_id=None):
    try:
        # 사용자가 프로젝트 ID를 제공하면 사용, 아니면 환경 기본값 사용
        client = bigquery.Client(project=project_id) if project_id else bigquery.Client()
        query_job = client.query(query)  # API request
        df_new = query_job.to_dataframe()  # Waits for query to finish
        update_dataframe_states(df_new, source_name="BigQuery")
        return True
    except Exception as e:
        st.error(f"BigQuery 데이터 로드 중 오류 발생: {e}")
        st.session_state.data_loaded_success = False
        st.session_state.df = None; st.session_state.headers = []; st.session_state.numeric_headers = []; st.session_state.string_headers = []
        return False

def apply_all_processing_steps():
    if 'df_raw_uploaded' not in st.session_state or st.session_state.df_raw_uploaded is None:
        # st.warning("데이터 처리 단계를 적용하려면 먼저 데이터를 로드해주세요.") # 버튼 클릭 시점에 이미 df가 있어야 함
        return

    current_df = st.session_state.df_raw_uploaded.copy()
    
    # TODO: 여기에 모든 정제, 변환, 구조변경 로직을 순서대로 적용해야 합니다.
    # 각 기능의 "적용" 버튼 클릭 시 실행되던 로직을 함수화하여 여기서 호출하거나,
    # 또는 각 버튼 클릭 시 st.session_state.df_raw_uploaded를 직접 수정하는 대신,
    # 설정값만 저장하고 이 함수에서 일괄 적용하도록 변경해야 합니다.
    # 현재는 이 부분이 비어있어, 고급 파생 변수만 df_raw_uploaded에 새로 적용됩니다.

    # 고급 파생 변수 적용
    current_df = apply_advanced_derived_variables(current_df)

    st.session_state.df = current_df
    st.session_state.headers = list(current_df.columns)
    st.session_state.numeric_headers, st.session_state.string_headers = get_column_types(current_df)


# --- UI 구성 ---
st.title("📊 고급 데이터 시각화 도구 v2.13") 
st.markdown("CSV 파일을 업로드하거나 BigQuery에서 직접 데이터를 가져와 다양한 차트를 생성하세요. 데이터 정제, 변환, 구조 변경, 파생 변수 생성(수식 기반 및 고급 GUI 기반) 등을 지원합니다.")

with st.sidebar:
    st.markdown("<h4>1. 데이터 업로드</h4>", unsafe_allow_html=True)
    
    upload_method = st.radio("데이터 가져오기 방식:", ("CSV 파일 업로드", "BigQuery에서 직접 로드"), key="upload_method_selector")

    if upload_method == "CSV 파일 업로드":
        uploaded_file = st.file_uploader("CSV 파일 선택", type="csv", key="file_uploader_v2_13")
        if uploaded_file:
            if st.button("CSV 데이터 로드/업데이트", key="load_csv_button_v2_13", use_container_width=True):
                load_data_from_csv(uploaded_file)
                st.experimental_rerun()
    
    elif upload_method == "BigQuery에서 직접 로드":
        st.info("BigQuery 접근을 위해서는 실행 환경에 GCP 인증 정보(예: 서비스 계정 키, ADC)가 설정되어 있어야 합니다.")
        st.session_state.bq_query = st.text_area("BigQuery SQL 쿼리 입력:", value=st.session_state.bq_query, height=200, key="bq_query_input")
        if st.button("BigQuery 데이터 로드", key="load_bq_button_v2_13", use_container_width=True):
            if st.session_state.bq_query.strip():
                load_data_from_bigquery(st.session_state.bq_query)
                st.experimental_rerun()
            else:
                st.warning("BigQuery SQL 쿼리를 입력해주세요.")

    st.divider()

    df = st.session_state.df
    headers = st.session_state.headers
    numeric_headers = st.session_state.numeric_headers

    if df is not None and st.session_state.data_loaded_success:
        st.markdown("<h4>2. 차트 설정</h4>", unsafe_allow_html=True)
        with st.expander("차트 옵션 보기/숨기기", expanded=True):
            chart_type_options = [
                '막대 (Bar)', '누적 막대 (Stacked Bar)',
                '선 (Line)', '누적 영역 (Stacked Area)',
                '파이 (Pie)',
                '히스토그램 (Histogram)', '박스 플롯 (Box Plot)', '밀도 플롯 (Density Plot)',
                '분산형 차트 (Scatter Plot)', '버블 차트 (Bubble Chart)',
                '레이더 차트 (Radar Chart)', '히트맵 (Heatmap - 상관관계)'
            ]
            st.session_state.chart_type = st.selectbox("차트 종류", chart_type_options,
                                                       index=chart_type_options.index(st.session_state.chart_type) if st.session_state.chart_type in chart_type_options else 0,
                                                       key="chart_type_select_v2_13")
            
            # --- 복원된 차트 상세 설정 UI 시작 ---
            chart_type = st.session_state.chart_type 
            is_pie_chart = (chart_type == '파이 (Pie)')
            is_stacked_chart = (chart_type in ['누적 막대 (Stacked Bar)', '누적 영역 (Stacked Area)'])
            is_distribution_chart = (chart_type in ['히스토그램 (Histogram)', '박스 플롯 (Box Plot)', '밀도 플롯 (Density Plot)'])
            is_relationship_chart = (chart_type in ['분산형 차트 (Scatter Plot)', '버블 차트 (Bubble Chart)'])
            is_radar_chart = (chart_type == '레이더 차트 (Radar Chart)')
            is_heatmap_chart = (chart_type == '히트맵 (Heatmap - 상관관계)')

            if not is_pie_chart and not is_distribution_chart and not is_relationship_chart and not is_radar_chart and not is_heatmap_chart: 
                st.session_state.x_axis = st.selectbox("X축", headers, index=headers.index(st.session_state.x_axis) if st.session_state.x_axis in headers else 0, key="x_axis_select_v2_13_common")
                group_by_options = ["None"] + [h for h in headers if h != st.session_state.x_axis] 
                st.session_state.group_by_col = st.selectbox("그룹화 기준 열 (선택)", group_by_options, index=group_by_options.index(st.session_state.group_by_col) if st.session_state.group_by_col in group_by_options else 0, key="group_by_select_v2_13_common")
                group_by_col = st.session_state.group_by_col 
                st.session_state.agg_method = st.selectbox("집계 방식", ['Sum', 'Mean', 'Median'], index=['Sum', 'Mean', 'Median'].index(st.session_state.agg_method), key="agg_method_select_v2_13_common")

                if group_by_col != "None": 
                    available_measure_cols = [h for h in numeric_headers if h != st.session_state.x_axis and h != group_by_col]
                    if not available_measure_cols: st.warning("그룹화에 사용할 적절한 숫자형 측정값 열이 없습니다."); st.session_state.y_axis_single = None
                    else: st.session_state.y_axis_single = st.selectbox("측정값 (기본 Y축)", available_measure_cols, index=available_measure_cols.index(st.session_state.y_axis_single) if st.session_state.y_axis_single in available_measure_cols else 0, key="y_single_grouped_select_v2_13")
                else: 
                    if chart_type in ['선 (Line)', '누적 영역 (Stacked Area)', '누적 막대 (Stacked Bar)']: 
                        available_y_multi = [h for h in numeric_headers if h != st.session_state.x_axis]
                        if not available_y_multi: st.warning(f"{chart_type}에 사용할 적절한 숫자형 Y축 열이 없습니다."); st.session_state.y_axis_multiple = []
                        else:
                            current_y_multi = [val for val in st.session_state.y_axis_multiple if val in available_y_multi]
                            if not current_y_multi and available_y_multi: current_y_multi = [available_y_multi[0]]
                            st.session_state.y_axis_multiple = st.multiselect("기본 Y축 (다중 가능)", available_y_multi, default=current_y_multi, key="y_multi_select_v2_13")
                    elif chart_type == '막대 (Bar)': 
                        available_y_single_bar = [h for h in numeric_headers if h != st.session_state.x_axis]
                        if not available_y_single_bar: st.warning("막대 그래프에 사용할 적절한 숫자형 Y축 열이 없습니다."); st.session_state.y_axis_single = None
                        else: st.session_state.y_axis_single = st.selectbox("기본 Y축", available_y_single_bar, index=available_y_single_bar.index(st.session_state.y_axis_single) if st.session_state.y_axis_single in available_y_single_bar else 0, key="y_single_bar_select_v2_13")
                
                if not is_stacked_chart: 
                    primary_y_selection_for_secondary = []
                    if group_by_col != "None":
                        if st.session_state.y_axis_single: primary_y_selection_for_secondary = [st.session_state.y_axis_single]
                    else: 
                        if chart_type == '선 (Line)': primary_y_selection_for_secondary = st.session_state.y_axis_multiple
                        elif chart_type == '막대 (Bar)' and st.session_state.y_axis_single: primary_y_selection_for_secondary = [st.session_state.y_axis_single]
                    secondary_y_options = ["None"] + [h for h in numeric_headers if h != st.session_state.x_axis and h != group_by_col and h not in primary_y_selection_for_secondary]
                    st.session_state.y_axis_secondary = st.selectbox("보조 Y축 (선택)", secondary_y_options, index=secondary_y_options.index(st.session_state.y_axis_secondary) if st.session_state.y_axis_secondary in secondary_y_options else 0, key="y_secondary_select_v2_13")
                else: st.session_state.y_axis_secondary = "None"

            elif is_distribution_chart: 
                if chart_type == '히스토그램 (Histogram)':
                    st.session_state.y_axis_multiple = st.multiselect("값 열 선택 (하나 이상, 숫자형)", numeric_headers, default=[col for col in st.session_state.y_axis_multiple if col in numeric_headers] or ([numeric_headers[0]] if numeric_headers else []), key="hist_value_cols_v2_13")
                    hist_color_options = ["None"] + [h for h in headers if h not in st.session_state.y_axis_multiple] 
                    st.session_state.group_by_col = st.selectbox("색상 구분 열 (선택)", hist_color_options, index=hist_color_options.index(st.session_state.group_by_col) if st.session_state.group_by_col in hist_color_options else 0, key="hist_color_col_v2_13")
                    st.session_state.hist_bins = st.number_input("구간(Bin) 개수 (선택)", min_value=1, value=st.session_state.hist_bins if st.session_state.hist_bins else 20, step=1, key="hist_bins_v2_13")
                elif chart_type == '박스 플롯 (Box Plot)':
                    st.session_state.y_axis_multiple = st.multiselect("Y축 값 열 선택 (하나 이상, 숫자형)", numeric_headers, default=[col for col in st.session_state.y_axis_multiple if col in numeric_headers] or ([numeric_headers[0]] if numeric_headers else []), key="box_y_cols_v2_13")
                    box_x_options = ["None"] + [h for h in headers if h not in st.session_state.y_axis_multiple] 
                    st.session_state.x_axis = st.selectbox("X축 범주 열 (선택)", box_x_options, index=box_x_options.index(st.session_state.x_axis) if st.session_state.x_axis in box_x_options else 0, key="box_x_col_v2_13")
                    box_color_options = ["None"] + [h for h in headers if h not in st.session_state.y_axis_multiple and h != st.session_state.x_axis] 
                    st.session_state.group_by_col = st.selectbox("색상 구분 열 (선택)", box_color_options, index=box_color_options.index(st.session_state.group_by_col) if st.session_state.group_by_col in box_color_options else 0, key="box_color_col_v2_13")
                    st.session_state.box_points = st.selectbox("표시할 포인트", ["outliers", "all", "suspectedoutliers", False], index=["outliers", "all", "suspectedoutliers", False].index(st.session_state.box_points), key="box_points_v2_13")
                elif chart_type == '밀도 플롯 (Density Plot)':
                    st.session_state.density_value_cols = st.multiselect("값 열 선택 (하나 이상, 숫자형)", numeric_headers, default=st.session_state.density_value_cols if all(item in numeric_headers for item in st.session_state.density_value_cols) else ([numeric_headers[0]] if numeric_headers else []), key="density_value_cols_v2_13")
                    density_color_options = ["None"] + [h for h in headers if h not in st.session_state.density_value_cols] 
                    st.session_state.density_color_col = st.selectbox("색상 구분 열 (선택)", density_color_options, index=density_color_options.index(st.session_state.density_color_col) if st.session_state.density_color_col in density_color_options else 0, key="density_color_col_v2_13")
                st.session_state.agg_method = 'Sum'; st.session_state.y_axis_secondary = "None" 
            
            elif is_relationship_chart: 
                st.session_state.scatter_x_col = st.selectbox("X축 (숫자형)", numeric_headers, index=numeric_headers.index(st.session_state.scatter_x_col) if st.session_state.scatter_x_col in numeric_headers else 0, key="scatter_x_v2_13")
                st.session_state.scatter_y_col = st.selectbox("Y축 (숫자형)", numeric_headers, index=numeric_headers.index(st.session_state.scatter_y_col) if st.session_state.scatter_y_col in numeric_headers else (1 if len(numeric_headers)>1 else 0), key="scatter_y_v2_13")
                color_options = ["None"] + headers 
                st.session_state.scatter_color_col = st.selectbox("색상 구분 열 (선택)", color_options, index=color_options.index(st.session_state.scatter_color_col) if st.session_state.scatter_color_col in color_options else 0, key="scatter_color_v2_13")
                if chart_type == '버블 차트 (Bubble Chart)':
                    size_options = ["None"] + numeric_headers 
                    st.session_state.scatter_size_col = st.selectbox("버블 크기 열 (선택, 숫자형)", size_options, index=size_options.index(st.session_state.scatter_size_col) if st.session_state.scatter_size_col in size_options else 0, key="scatter_size_v2_13")
                else: st.session_state.scatter_size_col = "None" 
                hover_name_options = ["None"] + headers
                st.session_state.scatter_hover_name_col = st.selectbox("마우스 오버 시 이름 표시 열 (선택)", hover_name_options, index=hover_name_options.index(st.session_state.scatter_hover_name_col) if st.session_state.scatter_hover_name_col in hover_name_options else 0, key="scatter_hover_v2_13")
                st.session_state.agg_method = 'Sum'; st.session_state.group_by_col = "None"; st.session_state.y_axis_secondary = "None"

            elif is_radar_chart:
                st.session_state.radar_category_col = st.selectbox("범주/그룹 열 (Theta 그룹)", headers, index=headers.index(st.session_state.radar_category_col) if st.session_state.radar_category_col in headers else 0, key="radar_cat_v2_13") 
                available_radar_values = [h for h in numeric_headers if h != st.session_state.radar_category_col]
                st.session_state.radar_value_cols = st.multiselect("값 열 선택 (Spokes - 여러개, 숫자형)", available_radar_values, default=[col for col in st.session_state.radar_value_cols if col in available_radar_values] or (available_radar_values[:1] if available_radar_values else []), key="radar_val_v2_13")
                st.session_state.agg_method = 'Sum'; st.session_state.group_by_col = "None"; st.session_state.y_axis_secondary = "None"; st.session_state.x_axis = None 

            elif is_heatmap_chart:
                st.session_state.heatmap_corr_cols = st.multiselect("상관관계 분석 대상 열 (여러개, 숫자형)", numeric_headers, default=[col for col in st.session_state.heatmap_corr_cols if col in numeric_headers] or numeric_headers[:min(len(numeric_headers), 5)], key="heatmap_cols_v2_13")
                st.session_state.agg_method = 'Sum'; st.session_state.group_by_col = "None"; st.session_state.y_axis_secondary = "None"; st.session_state.x_axis = None

            else: # 파이 차트
                st.session_state.pie_name_col = st.selectbox("레이블(이름) 열", headers, index=headers.index(st.session_state.pie_name_col) if st.session_state.pie_name_col in headers else 0, key="pie_name_select_v2_13") 
                available_pie_values = [h for h in numeric_headers if h != st.session_state.pie_name_col]
                if not available_pie_values: st.warning("파이 차트에 사용할 적절한 숫자형 값 열이 없습니다."); st.session_state.pie_value_col = None
                else: st.session_state.pie_value_col = st.selectbox("값 열", available_pie_values, index=available_pie_values.index(st.session_state.pie_value_col) if st.session_state.pie_value_col in available_pie_values else 0, key="pie_value_select_v2_13")
                st.session_state.group_by_col = "None"; st.session_state.y_axis_secondary = "None"; st.session_state.agg_method = 'Sum'
            # --- 복원된 차트 상세 설정 UI 끝 ---
        st.divider()
        st.markdown("<h4>3. 데이터 정제</h4>", unsafe_allow_html=True)
        with st.expander("결측치 처리", expanded=False):
            st.session_state.mv_selected_cols = st.multiselect("대상 열 선택 (결측치)", options=headers, default=[col for col in st.session_state.mv_selected_cols if col in headers], key="mv_target_cols_v2_13")
            mv_method_options = ["결측치가 있는 행 전체 제거", "평균값으로 대체 (숫자형 전용)", "중앙값으로 대체 (숫자형 전용)", "최빈값으로 대체", "특정 값으로 채우기"]
            st.session_state.mv_method = st.selectbox("처리 방법", options=mv_method_options, index=mv_method_options.index(st.session_state.mv_method), key="mv_method_v2_13")
            if st.session_state.mv_method == "특정 값으로 채우기": st.session_state.mv_specific_value = st.text_input("채울 특정 값", value=st.session_state.mv_specific_value, key="mv_specific_val_v2_13")
            if st.button("결측치 처리 적용", key="apply_mv_button_v2_13"): 
                if st.session_state.df is not None and st.session_state.mv_selected_cols:
                    df_processed = st.session_state.df_raw_uploaded.copy() 
                    if st.session_state.mv_method == "결측치가 있는 행 전체 제거":
                        df_processed.dropna(subset=st.session_state.mv_selected_cols, inplace=True)
                    st.session_state.df_raw_uploaded = df_processed 
                    st.success("결측치 처리 설정이 저장되었습니다. 전체 재처리가 필요합니다.")
                apply_all_processing_steps() 
                st.experimental_rerun()

        with st.expander("이상치 처리", expanded=False):
            if st.button("이상치 처리 적용", key="apply_ot_button_v2_13"): 
                apply_all_processing_steps()
                st.experimental_rerun()

        with st.expander("중복 데이터 처리", expanded=False):
            if st.button("중복 데이터 처리 적용", key="apply_dd_button_v2_13"): 
                apply_all_processing_steps()
                st.experimental_rerun()
        st.divider()

        st.markdown("<h4>4. 데이터 변환</h4>", unsafe_allow_html=True)
        with st.expander("필터링", expanded=False):
            if st.button("필터 적용", key="apply_filters_v2_13"): 
                apply_all_processing_steps()
                st.experimental_rerun()

        with st.expander("정렬", expanded=False):
            if st.button("정렬 적용", key="apply_sorts_v2_13"): 
                apply_all_processing_steps()
                st.experimental_rerun()
        st.divider()
        
        st.markdown("<h4>5. 데이터 구조 변경</h4>", unsafe_allow_html=True)
        with st.expander("피벗팅 (Pivoting)", expanded=False):
            if st.button("피벗 적용", key="apply_pivot_v2_13"): 
                apply_all_processing_steps()
                st.experimental_rerun()

        with st.expander("언피벗팅 (Unpivoting / Melt)", expanded=False):
            if st.button("언피벗 적용", key="apply_unpivot_v2_13"): 
                apply_all_processing_steps()
                st.experimental_rerun()
        st.divider()

        st.markdown("<h4>6. 파생 변수 생성</h4>", unsafe_allow_html=True)
        with st.expander("수식 기반 파생 변수 (간단)", expanded=False):
            st.session_state.derived_var_name = st.text_input("새 변수 이름", value=st.session_state.derived_var_name, key="derived_var_name_v2_13")
            st.session_state.derived_var_formula = st.text_area("수식 입력", value=st.session_state.derived_var_formula, height=100, key="derived_var_formula_v2_13", placeholder="예: (열1 + 열2) / 열3")
            if st.button("파생 변수 생성 적용", key="apply_derived_var_v2_13"): 
                if st.session_state.df_raw_uploaded is not None and st.session_state.derived_var_name.strip() and st.session_state.derived_var_formula.strip():
                    try:
                        temp_df = st.session_state.df_raw_uploaded.copy()
                        temp_df[st.session_state.derived_var_name.strip()] = temp_df.eval(st.session_state.derived_var_formula)
                        st.session_state.df_raw_uploaded = temp_df 
                        st.success(f"수식 기반 파생 변수 '{st.session_state.derived_var_name.strip()}' 임시 적용.")
                    except Exception as e:
                        st.error(f"수식 기반 파생변수 생성 오류: {e}")
                else:
                    st.warning("새 변수 이름과 수식을 모두 입력해주세요.")
                apply_all_processing_steps() 
                st.experimental_rerun()

        with st.expander("고급 파생 변수 편집기 (GUI)", expanded=st.session_state.show_adv_derived_var_builder):
            st.write("GUI를 사용하여 조건부 규칙 또는 창 함수 기반의 파생 변수를 생성 및 관리합니다.")
            if st.button("➕ 새 고급 파생 변수 만들기", key="add_new_adv_derived_var_btn"):
                st.session_state.show_adv_derived_var_builder = True
                st.session_state.editing_adv_derived_var_name = None
                st.session_state.adv_builder_var_name = ""
                st.session_state.adv_builder_var_type = 'conditional'
                first_col_for_builder = headers[0] if headers else ''
                st.session_state.adv_builder_conditional_rules = [{'id': str(uuid.uuid4()), 'variable1': first_col_for_builder, 'operator1': '==', 'value1': '', 'logical_op': '', 'variable2': '', 'operator2': '==', 'value2': '', 'then_value': ''}]
                st.session_state.adv_builder_else_value = ''
                st.session_state.adv_builder_window_func = ADV_WINDOW_FUNCTIONS[0]
                st.session_state.adv_builder_window_target_col = numeric_headers[0] if numeric_headers else ''
                st.session_state.adv_builder_window_partition_by = []
                st.session_state.adv_builder_window_order_by_col = ''
                st.session_state.adv_builder_window_order_by_dir = ADV_SORT_DIRECTIONS[0]
                st.session_state.adv_builder_window_lag_lead_offset = 1
                st.session_state.adv_builder_window_lag_lead_default = ''
                if st.session_state.editing_adv_derived_var_name: 
                    for flag_key_suffix in ["rules_loaded_for_", "win_loaded_for_"]:
                        flag_key = f"adv_{flag_key_suffix}{st.session_state.editing_adv_derived_var_name}"
                        if st.session_state.get(flag_key): del st.session_state[flag_key]
                st.experimental_rerun()

            if st.session_state.show_adv_derived_var_builder:
                is_editing_adv = st.session_state.editing_adv_derived_var_name is not None
                form_title = f"'{st.session_state.editing_adv_derived_var_name}' 수정" if is_editing_adv else "새 고급 파생 변수 정의"
                st.subheader(form_title)
                current_adv_def = {}
                if is_editing_adv:
                    current_adv_def = st.session_state.advanced_derived_definitions.get(st.session_state.editing_adv_derived_var_name, {})
                var_name_default = st.session_state.editing_adv_derived_var_name if is_editing_adv else st.session_state.adv_builder_var_name
                st.session_state.adv_builder_var_name = st.text_input("파생 변수 이름:", value=var_name_default, key="adv_builder_name_input")
                var_type_default = current_adv_def.get('type', st.session_state.adv_builder_var_type)
                st.session_state.adv_builder_var_type = st.selectbox("유형:", ['conditional', 'window'],
                                                                    index=['conditional', 'window'].index(var_type_default),
                                                                    key="adv_builder_type_select")
                available_vars_for_adv = get_all_available_variables_for_derived()
                if st.session_state.adv_builder_var_type == 'conditional':
                    if is_editing_adv and current_adv_def.get('type') == 'conditional':
                        if not st.session_state.get(f"adv_rules_loaded_for_{st.session_state.editing_adv_derived_var_name}", False):
                            st.session_state.adv_builder_conditional_rules = [dict(r, id=r.get('id', str(uuid.uuid4()))) for r in current_adv_def.get('rules', [])]
                            st.session_state.adv_builder_else_value = current_adv_def.get('else_value', '')
                            st.session_state[f"adv_rules_loaded_for_{st.session_state.editing_adv_derived_var_name}"] = True
                    elif not is_editing_adv and not st.session_state.adv_builder_conditional_rules :
                         st.session_state.adv_builder_conditional_rules = [{'id': str(uuid.uuid4()), 'variable1': available_vars_for_adv[0] if available_vars_for_adv else '', 'operator1': '==', 'value1': '', 'logical_op': '', 'variable2': '', 'operator2': '==', 'value2': '', 'then_value': ''}]
                         st.session_state.adv_builder_else_value = ''
                    num_cond_rules = len(st.session_state.adv_builder_conditional_rules)
                    for i in range(num_cond_rules):
                        rule = st.session_state.adv_builder_conditional_rules[i]
                        if 'id' not in rule: rule['id'] = str(uuid.uuid4())
                        st.markdown(f"--- **조건 {i+1} ({'IF' if i==0 else 'ELSE IF'})** ---")
                        cols_adv_cond = st.columns([3, 2, 3, 2, 3, 2, 3, 3, 0.8]) 
                        rule['variable1'] = cols_adv_cond[0].selectbox("변수1", available_vars_for_adv, index=available_vars_for_adv.index(rule.get('variable1', available_vars_for_adv[0] if available_vars_for_adv else '')) if rule.get('variable1') in available_vars_for_adv else 0, key=f"adv_rule{rule['id']}_var1")
                        rule['operator1'] = cols_adv_cond[1].selectbox("연산자1", ADV_COMPARISON_OPERATORS, index=ADV_COMPARISON_OPERATORS.index(rule.get('operator1', '==')), key=f"adv_rule{rule['id']}_op1")
                        var1_type_adv = get_variable_type_for_derived(rule['variable1'])
                        if rule['operator1'] in ['isnull', 'notnull']: rule['value1'] = ""; cols_adv_cond[2].markdown(" ") 
                        elif var1_type_adv == 'number': rule['value1'] = cols_adv_cond[2].number_input("값1", value=float(rule.get('value1', 0)) if str(rule.get('value1','0')).replace('.','',1).isdigit() else 0.0, key=f"adv_rule{rule['id']}_val1", format="%g", label_visibility="collapsed")
                        else: rule['value1'] = cols_adv_cond[2].text_input("값1", value=str(rule.get('value1', '')), key=f"adv_rule{rule['id']}_val1", label_visibility="collapsed")
                        rule['logical_op'] = cols_adv_cond[3].selectbox("논리", ["", "AND", "OR"], index=["", "AND", "OR"].index(rule.get('logical_op', "")), key=f"adv_rule{rule['id']}_logop")
                        if rule['logical_op']:
                            rule['variable2'] = cols_adv_cond[4].selectbox("변수2", available_vars_for_adv, index=available_vars_for_adv.index(rule.get('variable2', available_vars_for_adv[0] if available_vars_for_adv else '')) if rule.get('variable2') in available_vars_for_adv else 0, key=f"adv_rule{rule['id']}_var2")
                            rule['operator2'] = cols_adv_cond[5].selectbox("연산자2", ADV_COMPARISON_OPERATORS, index=ADV_COMPARISON_OPERATORS.index(rule.get('operator2', '==')), key=f"adv_rule{rule['id']}_op2")
                            var2_type_adv = get_variable_type_for_derived(rule['variable2'])
                            if rule['operator2'] in ['isnull', 'notnull']: rule['value2'] = ""; cols_adv_cond[6].markdown(" ")
                            elif var2_type_adv == 'number': rule['value2'] = cols_adv_cond[6].number_input("값2", value=float(rule.get('value2',0)) if str(rule.get('value2','0')).replace('.','',1).isdigit() else 0.0, key=f"adv_rule{rule['id']}_val2", format="%g", label_visibility="collapsed")
                            else: rule['value2'] = cols_adv_cond[6].text_input("값2", value=str(rule.get('value2','')), key=f"adv_rule{rule['id']}_val2", label_visibility="collapsed")
                        else: 
                            rule['variable2'], rule['operator2'], rule['value2'] = '', '', ''
                            cols_adv_cond[4].empty(); cols_adv_cond[5].empty(); cols_adv_cond[6].empty()
                        rule['then_value'] = cols_adv_cond[7].text_input("THEN 값", value=str(rule.get('then_value','')), key=f"adv_rule{rule['id']}_then")
                        if num_cond_rules > 1 and cols_adv_cond[8].button("➖", key=f"adv_remove_cond_rule_{rule['id']}", help="이 조건 삭제"):
                            st.session_state.adv_builder_conditional_rules.pop(i)
                            st.experimental_rerun()
                    if st.button("➕ ELSE IF 조건 추가", key="adv_add_cond_rule_btn"):
                        st.session_state.adv_builder_conditional_rules.append({'id': str(uuid.uuid4()), 'variable1': available_vars_for_adv[0] if available_vars_for_adv else '', 'operator1': '==', 'value1': '', 'logical_op': '', 'variable2': '', 'operator2': '==', 'value2': '', 'then_value': ''})
                        st.experimental_rerun()
                    st.session_state.adv_builder_else_value = st.text_input("ELSE 값 (모든 조건 불일치 시):", value=(st.session_state.adv_builder_else_value), key="adv_builder_else_input")
                elif st.session_state.adv_builder_var_type == 'window':
                    win_conf_default = current_adv_def.get('config', {}) if is_editing_adv else {}
                    if is_editing_adv : 
                        if not st.session_state.get(f"adv_win_loaded_for_{st.session_state.editing_adv_derived_var_name}", False):
                            st.session_state.adv_builder_window_func = win_conf_default.get('function', ADV_WINDOW_FUNCTIONS[0])
                            st.session_state.adv_builder_window_target_col = win_conf_default.get('target_col', '')
                            st.session_state.adv_builder_window_partition_by = list(win_conf_default.get('partition_by', []))
                            st.session_state.adv_builder_window_order_by_col = win_conf_default.get('order_by_col', '')
                            st.session_state.adv_builder_window_order_by_dir = win_conf_default.get('order_by_dir', ADV_SORT_DIRECTIONS[0])
                            st.session_state.adv_builder_window_lag_lead_offset = win_conf_default.get('offset', 1)
                            st.session_state.adv_builder_window_lag_lead_default = win_conf_default.get('default_value', '')
                            st.session_state[f"adv_win_loaded_for_{st.session_state.editing_adv_derived_var_name}"] = True
                    st.session_state.adv_builder_window_func = st.selectbox("함수:", ADV_WINDOW_FUNCTIONS, index=ADV_WINDOW_FUNCTIONS.index(st.session_state.adv_builder_window_func), key="adv_win_func_select")
                    numeric_vars_for_win = [col for col in available_vars_for_adv if get_variable_type_for_derived(col) == 'number']
                    target_vars_options_win = numeric_vars_for_win if st.session_state.adv_builder_window_func in ['SUM', 'AVG', 'MIN', 'MAX'] else available_vars_for_adv
                    if st.session_state.adv_builder_window_func in ['SUM', 'AVG', 'MIN', 'MAX', 'LAG', 'LEAD', 'COUNT']:
                        st.session_state.adv_builder_window_target_col = st.selectbox(f"대상 변수 ({st.session_state.adv_builder_window_func} 적용):", options=[""] + target_vars_options_win, 
                                                                                        index=([""] + target_vars_options_win).index(st.session_state.adv_builder_window_target_col) if st.session_state.adv_builder_window_target_col in ([""] + target_vars_options_win) else 0,
                                                                                        key="adv_win_target_select")
                    if st.session_state.adv_builder_window_func in ['LAG', 'LEAD']:
                        st.session_state.adv_builder_window_lag_lead_offset = st.number_input("Offset:", min_value=1, value=int(st.session_state.adv_builder_window_lag_lead_offset), step=1, key="adv_win_lag_offset")
                        st.session_state.adv_builder_window_lag_lead_default = st.text_input("기본값 (Offset 벗어날 경우, 비워두면 None):", value=str(st.session_state.adv_builder_window_lag_lead_default), key="adv_win_lag_default")
                    st.session_state.adv_builder_window_partition_by = st.multiselect("PARTITION BY (그룹화 기준, 선택):", available_vars_for_adv, default=st.session_state.adv_builder_window_partition_by, key="adv_win_partition_select")
                    st.session_state.adv_builder_window_order_by_col = st.selectbox("ORDER BY (정렬 기준, 선택):", [""] + available_vars_for_adv, 
                                                                                    index=([""] + available_vars_for_adv).index(st.session_state.adv_builder_window_order_by_col) if st.session_state.adv_builder_window_order_by_col in ([""] + available_vars_for_adv) else 0,
                                                                                    key="adv_win_orderby_col_select")
                    if st.session_state.adv_builder_window_order_by_col:
                        st.session_state.adv_builder_window_order_by_dir = st.selectbox("정렬 방향:", ADV_SORT_DIRECTIONS, index=ADV_SORT_DIRECTIONS.index(st.session_state.adv_builder_window_order_by_dir), key="adv_win_orderby_dir_select")

                adv_btn_cols = st.columns(2)
                if adv_btn_cols[0].button("💾 고급 파생 변수 저장", type="primary", use_container_width=True, key="save_adv_derived_var_btn"):
                    new_adv_var_name_val = st.session_state.adv_builder_var_name.strip()
                    if not new_adv_var_name_val: st.error("파생 변수 이름을 입력해야 합니다.")
                    elif not is_editing_adv and new_adv_var_name_val in get_all_available_variables_for_derived(): st.error(f"'{new_adv_var_name_val}' 이름의 변수가 이미 존재합니다.")
                    elif is_editing_adv and new_adv_var_name_val != st.session_state.editing_adv_derived_var_name and new_adv_var_name_val in get_all_available_variables_for_derived(): st.error(f"새 이름 '{new_adv_var_name_val}'의 변수가 이미 존재합니다.")
                    else:
                        adv_definition_to_save = {'type': st.session_state.adv_builder_var_type}
                        valid_def = True
                        if st.session_state.adv_builder_var_type == 'conditional':
                            if not st.session_state.adv_builder_conditional_rules: st.error("최소 하나 이상의 조건 규칙이 필요합니다."); valid_def = False
                            for r_idx, r_adv in enumerate(st.session_state.adv_builder_conditional_rules):
                                if not r_adv.get('variable1') or not r_adv.get('then_value','').strip(): st.error(f"조건 {r_idx+1}: 변수1과 THEN 값은 필수입니다."); valid_def = False; break
                                if r_adv.get('logical_op') and not r_adv.get('variable2'): st.error(f"조건 {r_idx+1}: {r_adv['logical_op']} 시 변수2는 필수입니다."); valid_def = False; break
                            if not st.session_state.adv_builder_else_value.strip() and valid_def: st.error("ELSE 값은 필수입니다."); valid_def = False
                            if valid_def: adv_definition_to_save['rules'] = st.session_state.adv_builder_conditional_rules; adv_definition_to_save['else_value'] = st.session_state.adv_builder_else_value
                        elif st.session_state.adv_builder_var_type == 'window':
                            adv_definition_to_save['config'] = {
                                'function': st.session_state.adv_builder_window_func,
                                'target_col': st.session_state.adv_builder_window_target_col,
                                'partition_by': st.session_state.adv_builder_window_partition_by,
                                'order_by_col': st.session_state.adv_builder_window_order_by_col,
                                'order_by_dir': st.session_state.adv_builder_window_order_by_dir if st.session_state.adv_builder_window_order_by_col else '',
                                'offset': st.session_state.adv_builder_window_lag_lead_offset if st.session_state.adv_builder_window_func in ['LAG', 'LEAD'] else None,
                                'default_value': st.session_state.adv_builder_window_lag_lead_default if st.session_state.adv_builder_window_func in ['LAG', 'LEAD'] and st.session_state.adv_builder_window_lag_lead_default.strip() else None,
                            }
                            if st.session_state.adv_builder_window_func in ['SUM', 'AVG', 'MIN', 'MAX', 'LAG', 'LEAD'] and not adv_definition_to_save['config']['target_col']:
                                st.error(f"{st.session_state.adv_builder_window_func} 함수는 대상 변수가 필요합니다."); valid_def = False
                        if valid_def:
                            if is_editing_adv and new_adv_var_name_val != st.session_state.editing_adv_derived_var_name: 
                                if st.session_state.editing_adv_derived_var_name in st.session_state.advanced_derived_definitions:
                                    del st.session_state.advanced_derived_definitions[st.session_state.editing_adv_derived_var_name]
                                for flag_key_suffix in ["rules_loaded_for_", "win_loaded_for_"]: # 이전 이름의 플래그 삭제
                                    old_flag_key = f"adv_{flag_key_suffix}{st.session_state.editing_adv_derived_var_name}"
                                    if st.session_state.get(old_flag_key): del st.session_state[old_flag_key]
                            st.session_state.advanced_derived_definitions[new_adv_var_name_val] = adv_definition_to_save
                            apply_all_processing_steps() 
                            st.session_state.show_adv_derived_var_builder = False
                            st.session_state.editing_adv_derived_var_name = None
                            for flag_key_suffix in ["rules_loaded_for_", "win_loaded_for_"]: # 새 이름의 플래그도 정리
                                new_flag_key = f"adv_{flag_key_suffix}{new_adv_var_name_val}"
                                if st.session_state.get(new_flag_key): del st.session_state[new_flag_key]
                            st.success(f"고급 파생 변수 '{new_adv_var_name_val}'이(가) 저장되었습니다.")
                            st.experimental_rerun()
                if adv_btn_cols[1].button("🚫 고급 편집기 닫기", use_container_width=True, key="cancel_adv_derived_var_btn"):
                    st.session_state.show_adv_derived_var_builder = False
                    if st.session_state.editing_adv_derived_var_name: # 수정 중 취소 시 플래그 정리
                        for flag_key_suffix in ["rules_loaded_for_", "win_loaded_for_"]:
                            flag_key = f"adv_{flag_key_suffix}{st.session_state.editing_adv_derived_var_name}"
                            if st.session_state.get(flag_key): del st.session_state[flag_key]
                    st.session_state.editing_adv_derived_var_name = None
                    st.experimental_rerun()
            st.markdown("--- **생성된 고급 파생 변수 목록** ---")
            if not st.session_state.advanced_derived_definitions:
                st.caption("아직 생성된 고급 파생 변수가 없습니다.")
            else:
                for adv_var_name_item, adv_def_item in list(st.session_state.advanced_derived_definitions.items()):
                    cols_adv_item = st.columns([3,1,1])
                    cols_adv_item[0].markdown(f"**{adv_var_name_item}** (`{adv_def_item['type']}`)")
                    if cols_adv_item[1].button("✏️", key=f"edit_adv_{adv_var_name_item}", help="이 고급 파생 변수 수정"):
                        st.session_state.show_adv_derived_var_builder = True
                        st.session_state.editing_adv_derived_var_name = adv_var_name_item
                        # 로드 플래그 삭제하여 수정 시 빌더가 새 값으로 채워지도록 유도
                        for flag_key_suffix in ["rules_loaded_for_", "win_loaded_for_"]:
                            flag_key = f"adv_{flag_key_suffix}{adv_var_name_item}"
                            if st.session_state.get(flag_key): del st.session_state[flag_key]
                        st.experimental_rerun() 
                    if cols_adv_item[2].button("🗑️", key=f"delete_adv_{adv_var_name_item}", help="이 고급 파생 변수 삭제"):
                        if adv_var_name_item in st.session_state.advanced_derived_definitions:
                            del st.session_state.advanced_derived_definitions[adv_var_name_item]
                        if st.session_state.editing_adv_derived_var_name == adv_var_name_item: 
                            st.session_state.show_adv_derived_var_builder = False
                            st.session_state.editing_adv_derived_var_name = None
                        apply_all_processing_steps() 
                        st.success(f"고급 파생 변수 '{adv_var_name_item}'이(가) 삭제되었습니다.")
                        st.experimental_rerun()

    elif uploaded_file and not st.session_state.data_loaded_success:
        st.sidebar.warning("데이터 로드에 실패했습니다. 파일을 확인하고 다시 시도해주세요.")
    else:
        st.sidebar.info("CSV 파일을 업로드하거나 BigQuery에서 데이터를 로드하면 추가 옵션이 나타납니다.")

if df is None:
    st.info("데이터를 시각화하려면 왼쪽 사이드바에서 CSV 파일을 업로드하거나 BigQuery에서 데이터를 로드해주세요.")
elif not st.session_state.data_loaded_success:
    st.warning("데이터 로드에 실패했습니다. 파일을 확인하고 사이드바에서 다시 시도해주세요.")
else:
    st.subheader("데이터 미리보기 (상위 5행)")
    st.dataframe(df.head())
    st.subheader("시각화 결과")
    chart_placeholder = st.empty()
    try:
        fig = None
        chart_type = st.session_state.chart_type
        x_axis = st.session_state.x_axis
        y_single = st.session_state.y_axis_single
        y_multiple = st.session_state.y_axis_multiple
        y_secondary = st.session_state.y_axis_secondary
        group_by_col = st.session_state.group_by_col
        agg_method = st.session_state.agg_method
        pie_name_col = st.session_state.pie_name_col
        pie_value_col = st.session_state.pie_value_col
        density_values = st.session_state.density_value_cols
        density_color = st.session_state.density_color_col if st.session_state.density_color_col != "None" else None
        radar_cat = st.session_state.radar_category_col
        radar_vals = st.session_state.radar_value_cols
        heatmap_cols = st.session_state.heatmap_corr_cols
        scatter_x = st.session_state.scatter_x_col
        scatter_y = st.session_state.scatter_y_col
        scatter_color = st.session_state.scatter_color_col if st.session_state.scatter_color_col != "None" else None
        scatter_size = st.session_state.scatter_size_col if st.session_state.scatter_size_col != "None" else None
        scatter_hover = st.session_state.scatter_hover_name_col if st.session_state.scatter_hover_name_col != "None" else None
        is_pie_chart = (chart_type == '파이 (Pie)')
        is_distribution_chart = (chart_type in ['히스토그램 (Histogram)', '박스 플롯 (Box Plot)', '밀도 플롯 (Density Plot)'])
        is_relationship_chart = (chart_type in ['분산형 차트 (Scatter Plot)', '버블 차트 (Bubble Chart)'])
        is_radar_chart = (chart_type == '레이더 차트 (Radar Chart)')
        is_heatmap_chart = (chart_type == '히트맵 (Heatmap - 상관관계)')

        if is_pie_chart:
            if not pie_name_col or not pie_value_col: chart_placeholder.warning("파이 차트의 레이블 열과 값 열을 모두 선택해주세요."); st.stop()
            if pie_value_col not in st.session_state.numeric_headers: chart_placeholder.error(f"파이 차트의 값 열 ('{pie_value_col}')은 숫자형이어야 합니다."); st.stop()
        elif is_distribution_chart:
            if chart_type == '히스토그램 (Histogram)' and (not y_multiple or not all(y in st.session_state.numeric_headers for y in y_multiple)): chart_placeholder.warning(f"히스토그램에 사용할 숫자형 값 열(들)을 선택해주세요."); st.stop()
            if chart_type == '박스 플롯 (Box Plot)' and (not y_multiple or not all(y in st.session_state.numeric_headers for y in y_multiple)): chart_placeholder.warning(f"박스 플롯에 사용할 숫자형 Y축 값 열(들)을 선택해주세요."); st.stop()
            if chart_type == '밀도 플롯 (Density Plot)' and (not density_values or not all(y in st.session_state.numeric_headers for y in density_values)): chart_placeholder.warning("밀도 플롯에 사용할 숫자형 값 열(들)을 선택해주세요."); st.stop()
        elif is_relationship_chart:
            if not scatter_x or not scatter_y: chart_placeholder.warning("분산형/버블 차트의 X축과 Y축을 모두 선택해주세요 (숫자형)."); st.stop()
            if scatter_x not in st.session_state.numeric_headers or scatter_y not in st.session_state.numeric_headers: chart_placeholder.error("분산형/버블 차트의 X축과 Y축은 숫자형이어야 합니다."); st.stop()
            if chart_type == '버블 차트 (Bubble Chart)' and scatter_size and scatter_size not in st.session_state.numeric_headers: chart_placeholder.error("버블 차트의 크기 열은 숫자형이어야 합니다."); st.stop()
        elif is_radar_chart:
            if not radar_cat: chart_placeholder.warning("레이더 차트의 범주/그룹 열을 선택해주세요."); st.stop()
            if not radar_vals or not all(y in st.session_state.numeric_headers for y in radar_vals): chart_placeholder.warning("레이더 차트의 숫자형 값 열(Spokes)을 하나 이상 선택해주세요."); st.stop()
        elif is_heatmap_chart:
            if not heatmap_cols or len(heatmap_cols) < 2 or not all(y in st.session_state.numeric_headers for y in heatmap_cols): chart_placeholder.warning("히트맵에 사용할 숫자형 열을 2개 이상 선택해주세요."); st.stop()
        elif not x_axis:
            if headers: chart_placeholder.warning("X축을 선택해주세요.")
            st.stop()

        processed_df_for_chart = df.copy()
        agg_functions = {'Sum': 'sum', 'Mean': 'mean', 'Median': 'median'}
        current_agg_func = agg_functions[agg_method]
        y_val_for_chart = None
        y_multi_for_chart = []
        color_col_for_chart = None
        
        if chart_type == '파이 (Pie)':
            pie_data_agg = processed_df_for_chart.groupby(pie_name_col, as_index=False).agg({pie_value_col: current_agg_func})
            fig = px.pie(pie_data_agg, names=pie_name_col, values=pie_value_col, title=f"{pie_name_col} 별 {pie_value_col} 분포 ({agg_method} 기준)")
        elif chart_type == '히스토그램 (Histogram)':
            y_val_for_hist = y_multiple[0] if y_multiple else None
            if not y_val_for_hist : chart_placeholder.warning("히스토그램에 사용할 값 열을 선택해주세요."); st.stop()
            if len(y_multiple) > 1: st.info(f"히스토그램은 현재 하나의 값 열('{y_val_for_hist}')에 대해서만 그려집니다. 색상 구분 열을 활용하세요.")
            fig = px.histogram(processed_df_for_chart, x=y_val_for_hist, color=group_by_col if group_by_col != "None" else None, nbins=st.session_state.hist_bins, title=f"{y_val_for_hist}의 분포" + (f" (색상: {group_by_col})" if group_by_col != "None" else ""))
        elif chart_type == '박스 플롯 (Box Plot)':
            current_x_for_box = x_axis if x_axis != "None" else None
            current_color_for_box = group_by_col if group_by_col != "None" else None
            if not y_multiple: chart_placeholder.warning("박스 플롯에 사용할 Y축 값 열을 선택해주세요."); st.stop()
            fig = px.box(processed_df_for_chart, x=current_x_for_box, y=y_multiple, color=current_color_for_box, points=st.session_state.box_points, title="박스 플롯" + (f" (X: {current_x_for_box})" if current_x_for_box else "") + (f" (색상: {current_color_for_box})" if current_color_for_box else ""))
        elif chart_type == '밀도 플롯 (Density Plot)':
            if not density_values: chart_placeholder.warning("밀도 플롯에 사용할 값 열을 선택해주세요."); st.stop()
            if len(density_values) == 1 and not density_color:
                 fig = px.histogram(processed_df_for_chart, x=density_values[0], marginal="rug", histnorm='probability density', title=f"{density_values[0]}의 밀도 플롯")
            elif len(density_values) == 1 and density_color:
                 fig = px.histogram(processed_df_for_chart, x=density_values[0], color=density_color, marginal="rug", histnorm='probability density', barmode="overlay", opacity=0.7, title=f"{density_values[0]}의 밀도 플롯 (색상: {density_color})")
            else:
                melted_df_density = pd.melt(processed_df_for_chart, value_vars=density_values, var_name='변수', value_name='값')
                fig = px.histogram(melted_df_density, x='값', color='변수', marginal="rug", histnorm='probability density', barmode="overlay", opacity=0.7, title="선택된 열들의 밀도 플롯")
        elif chart_type == '레이더 차트 (Radar Chart)':
            df_for_radar = processed_df_for_chart
            if not radar_vals: chart_placeholder.warning("레이더 차트에 사용할 값 열을 선택해주세요."); st.stop()
            if radar_cat not in df_for_radar.columns: chart_placeholder.error(f"레이더 차트 범주 열 '{radar_cat}'이 데이터에 없습니다."); st.stop()
            if not df_for_radar.groupby(radar_cat).size().eq(1).all(): 
                st.info(f"레이더 차트: '{radar_cat}'별로 여러 행이 존재하여 각 값 열에 대해 평균값을 사용합니다.")
                df_for_radar = df_for_radar.groupby(radar_cat, as_index=False)[radar_vals].mean()
            fig_radar = go.Figure()
            for i, category_item in enumerate(df_for_radar[radar_cat].unique()):
                filtered_data = df_for_radar[df_for_radar[radar_cat] == category_item]
                if not filtered_data.empty:
                    r_values = filtered_data[radar_vals].iloc[0].tolist()
                    fig_radar.add_trace(go.Scatterpolar(r=r_values, theta=radar_vals, fill='toself', name=str(category_item), marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]))
            max_r_val = 0
            if radar_vals and not df_for_radar[radar_vals].empty:
                try: max_r_val = df_for_radar[radar_vals].max(numeric_only=True).max(numeric_only=True) 
                except: max_r_val = 1 
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_r_val if max_r_val > 0 else 1])), showlegend=True, title=f"레이더 차트 (범주: {radar_cat})")
            fig = fig_radar
        elif chart_type == '히트맵 (Heatmap - 상관관계)':
            if not heatmap_cols or len(heatmap_cols) < 2 : chart_placeholder.warning("히트맵에 사용할 숫자형 열을 2개 이상 선택해주세요."); st.stop()
            corr_matrix = processed_df_for_chart[heatmap_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="상관관계 히트맵")
        elif chart_type == '분산형 차트 (Scatter Plot)':
            fig = px.scatter(processed_df_for_chart, x=scatter_x, y=scatter_y, color=scatter_color, hover_name=scatter_hover, title=f"{scatter_x} vs {scatter_y} 관계")
        elif chart_type == '버블 차트 (Bubble Chart)':
            fig = px.scatter(processed_df_for_chart, x=scatter_x, y=scatter_y, color=scatter_color, size=scatter_size, hover_name=scatter_hover, title=f"{scatter_x} vs {scatter_y} (크기: {scatter_size}, 색상: {scatter_color})")
        else: 
            y_cols_to_aggregate = []
            grouping_cols = [x_axis] if x_axis else []
            if not grouping_cols: chart_placeholder.error("X축이 선택되지 않아 차트를 생성할 수 없습니다."); st.stop()
            if group_by_col != "None":
                if not y_single or y_single not in st.session_state.numeric_headers: chart_placeholder.warning(f"그룹화에 사용할 측정값(기본 Y축)을 선택해주세요 (숫자형). 현재 선택: {y_single}"); st.stop()
                y_cols_to_aggregate = [y_single]
                if group_by_col not in grouping_cols : grouping_cols.append(group_by_col)
                color_col_for_chart = group_by_col
                y_val_for_chart = y_single
            else:
                if chart_type in ['막대 (Bar)']:
                    if not y_single or y_single not in st.session_state.numeric_headers: chart_placeholder.warning(f"막대 차트의 Y축을 선택해주세요 (숫자형). 현재 선택: {y_single}"); st.stop()
                    y_cols_to_aggregate = [y_single]
                    y_val_for_chart = y_single
                elif chart_type in ['선 (Line)', '누적 막대 (Stacked Bar)', '누적 영역 (Stacked Area)']:
                    if not y_multiple or not all(y in st.session_state.numeric_headers for y in y_multiple): chart_placeholder.warning(f"{chart_type}의 Y축을 하나 이상 선택해주세요 (숫자형). 현재 선택: {y_multiple}"); st.stop()
                    y_cols_to_aggregate = y_multiple
                    y_multi_for_chart = y_multiple
            if y_cols_to_aggregate and grouping_cols:
                agg_dict = {y_col: current_agg_func for y_col in y_cols_to_aggregate}
                processed_df_for_chart = processed_df_for_chart.groupby(grouping_cols, as_index=False).agg(agg_dict)
            elif not y_cols_to_aggregate and chart_type != '파이 (Pie)': chart_placeholder.warning("차트에 표시할 Y축 숫자형 데이터를 선택해주세요."); st.stop()
            is_stacked_chart = (chart_type in ['누적 막대 (Stacked Bar)', '누적 영역 (Stacked Area)'])
            if y_secondary != "None" and not is_stacked_chart:
                if y_secondary not in st.session_state.numeric_headers: chart_placeholder.error(f"보조 Y축 열 ('{y_secondary}')은 숫자형이어야 합니다."); st.stop()
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                if group_by_col != "None" and y_val_for_chart:
                    unique_groups = processed_df_for_chart[group_by_col].unique()
                    for i, group_val in enumerate(unique_groups):
                        trace_data = processed_df_for_chart[processed_df_for_chart[group_by_col] == group_val]
                        if chart_type == '막대 (Bar)': fig.add_trace(go.Bar(x=trace_data[x_axis], y=trace_data[y_val_for_chart], name=f"{group_val} ({y_val_for_chart})", marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]), secondary_y=False)
                        elif chart_type == '선 (Line)': fig.add_trace(go.Scatter(x=trace_data[x_axis], y=trace_data[y_val_for_chart], mode='lines+markers', name=f"{group_val} ({y_val_for_chart})", line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)])), secondary_y=False)
                else:
                    if chart_type == '막대 (Bar)' and y_val_for_chart: fig.add_trace(go.Bar(x=processed_df_for_chart[x_axis], y=processed_df_for_chart[y_val_for_chart], name=y_val_for_chart), secondary_y=False)
                    elif chart_type == '선 (Line)' and y_multi_for_chart:
                        for i, y_col_line in enumerate(y_multi_for_chart): fig.add_trace(go.Scatter(x=processed_df_for_chart[x_axis], y=processed_df_for_chart[y_col_line], mode='lines+markers', name=y_col_line, line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)])), secondary_y=False)
                if x_axis in df.columns and y_secondary in df.columns: 
                    secondary_base_data = df.groupby(x_axis, as_index=False)[y_secondary].mean() 
                    unique_x_in_primary = processed_df_for_chart[x_axis].drop_duplicates().sort_values()
                    temp_secondary_df = pd.merge(pd.DataFrame({x_axis: unique_x_in_primary}), secondary_base_data, on=x_axis, how='left')
                    fig.add_trace(go.Scatter(x=temp_secondary_df[x_axis], y=temp_secondary_df[y_secondary], mode='lines+markers', name=f"{y_secondary} (보조)", yaxis='y2', line=dict(dash='dot')), secondary_y=True)
                title_y_primary_text = y_val_for_chart if y_val_for_chart else ', '.join(y_multi_for_chart)
                fig.update_layout(title_text=f"{x_axis} 별 {title_y_primary_text} 및 {y_secondary} (보조)")
                fig.update_yaxes(title_text=f"기본 Y ({agg_method})", secondary_y=False); fig.update_yaxes(title_text=f"{y_secondary} (보조, 평균)", secondary_y=True)
                if chart_type == '막대 (Bar)' and group_by_col != "None": fig.update_layout(barmode='group')
            else:
                y_plot_val = y_val_for_chart if y_val_for_chart else y_multi_for_chart
                if not y_plot_val: chart_placeholder.warning("Y축 값을 선택해주세요."); st.stop()
                if chart_type == '막대 (Bar)': fig = px.bar(processed_df_for_chart, x=x_axis, y=y_plot_val, color=color_col_for_chart, barmode='group' if color_col_for_chart else 'relative', title=f"{x_axis} 별 Y값 ({agg_method})")
                elif chart_type == '누적 막대 (Stacked Bar)':
                    if color_col_for_chart: fig = px.bar(processed_df_for_chart, x=x_axis, y=y_plot_val, color=color_col_for_chart, barmode='stack', title=f"{x_axis} 별 {color_col_for_chart} 그룹 {y_plot_val} (누적, {agg_method})")
                    else: melted_df = pd.melt(processed_df_for_chart, id_vars=[x_axis], value_vars=y_plot_val, var_name='범례', value_name='값'); fig = px.bar(melted_df, x=x_axis, y='값', color='범례', barmode='stack', title=f"{x_axis} 별 Y값 누적 ({agg_method})")
                elif chart_type == '선 (Line)': fig = px.line(processed_df_for_chart, x=x_axis, y=y_plot_val, color=color_col_for_chart, markers=True, title=f"{x_axis} 별 Y값 ({agg_method})")
                elif chart_type == '누적 영역 (Stacked Area)':
                    if color_col_for_chart: fig = px.area(processed_df_for_chart, x=x_axis, y=y_plot_val, color=color_col_for_chart, title=f"{x_axis} 별 {color_col_for_chart} 그룹 {y_plot_val} (누적 영역, {agg_method})")
                    else: melted_df = pd.melt(processed_df_for_chart, id_vars=[x_axis], value_vars=y_plot_val, var_name='범례', value_name='값'); fig = px.area(melted_df, x=x_axis, y='값', color='범례', title=f"{x_axis} 별 Y값 누적 영역 ({agg_method})")
        if fig:
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            if df is not None and headers and st.session_state.data_loaded_success:
                 pass
    except Exception as e:
        st.error(f"차트 생성 중 예측하지 못한 오류 발생: {e}")
        import traceback
        st.error(traceback.format_exc())

