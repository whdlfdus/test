import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # For numeric type checking / conversion

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="고급 데이터 시각화 도구 v2.1")

# --- 초기 세션 상태 설정 ---
def init_session_state():
    defaults = {
        'df': None, 'headers': [], 'numeric_headers': [], 'string_headers': [],
        'chart_type': '막대 (Bar)', 'x_axis': None,
        'y_axis_single': None, 
        'y_axis_multiple': [], 
        'y_axis_secondary': "None", 
        'group_by_col': "None", 'agg_method': 'Sum',
        'pie_name_col': None, 'pie_value_col': None,
        'last_uploaded_filename': None, 
        'data_loaded_success': False,
        # 결측치 처리 관련 세션 상태
        'mv_selected_cols': [],
        'mv_method': "결측치가 있는 행 전체 제거",
        'mv_specific_value': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- 도우미 함수 ---
def get_column_types(df):
    if df is None:
        return [], []
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    all_cols_for_cat = list(df.columns)
    return numeric_cols, all_cols_for_cat

def load_data(uploaded_file):
    try:
        df_new = pd.read_csv(uploaded_file) # 버튼 클릭 시 항상 새로 로드
        st.session_state.df = df_new
        st.session_state.headers = list(df_new.columns)
        numeric_cols, string_cols = get_column_types(df_new)
        st.session_state.numeric_headers = numeric_cols
        st.session_state.string_headers = string_cols 

        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.data_loaded_success = True

        # 데이터 로드 시 기본 축 자동 설정 (기존 값 최대한 유지)
        all_cols_for_x_group_pie_name = st.session_state.headers

        if not st.session_state.x_axis or st.session_state.x_axis not in all_cols_for_x_group_pie_name:
            st.session_state.x_axis = all_cols_for_x_group_pie_name[0] if all_cols_for_x_group_pie_name else None

        if st.session_state.numeric_headers:
            if not st.session_state.y_axis_single or st.session_state.y_axis_single not in st.session_state.numeric_headers:
                st.session_state.y_axis_single = st.session_state.numeric_headers[0]
            
            current_y_multi = [val for val in st.session_state.y_axis_multiple if val in st.session_state.numeric_headers]
            if not current_y_multi and st.session_state.numeric_headers:
                current_y_multi = [st.session_state.numeric_headers[0]]
            st.session_state.y_axis_multiple = current_y_multi

            if not st.session_state.pie_value_col or st.session_state.pie_value_col not in st.session_state.numeric_headers:
                st.session_state.pie_value_col = st.session_state.numeric_headers[0]
        else: 
            st.session_state.y_axis_single = None
            st.session_state.y_axis_multiple = []
            st.session_state.pie_value_col = None

        if not st.session_state.pie_name_col or st.session_state.pie_name_col not in all_cols_for_x_group_pie_name:
            st.session_state.pie_name_col = all_cols_for_x_group_pie_name[0] if all_cols_for_x_group_pie_name else None
        
        if st.session_state.group_by_col not in all_cols_for_x_group_pie_name and st.session_state.group_by_col != "None":
            st.session_state.group_by_col = "None"
        
        if st.session_state.y_axis_secondary not in st.session_state.numeric_headers and st.session_state.y_axis_secondary != "None":
             st.session_state.y_axis_secondary = "None"

        st.success("데이터가 성공적으로 로드 및 업데이트되었습니다!")
        return True
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        st.session_state.data_loaded_success = False
        st.session_state.df = None 
        st.session_state.headers = []
        st.session_state.numeric_headers = []
        st.session_state.string_headers = []
        return False

# --- UI 구성 ---
st.title("📊 고급 데이터 시각화 도구 v2.1") # 버전 업데이트
st.markdown("CSV 파일을 업로드하여 다양한 차트를 생성하세요. 데이터 정제, 그룹화, 집계, 누적 차트, 이중 Y축 기능을 지원합니다.")

# --- 사이드바 ---
with st.sidebar:
    st.header("1. 데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일 선택", type="csv", key="file_uploader_v2_1")

    if uploaded_file:
        if st.button("데이터 로드/업데이트", key="load_data_button_v2_1"):
            load_data(uploaded_file)
    
    df = st.session_state.df
    headers = st.session_state.headers
    numeric_headers = st.session_state.numeric_headers
    string_headers = st.session_state.string_headers

    if df is not None and st.session_state.data_loaded_success:
        st.divider()
        st.header("2. 차트 설정")
        # (기존 차트 설정 UI - 변경 없음)
        chart_type_options = ['막대 (Bar)', '누적 막대 (Stacked Bar)', '선 (Line)', '누적 영역 (Stacked Area)', '파이 (Pie)']
        st.session_state.chart_type = st.selectbox(
            "차트 종류", chart_type_options,
            index=chart_type_options.index(st.session_state.chart_type) if st.session_state.chart_type in chart_type_options else 0,
            key="chart_type_select_v2_1"
        )
        chart_type = st.session_state.chart_type 

        is_pie_chart = (chart_type == '파이 (Pie)')
        is_stacked_chart = (chart_type in ['누적 막대 (Stacked Bar)', '누적 영역 (Stacked Area)'])

        if not is_pie_chart:
            st.session_state.x_axis = st.selectbox(
                "X축", string_headers, 
                index=string_headers.index(st.session_state.x_axis) if st.session_state.x_axis in string_headers else 0,
                key="x_axis_select_v2_1"
            )
            
            group_by_options = ["None"] + [h for h in string_headers if h != st.session_state.x_axis]
            st.session_state.group_by_col = st.selectbox(
                "그룹화 기준 열 (선택)", group_by_options,
                index=group_by_options.index(st.session_state.group_by_col) if st.session_state.group_by_col in group_by_options else 0,
                key="group_by_select_v2_1"
            )
            group_by_col = st.session_state.group_by_col 

            st.session_state.agg_method = st.selectbox(
                "집계 방식", ['Sum', 'Mean', 'Median'],
                index=['Sum', 'Mean', 'Median'].index(st.session_state.agg_method),
                key="agg_method_select_v2_1"
            )

            if group_by_col != "None": 
                available_measure_cols = [h for h in numeric_headers if h != st.session_state.x_axis and h != group_by_col]
                if not available_measure_cols:
                    st.warning("그룹화에 사용할 적절한 숫자형 측정값 열이 없습니다.")
                    st.session_state.y_axis_single = None
                else:
                    st.session_state.y_axis_single = st.selectbox(
                        "측정값 (기본 Y축)", available_measure_cols,
                        index=available_measure_cols.index(st.session_state.y_axis_single) if st.session_state.y_axis_single in available_measure_cols else 0,
                        key="y_single_grouped_select_v2_1"
                    )
            else: 
                if chart_type in ['선 (Line)', '누적 영역 (Stacked Area)', '누적 막대 (Stacked Bar)']: 
                    available_y_multi = [h for h in numeric_headers if h != st.session_state.x_axis]
                    if not available_y_multi:
                        st.warning(f"{chart_type}에 사용할 적절한 숫자형 Y축 열이 없습니다.")
                        st.session_state.y_axis_multiple = []
                    else:
                        current_y_multi = [val for val in st.session_state.y_axis_multiple if val in available_y_multi]
                        if not current_y_multi and available_y_multi: current_y_multi = [available_y_multi[0]]
                        st.session_state.y_axis_multiple = st.multiselect(
                            "기본 Y축 (다중 가능)", available_y_multi, default=current_y_multi, key="y_multi_select_v2_1"
                        )
                elif chart_type == '막대 (Bar)': 
                    available_y_single_bar = [h for h in numeric_headers if h != st.session_state.x_axis]
                    if not available_y_single_bar:
                        st.warning("막대 그래프에 사용할 적절한 숫자형 Y축 열이 없습니다.")
                        st.session_state.y_axis_single = None
                    else:
                        st.session_state.y_axis_single = st.selectbox(
                            "기본 Y축", available_y_single_bar,
                            index=available_y_single_bar.index(st.session_state.y_axis_single) if st.session_state.y_axis_single in available_y_single_bar else 0,
                            key="y_single_bar_select_v2_1"
                        )
            
            if not is_stacked_chart:
                primary_y_selection_for_secondary = []
                if group_by_col != "None":
                    if st.session_state.y_axis_single: primary_y_selection_for_secondary = [st.session_state.y_axis_single]
                else: 
                    if chart_type == '선 (Line)': primary_y_selection_for_secondary = st.session_state.y_axis_multiple
                    elif chart_type == '막대 (Bar)' and st.session_state.y_axis_single: primary_y_selection_for_secondary = [st.session_state.y_axis_single]
                
                secondary_y_options = ["None"] + [
                    h for h in numeric_headers if h != st.session_state.x_axis and h != group_by_col and h not in primary_y_selection_for_secondary
                ]
                st.session_state.y_axis_secondary = st.selectbox(
                    "보조 Y축 (선택)", secondary_y_options,
                    index=secondary_y_options.index(st.session_state.y_axis_secondary) if st.session_state.y_axis_secondary in secondary_y_options else 0,
                    key="y_secondary_select_v2_1"
                )
            else: 
                st.session_state.y_axis_secondary = "None"
        
        else: 
            st.session_state.pie_name_col = st.selectbox(
                "레이블(이름) 열", string_headers,
                index=string_headers.index(st.session_state.pie_name_col) if st.session_state.pie_name_col in string_headers else 0,
                key="pie_name_select_v2_1"
            )
            
            available_pie_values = [h for h in numeric_headers if h != st.session_state.pie_name_col]
            if not available_pie_values:
                st.warning("파이 차트에 사용할 적절한 숫자형 값 열이 없습니다.")
                st.session_state.pie_value_col = None
            else:
                 st.session_state.pie_value_col = st.selectbox(
                    "값 열", available_pie_values,
                    index=available_pie_values.index(st.session_state.pie_value_col) if st.session_state.pie_value_col in available_pie_values else 0,
                    key="pie_value_select_v2_1"
                )
            st.session_state.group_by_col = "None"
            st.session_state.y_axis_secondary = "None"

        st.divider() # 다음 섹션 구분
        st.header("3. 데이터 정제")
        st.subheader("결측치 처리")

        st.session_state.mv_selected_cols = st.multiselect(
            "대상 열 선택 (결측치 처리)",
            options=headers, 
            default=st.session_state.mv_selected_cols if all(item in headers for item in st.session_state.mv_selected_cols) else [],
            key="mv_target_cols_v2_1"
        )

        mv_method_options = [
            "결측치가 있는 행 전체 제거", # Remove rows with any NA in selected columns
            "평균값으로 대체 (숫자형 전용)",
            "중앙값으로 대체 (숫자형 전용)",
            "최빈값으로 대체",
            "특정 값으로 채우기"
        ]
        st.session_state.mv_method = st.selectbox(
            "처리 방법 선택",
            options=mv_method_options,
            index=mv_method_options.index(st.session_state.mv_method),
            key="mv_method_v2_1"
        )

        if st.session_state.mv_method == "특정 값으로 채우기":
            st.session_state.mv_specific_value = st.text_input(
                "채울 특정 값 입력", 
                value=st.session_state.mv_specific_value, 
                key="mv_specific_val_v2_1"
            )

        if st.button("결측치 처리 적용", key="apply_mv_button_v2_1"):
            if not st.session_state.mv_selected_cols:
                st.warning("결측치를 처리할 대상 열을 하나 이상 선택해주세요.")
            else:
                df_processed = df.copy()
                processed_cols_count = 0
                error_cols = []

                for col in st.session_state.mv_selected_cols:
                    if df_processed[col].isnull().sum() == 0: # 선택한 열에 결측치가 없으면 건너뜀
                        st.info(f"'{col}' 열에는 결측치가 없습니다.")
                        continue

                    try:
                        if st.session_state.mv_method == "결측치가 있는 행 전체 제거":
                            # 이 옵션은 모든 선택된 열에 대해 한 번에 적용해야 함
                            # 버튼 로직 상단에서 한 번만 처리하도록 변경 필요
                            pass # 아래에서 일괄 처리
                        elif st.session_state.mv_method == "평균값으로 대체 (숫자형 전용)":
                            if col in numeric_headers:
                                mean_val = df_processed[col].mean()
                                df_processed[col].fillna(mean_val, inplace=True)
                                processed_cols_count +=1
                            else: error_cols.append(f"{col} (숫자형 아님)"); continue
                        elif st.session_state.mv_method == "중앙값으로 대체 (숫자형 전용)":
                            if col in numeric_headers:
                                median_val = df_processed[col].median()
                                df_processed[col].fillna(median_val, inplace=True)
                                processed_cols_count +=1
                            else: error_cols.append(f"{col} (숫자형 아님)"); continue
                        elif st.session_state.mv_method == "최빈값으로 대체":
                            mode_val = df_processed[col].mode()
                            if not mode_val.empty:
                                df_processed[col].fillna(mode_val[0], inplace=True)
                                processed_cols_count +=1
                            else: error_cols.append(f"{col} (최빈값 계산 불가)"); continue # 모든 값이 NaN인 경우 등
                        elif st.session_state.mv_method == "특정 값으로 채우기":
                            fill_value_str = st.session_state.mv_specific_value
                            # 데이터 타입에 맞게 변환 시도
                            try:
                                if col in numeric_headers: # 숫자형 열이면 숫자로 변환 시도
                                    fill_value = float(fill_value_str)
                                else: # 그 외에는 문자열 그대로 사용 (또는 다른 타입 변환 로직 추가 가능)
                                    fill_value = fill_value_str
                                df_processed[col].fillna(fill_value, inplace=True)
                                processed_cols_count +=1
                            except ValueError:
                                error_cols.append(f"{col} ('{fill_value_str}' 값 변환 실패)"); continue
                    except Exception as e_col:
                        error_cols.append(f"{col} (처리 중 오류: {str(e_col)[:30]}...)")
                        continue
                
                # "결측치가 있는 행 전체 제거" 일괄 처리
                if st.session_state.mv_method == "결측치가 있는 행 전체 제거":
                    original_rows = len(df_processed)
                    df_processed.dropna(subset=st.session_state.mv_selected_cols, inplace=True)
                    rows_removed = original_rows - len(df_processed)
                    if rows_removed > 0 : processed_cols_count = len(st.session_state.mv_selected_cols) # 처리된 것으로 간주
                    st.info(f"선택된 열 기준 결측치 포함 행 {rows_removed}개 제거 완료.")


                if processed_cols_count > 0 :
                    st.session_state.df = df_processed
                    # 컬럼 타입이 변경되었을 수 있으므로 헤더 정보 업데이트
                    st.session_state.numeric_headers, st.session_state.string_headers = get_column_types(df_processed)
                    st.success(f"{processed_cols_count}개 선택 열에 대한 결측치 처리가 완료되었습니다. 데이터 미리보기가 업데이트됩니다.")
                    # 선택 초기화
                    st.session_state.mv_selected_cols = [] 
                    # st.experimental_rerun() # UI 즉시 업데이트를 위해 (필요에 따라)
                
                if error_cols:
                    st.error(f"다음 열 처리 중 문제 발생: {', '.join(error_cols)}")
                elif processed_cols_count == 0 and not (st.session_state.mv_method == "결측치가 있는 행 전체 제거"): # 실제로 처리된 열이 없는 경우 (이미 결측치가 없었거나, 오류로 처리 못한 경우 제외)
                    st.info("선택된 열에 대해 적용할 결측치가 없거나, 조건에 맞지 않아 변경사항이 없습니다.")


    elif uploaded_file and not st.session_state.data_loaded_success:
        st.sidebar.warning("데이터 로드에 실패했습니다. 파일을 확인하고 다시 시도해주세요.")
    else:
        st.sidebar.info("CSV 파일을 업로드하고 '데이터 로드/업데이트' 버튼을 누르면 추가 옵션이 나타납니다.")

# --- 메인 화면 ---
if df is None:
    st.info("데이터를 시각화하려면 왼쪽 사이드바에서 CSV 파일을 업로드하고 '데이터 로드/업데이트' 버튼을 눌러주세요.")
elif not st.session_state.data_loaded_success: 
    st.warning("데이터 로드에 실패했습니다. 파일을 확인하고 사이드바에서 다시 시도해주세요.")
else:
    st.subheader("데이터 미리보기 (상위 5행)")
    st.dataframe(df.head())

    st.subheader("시각화 결과")
    chart_placeholder = st.empty()

    try:
        fig = None
        # (기존 차트 생성 로직 - 변경 없음. 단, st.session_state.df가 변경되었으므로 자동으로 업데이트된 데이터를 사용)
        chart_type = st.session_state.chart_type
        x_axis = st.session_state.x_axis
        y_single = st.session_state.y_axis_single
        y_multiple = st.session_state.y_axis_multiple
        y_secondary = st.session_state.y_axis_secondary
        group_by_col = st.session_state.group_by_col
        agg_method = st.session_state.agg_method
        pie_name_col = st.session_state.pie_name_col
        pie_value_col = st.session_state.pie_value_col
        
        # 필수 선택 항목 검사
        if chart_type != '파이 (Pie)' and not x_axis:
            chart_placeholder.warning("X축을 선택해주세요."); st.stop()
        if chart_type == '파이 (Pie)' and (not pie_name_col or not pie_value_col):
            chart_placeholder.warning("파이 차트의 레이블 열과 값 열을 모두 선택해주세요."); st.stop()
        if chart_type == '파이 (Pie)' and pie_value_col not in st.session_state.numeric_headers: # numeric_headers 세션 사용
            chart_placeholder.error(f"파이 차트의 값 열 ('{pie_value_col}')은 숫자형이어야 합니다."); st.stop()

        processed_df_for_chart = df.copy() # 차트용 데이터 복사
        agg_functions = {'Sum': 'sum', 'Mean': 'mean', 'Median': 'median'}
        current_agg_func = agg_functions[agg_method]

        y_val_for_chart = None 
        y_multi_for_chart = [] 
        color_col_for_chart = None
        barmode_for_chart = 'group' 

        if chart_type == '파이 (Pie)':
            if pie_value_col not in st.session_state.numeric_headers: 
                chart_placeholder.error(f"파이 차트의 값 열 ('{pie_value_col}')은 숫자형이어야 합니다."); st.stop()
            pie_data_agg = processed_df_for_chart.groupby(pie_name_col, as_index=False).agg({pie_value_col: current_agg_func})
            fig = px.pie(pie_data_agg, names=pie_name_col, values=pie_value_col, title=f"{pie_name_col} 별 {pie_value_col} 분포 ({agg_method} 기준)")
        
        else:
            y_cols_to_aggregate = []
            if group_by_col != "None":
                if not y_single or y_single not in st.session_state.numeric_headers:
                    chart_placeholder.warning(f"그룹화에 사용할 측정값(기본 Y축)을 선택해주세요 (숫자형). 현재 선택: {y_single}"); st.stop()
                y_cols_to_aggregate = [y_single]
                grouping_cols = [x_axis, group_by_col]
                color_col_for_chart = group_by_col
                y_val_for_chart = y_single # 집계 후에도 컬럼명 유지됨
            else: # 그룹화 미사용
                grouping_cols = [x_axis]
                if chart_type in ['막대 (Bar)']:
                    if not y_single or y_single not in st.session_state.numeric_headers:
                        chart_placeholder.warning(f"막대 차트의 Y축을 선택해주세요 (숫자형). 현재 선택: {y_single}"); st.stop()
                    y_cols_to_aggregate = [y_single]
                    y_val_for_chart = y_single
                elif chart_type in ['선 (Line)', '누적 막대 (Stacked Bar)', '누적 영역 (Stacked Area)']:
                    if not y_multiple or not all(y in st.session_state.numeric_headers for y in y_multiple):
                        chart_placeholder.warning(f"{chart_type}의 Y축을 하나 이상 선택해주세요 (숫자형). 현재 선택: {y_multiple}"); st.stop()
                    y_cols_to_aggregate = y_multiple
                    y_multi_for_chart = y_multiple
            
            # 실제 집계 수행 (y_cols_to_aggregate가 비어있지 않을 때만)
            if y_cols_to_aggregate:
                agg_dict = {y_col: current_agg_func for y_col in y_cols_to_aggregate}
                processed_df_for_chart = processed_df_for_chart.groupby(grouping_cols, as_index=False).agg(agg_dict)


            # 이중 Y축 처리 (누적 차트에서는 미적용)
            is_stacked_chart = (chart_type in ['누적 막대 (Stacked Bar)', '누적 영역 (Stacked Area)'])
            if y_secondary != "None" and not is_stacked_chart:
                if y_secondary not in st.session_state.numeric_headers:
                    chart_placeholder.error(f"보조 Y축 열 ('{y_secondary}')은 숫자형이어야 합니다."); st.stop()
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                plot_primary_y_data = y_val_for_chart if y_val_for_chart else y_multi_for_chart

                if group_by_col != "None" and y_val_for_chart: # 그룹화된 단일 측정값
                    unique_groups = processed_df_for_chart[group_by_col].unique()
                    for i, group_val in enumerate(unique_groups):
                        trace_data = processed_df_for_chart[processed_df_for_chart[group_by_col] == group_val]
                        if chart_type == '막대 (Bar)':
                            fig.add_trace(go.Bar(x=trace_data[x_axis], y=trace_data[y_val_for_chart], name=f"{group_val} ({y_val_for_chart})", marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]), secondary_y=False)
                        elif chart_type == '선 (Line)':
                            fig.add_trace(go.Scatter(x=trace_data[x_axis], y=trace_data[y_val_for_chart], mode='lines+markers', name=f"{group_val} ({y_val_for_chart})", line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)])), secondary_y=False)
                else: # 그룹화 안 됨 (단일 또는 다중 기본 Y)
                    if chart_type == '막대 (Bar)' and y_val_for_chart:
                         fig.add_trace(go.Bar(x=processed_df_for_chart[x_axis], y=processed_df_for_chart[y_val_for_chart], name=y_val_for_chart), secondary_y=False)
                    elif chart_type == '선 (Line)' and y_multi_for_chart:
                        for i, y_col in enumerate(y_multi_for_chart):
                            fig.add_trace(go.Scatter(x=processed_df_for_chart[x_axis], y=processed_df_for_chart[y_col], mode='lines+markers', name=y_col, line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)])), secondary_y=False)

                # 보조 Y축 (원본 df에서 X축 기준으로 평균 집계)
                if x_axis in df.columns and y_secondary in df.columns:
                    # 원본 df에서 보조 Y축 데이터 가져오기. processed_df_for_chart의 x_axis에 맞춰야 함.
                    # processed_df_for_chart는 이미 x_axis로 집계되었으므로, 이 x_axis 값들을 사용.
                    secondary_base_data = df.groupby(x_axis, as_index=False)[y_secondary].mean() # 예: 평균
                    
                    # processed_df_for_chart의 x축 순서와 값을 기준으로 보조 y축 데이터를 매핑
                    # (processed_df_for_chart가 여러 그룹으로 나뉘어져 복제된 x값을 가질 수 있으므로 unique 처리)
                    unique_x_in_primary = processed_df_for_chart[x_axis].drop_duplicates().sort_values()
                    temp_secondary_df = pd.merge(pd.DataFrame({x_axis: unique_x_in_primary}), secondary_base_data, on=x_axis, how='left')

                    fig.add_trace(go.Scatter(x=temp_secondary_df[x_axis], y=temp_secondary_df[y_secondary], mode='lines+markers', name=f"{y_secondary} (보조)", yaxis='y2', line=dict(dash='dot')), secondary_y=True)
                
                title_y_primary_text = y_val_for_chart if y_val_for_chart else ', '.join(y_multi_for_chart)
                fig.update_layout(title_text=f"{x_axis} 별 {title_y_primary_text} 및 {y_secondary} (보조)")
                fig.update_yaxes(title_text=f"기본 Y ({agg_method})", secondary_y=False)
                fig.update_yaxes(title_text=f"{y_secondary} (보조, 평균)", secondary_y=True)
                if chart_type == '막대 (Bar)' and group_by_col != "None": fig.update_layout(barmode='group')

            # 단일 Y축 또는 누적 차트 (이중 Y축 미적용)
            else:
                y_plot_val = y_val_for_chart if y_val_for_chart else y_multi_for_chart # px 함수에 전달될 Y값
                
                if chart_type == '막대 (Bar)':
                    fig = px.bar(processed_df_for_chart, x=x_axis, y=y_plot_val, color=color_col_for_chart, barmode='group' if color_col_for_chart else 'relative', title=f"{x_axis} 별 Y값 ({agg_method})")
                elif chart_type == '누적 막대 (Stacked Bar)':
                    if color_col_for_chart: # 그룹별 누적
                         fig = px.bar(processed_df_for_chart, x=x_axis, y=y_plot_val, color=color_col_for_chart, barmode='stack', title=f"{x_axis} 별 {color_col_for_chart} 그룹 {y_plot_val} (누적, {agg_method})")
                    else: # 다중 Y 컬럼 누적 (y_multi_for_chart 사용)
                        melted_df = pd.melt(processed_df_for_chart, id_vars=[x_axis], value_vars=y_multi_for_chart, var_name='범례', value_name='값')
                        fig = px.bar(melted_df, x=x_axis, y='값', color='범례', barmode='stack', title=f"{x_axis} 별 Y값 누적 ({agg_method})")
                elif chart_type == '선 (Line)':
                    # 그룹화 안되고 다중 Y축일 경우, y_plot_val은 리스트임
                    fig = px.line(processed_df_for_chart, x=x_axis, y=y_plot_val, color=color_col_for_chart, markers=True, title=f"{x_axis} 별 Y값 ({agg_method})")
                elif chart_type == '누적 영역 (Stacked Area)':
                    if color_col_for_chart: # 그룹별 누적 영역
                        fig = px.area(processed_df_for_chart, x=x_axis, y=y_plot_val, color=color_col_for_chart, title=f"{x_axis} 별 {color_col_for_chart} 그룹 {y_plot_val} (누적 영역, {agg_method})")
                    else: # 다중 Y 컬럼 누적 영역 (y_multi_for_chart 사용)
                        melted_df = pd.melt(processed_df_for_chart, id_vars=[x_axis], value_vars=y_multi_for_chart, var_name='범례', value_name='값')
                        fig = px.area(melted_df, x=x_axis, y='값', color='범례', title=f"{x_axis} 별 Y값 누적 영역 ({agg_method})")
        
        if fig:
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        else:
            if df is not None and headers and st.session_state.data_loaded_success:
                 if not (chart_type == '파이 (Pie)' and (not pie_name_col or not pie_value_col)) and \
                    not (chart_type != '파이 (Pie)' and not x_axis) :
                    # 이 조건은 너무 복잡하므로, 필수 값들이 설정되지 않았을 때 더 구체적인 메시지 필요
                    pass # 이미 각 조건별로 warning/error가 표시됨

    except Exception as e:
        st.error(f"차트 생성 중 예측하지 못한 오류 발생: {e}")
        import traceback
        st.error(traceback.format_exc())
