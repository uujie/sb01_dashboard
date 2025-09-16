import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -------------------------------
# 1. 載入資料
# -------------------------------
layout_df = pd.read_excel("./SB01 layout_20241028V8.xlsx", sheet_name=None)

# 設定各樓層對應的檔案
floor_options = {
    "1F": {
        "tt": "./SB01-RAW_OK/SB01_1F_TT.csv",
        "dcc": "./SB01-RAW_OK/SB01-1F-DCC0501-0601.csv",
        "layout_sheet": "1F",
        "pv_replace": None
    },
    "2F": {
        "tt": "./SB01-RAW_OK/SB01_2F_TT.csv",
        "dcc": "./SB01-RAW_OK/SB01-2F-DCC0501-0601.csv",
        "layout_sheet": "2F",
        "pv_replace": ("_TTHT_TT", "_TTHT_HT")
    },
    "3F": {
        "tt": "./SB01-RAW_OK/SB01_3F_TT.csv",
        "dcc": "./SB01-RAW_OK/SB01-3F-DCC0501-0601.csv",
        "layout_sheet": "3F",
        "pv_replace": ("_TTHT_TT", "_TTHT_HT")
    },
    "4F": {
        "tt": "./SB01-RAW_OK/SB01_4F_TT.csv",
        "dcc": "./SB01-RAW_OK/SB01-4F-DCC0501-0601.csv",
        "layout_sheet": "4F",
        "pv_replace": None
    }
}

# 選擇樓層
selected_floor = st.sidebar.selectbox("請選擇樓層", list(floor_options.keys()))
config = floor_options[selected_floor]

# 讀取資料
air_df = pd.read_csv(config["tt"], low_memory=False)
dcc_df = pd.read_csv(config["dcc"])
layout_floor = layout_df[config["layout_sheet"]]
layout_floor.columns = layout_floor.columns.astype(str)
layout_floor = layout_floor.fillna(method='ffill')  # 合併儲存格補值

# -------------------------------
# 2. 多選看板
# -------------------------------
panels = layout_floor['溫溼度看板'].dropna().unique().tolist()
selected_panels = st.multiselect("請選擇溫溼度看板 (TTHTXXX)", panels)
if not selected_panels:
    st.warning("請至少選擇一個看板！")
    st.stop()

# -------------------------------
# 3. 抓 DCC 與 PV 欄位
# -------------------------------
# 決定 DCC 欄位（繁簡體擇一）
if 'DCC名稱' in layout_floor.columns:
    dcc_col = 'DCC名稱'
elif 'DCC名称' in layout_floor.columns:
    dcc_col = 'DCC名称'
else:
    st.error("缺少欄位: DCC名稱/DCC名称")
    st.stop()
related_dccs = layout_floor[layout_floor['溫溼度看板'].isin(selected_panels)][dcc_col].dropna().unique().tolist()

# 決定看板點位欄位（繁簡體擇一）
if '看板點位' in layout_floor.columns:
    pv_base_col = '看板點位'
elif '看板点位' in layout_floor.columns:
    pv_base_col = '看板点位'
else:
    st.error("缺少欄位: 看板點位/看板点位")
    st.stop()
pv_base_list = layout_floor[layout_floor['溫溼度看板'].isin(selected_panels)][pv_base_col].dropna().tolist()

# 蒐集 PV 欄位
pv_cols = []
for base in pv_base_list:
    base = str(base).strip()
    if config['pv_replace']:
        base = base.replace(*config['pv_replace'])
    norm = base.upper()
    candidates = [c for c in air_df.columns if c.upper().endswith('.PV') and norm in c.upper()]
    if candidates:
        for c in candidates:
            if c not in pv_cols:
                pv_cols.append(c)
    else:
        st.warning(f"⚠️ 缺少 PV 欄位: {base}.PV")
if not pv_cols:
    st.error("找不到任何 PV 欄位，請檢查檔案與命名")
    st.stop()

# -------------------------------
# 4. 合併資料
# -------------------------------
air_df['DateTime'] = pd.to_datetime(air_df['DateTime'])
dcc_df['DateTime'] = pd.to_datetime(dcc_df['DateTime'])
merged = pd.merge(dcc_df, air_df[['DateTime'] + pv_cols], on='DateTime', how='inner')

# -------------------------------
# 5. 區塊平均函式
# -------------------------------
def block_average(arr, block_size=15):
    arr = pd.to_numeric(arr, errors='coerce').to_numpy(dtype=float)
    n = arr.size
    m = int(np.ceil(n / block_size)) * block_size
    padded = np.full(m, np.nan)
    padded[:n] = arr
    reshaped = padded.reshape(-1, block_size)
    return np.nanmean(reshaped, axis=1)

# 計算平均
block_size = 15
avg_dcc = {}
for d in related_dccs:
    for suf in ['_CV5', '_TT4']:
        col = d + suf
        if col in merged.columns:
            avg_dcc[col] = block_average(merged[col], block_size)
avg_pv = {c: block_average(merged[c], block_size) for c in pv_cols}

# 統一長度
def pad_to_max(data_dict, max_len):
    return {k: np.concatenate([v, np.full(max_len - len(v), np.nan)]) for k, v in data_dict.items()}
max_len = max(len(v) for v in list(avg_dcc.values()) + list(avg_pv.values()))
avg_dcc = pad_to_max(avg_dcc, max_len)
avg_pv = pad_to_max(avg_pv, max_len)
x_vals = np.arange(max_len)

# -------------------------------
# 6. 繪製圖表
# -------------------------------
fig = make_subplots(specs=[[{"secondary_y": True}]])
# DCC
for col, data in avg_dcc.items():
    sec = col.endswith('_TT4')
    fig.add_trace(go.Scatter(x=x_vals, y=data, name=col), secondary_y=sec)
# 看板溫度
for col, data in avg_pv.items():
    fig.add_trace(go.Scatter(x=x_vals, y=data, mode='lines', name=col[:-3], line=dict(width=3, dash='dot')), secondary_y=True)

fig.update_layout(
    title=f"{selected_floor} DCC & 看板溫度",
    xaxis_title='區塊 (每15筆)',
    xaxis=dict(rangeslider=dict(visible=True)),
    legend_title='圖例'
)
fig.update_yaxes(title_text='風門開度 (%)', secondary_y=False)
fig.update_yaxes(title_text='溫度 (°C)', secondary_y=True)

st.plotly_chart(fig, use_container_width=True)