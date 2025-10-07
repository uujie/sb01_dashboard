import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 自動掃描 QA02 RAW 中所有 DCC 檔案（CSV）===
def scan_qa02_dcc(base_path="CHD/QA02_NOW/QA02 RAW"):
    dcc_map = {"1F": [], "2F": [], "3F": [], "4F": []}  # 沒有4F
    if not os.path.exists(base_path):
        return dcc_map
    for file in os.listdir(base_path):
        if file.endswith(".csv") and "DCC" in file.upper():
            f_upper = file.upper()
            for floor in dcc_map.keys():
                if f"-{floor}-" in f_upper:
                    dcc_map[floor].append(os.path.join(base_path, file))
    return dcc_map

# === 通用掃描（支援 .xlsx/.csv，並自動偵測 1F/2F/3F/4F）→ 給 QA08 用 ===
def scan_dcc_any(base_path, floors_hint=("1F","2F","3F","4F"), exts=(".xlsx", ".csv")):
    dcc_map = {f: [] for f in floors_hint}
    if not os.path.exists(base_path):
        return {}
    for file in os.listdir(base_path):
        if not any(file.lower().endswith(ext) for ext in exts):
            continue
        if "DCC" not in file.upper():
            continue
        f_upper = file.upper()
        for floor in floors_hint:
            if f"-{floor}-" in f_upper or f"_{floor}_" in f_upper:
                dcc_map[floor].append(os.path.join(base_path, file))
    # 只保留有檔案的樓層
    return {k: v for k, v in dcc_map.items() if v}

# === 小工具：讀表（自動判斷 CSV / Excel）===
def read_table(path, **kwargs):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    else:
        return pd.read_excel(path, sheet_name=0, **kwargs)

qa02_raw_path = "CHD/QA02_NOW/QA02 RAW"
qa02_layout_path = "CHD/QA02_NOW/QA02-DCC-Layout_20240316V3.xlsx"
qa02_dcc_map = scan_qa02_dcc()

qa08_raw_path = "CHD/QA08_NOW/RAW"
qa08_layout_path = "CHD/QA08_NOW/QA08-DCC-Layout_20241025.xlsx"
qa08_dcc_map = scan_qa02_dcc(base_path="CHD/QA08_NOW/RAW")

ql01_raw_path = "CHD/QL01_NOW/RAW"
ql01_layout_path = "CHD/QL01_NOW/QL01-DCC-Layout_20240920.xlsx"
ql01_dcc_map = scan_qa02_dcc(base_path="CHD/QL01_NOW/RAW")

qa06_raw_path = "CHD/QA06_NOW/RAW"
qa06_layout_path = "CHD/QA06_NOW/QA06-DCC-Layout_20241028.xlsx"
qa06_dcc_map = scan_qa02_dcc(base_path="CHD/QA06_NOW/RAW")

area_config = {
    "SJ": {
        "factories": {
            "SB01": {
                "layout_file": "./SJ/SB01_NOW/SB01 layout_20250421.xlsx",
                "comparison_file": "./SJ/SB01_NOW/SB01_ComparisonTable.xlsx",  # Max correlation 對應表
                "floors": {
                    "1F": {"tt": "./SJ/SB01_NOW/SB01-RAW_OK/SB01_1F_TT.csv", "dcc": "./SJ/SB01_NOW/SB01-RAW_OK/SB01-1F-DCC0501-0601.csv", "pv_replace": None},
                    "2F": {"tt": "./SJ/SB01_NOW/SB01-RAW_OK/SB01_2F_TT.csv", "dcc": "./SJ/SB01_NOW/SB01-RAW_OK/SB01-2F-DCC0501-0601.csv", "pv_replace": ("_TTHT_TT", "_TTHT_HT")},
                    "3F": {"tt": "./SJ/SB01_NOW/SB01-RAW_OK/SB01_3F_TT.csv", "dcc": "./SJ/SB01_NOW/SB01-RAW_OK/SB01-3F-DCC0501-0601.csv", "pv_replace": ("_TTHT_TT", "_TTHT_HT")},
                    "4F": {"tt": "./SJ/SB01_NOW/SB01-RAW_OK/SB01_4F_TT.csv", "dcc": "./SJ/SB01_NOW/SB01-RAW_OK/SB01-4F-DCC0501-0601.csv", "pv_replace": None},
                },
            },
            "SB02": {
                "layout_file": "./SJ/SB02_NOW/SB02-DCC-Layout_20250213.xlsx",
                "comparison_file": "./SJ/SB02_NOW/SB02_ComparisonTable.xlsx",
                "floors": {
                    "1F": {"tt": "./SJ/SB02_NOW/RAW/SB02_1F_TT_202412.csv", "dcc": "./SJ/SB02_NOW/RAW/SB02-1F-DCC1201-1231.csv", "pv_replace": None},
                    "2F": {"tt": "./SJ/SB02_NOW/RAW/SB02_2F_TT_202412.csv", "dcc": "./SJ/SB02_NOW/RAW/SB02-2F-DCC1201-1231.csv", "pv_replace": ("_TTHT_TT", "_TTHT_HT")},
                },
            },
            "SL01": { 
                "layout_file": "SJ/SL01_NOW/SL01 layout - 20250421.xlsx", 
                "floors": { 
                    floor: { "tt": os.path.join("SJ/SL01_NOW/SL-RAW_20240418", 
                    f"SL_{floor}_TT_202305.csv"), 
                    "dcc_multi": dcc_list, # 關鍵：2F 以上把 TT 的 base 名稱強制改成 *_TTHTPT_HT01.PV 
                    "pv_replace": None if floor in ["1F"] else ("_TTHTPT_TT", "_TTHTPT_HT")
                    } for floor, dcc_list in scan_dcc_any("SJ/SL01_NOW/SL-RAW_20240418", ("1F","2F","3F","4F","5F","6F"), (".csv",)).items() 
                } 
            },
        },
    },
    "CHD": {
        "factories": {
            "QA02": {
                "layout_file": qa02_layout_path,
                "comparison_file": "./CHD/QA02_NOW/A02_ComparisonTable.xlsx",  # Max correlation 對應表
                "floors": {
                    floor: {
                        "tt": f"CHD/QA02_NOW/QA02 RAW/A02_{floor}_TT.csv",
                        "dcc_multi": dcc_list,
                        "pv_replace": None if floor in ["1F"] else ("_TTHT_TT", "_TTHT_HT"),
                    }
                    for floor, dcc_list in qa02_dcc_map.items()
                },
            },
            "QA06": {
                "layout_file": qa06_layout_path,
                "floors": {
                    floor: {
                        "tt": os.path.join(qa06_raw_path, f"QA06_{floor}_TT.csv"),
                        "dcc_multi": dcc_list,
                        "pv_replace": None if floor in ["1F"] else ("_TTHT_TT", "_TTHT_HT"),
                    }
                    for floor, dcc_list in qa06_dcc_map.items()
                },
            },
            "QL01": {
                "layout_file": ql01_layout_path,
                "floors": {
                    floor: {
                        "tt": os.path.join(ql01_raw_path, f"QL01_{floor}_TT.csv"),
                        "dcc_multi": dcc_list,
                        "pv_replace": ("_ROOM_TT", "_ROOM_HT"),
                    }
                    for floor, dcc_list in ql01_dcc_map.items()
                },
            },
            "QA08": {
                "layout_file": qa08_layout_path,
                "floors": {
                    floor: {
                        "tt": os.path.join(qa08_raw_path, f"QA08_{floor}_TT.csv"),
                        "dcc_multi": dcc_list,
                        "pv_replace": None if floor in ["1F"] else ("_TTHT_TT", "_TTHT_HT"),
                    }
                    for floor, dcc_list in qa08_dcc_map.items()
                },
            },
        },
    },
}

# === Streamlit UI ===
st.set_page_config(page_title="DCC/TT 每分鐘真實曲線", layout="wide")
selected_area = st.sidebar.selectbox("選擇廠區", list(area_config.keys()))
selected_factory = st.sidebar.selectbox("選擇工廠", list(area_config[selected_area]["factories"].keys()))
factory_cfg = area_config[selected_area]["factories"][selected_factory]

def sort_floors(floors):
    floor_order = {'1F': 1, '2F': 2, '3F': 3, '4F': 4}
    return sorted(floors, key=lambda f: floor_order.get(f, 999))

sorted_floors = sort_floors(list(factory_cfg["floors"].keys()))
selected_floor = st.sidebar.selectbox("選擇樓層", sorted_floors)
floor_cfg = factory_cfg["floors"][selected_floor]

# === Layout ===
layout_df_all = pd.read_excel(factory_cfg["layout_file"], sheet_name=None)
if selected_floor not in layout_df_all:
    st.error(f"Layout 中找不到樓層 {selected_floor}")
    st.stop()
layout_df = layout_df_all[selected_floor].copy()
layout_df.columns = layout_df.columns.astype(str).str.strip()
layout_df.fillna(method="ffill", inplace=True)

# === 讀 TT ===
air_df = read_table(floor_cfg["tt"])
air_df.columns = air_df.columns.str.strip()
tt_time_col = next((col for col in air_df.columns if col.strip().lower() in ["datetime", "startdatetime"]), None)
if not tt_time_col:
    st.error("❌ TT 檔案中找不到 'DateTime' 或 'StartDateTime' 欄位")
    st.write("目前欄位：", air_df.columns.tolist())
    st.stop()
air_df["DateTime"] = pd.to_datetime(air_df[tt_time_col])

# === 讀 DCC（支援 dcc_multi or dcc）===
dcc_df = None
if "dcc_multi" in floor_cfg:
    dcc_parts = []
    for path in floor_cfg["dcc_multi"]:
        try:
            df = read_table(path)
            df.columns = df.columns.str.strip()
            dcc_time_col = next((col for col in df.columns if col.strip().lower() in ["datetime", "startdatetime"]), None)
            if not dcc_time_col:
                st.warning(f"❌ 找不到時間欄位於檔案：{path}")
                continue
            df["DateTime"] = pd.to_datetime(df[dcc_time_col])
            dcc_parts.append(df)
        except Exception as e:
            st.warning(f"DCC 檔讀取錯誤：{path} - {e}")
    if not dcc_parts:
        st.error("無可用 DCC 檔")
        st.stop()
    dcc_df = pd.concat(dcc_parts, ignore_index=True)
else:
    dcc_df = read_table(floor_cfg["dcc"])
    if "DateTime" not in dcc_df.columns:
        dt_col = next((c for c in dcc_df.columns if str(c).lower() == "datetime"), None)
        if dt_col:
            dcc_df = dcc_df.rename(columns={dt_col: "DateTime"})
    dcc_df["DateTime"] = pd.to_datetime(dcc_df["DateTime"])

# === 看板選擇 ===
if '溫溼度看板' not in layout_df.columns:
    st.error("缺少『溫溼度看板』欄位")
    st.stop()
panels = layout_df['溫溼度看板'].dropna().unique().tolist()
selected_panels = st.multiselect("請選擇看板", panels)
if not selected_panels:
    st.warning("請至少選擇一個看板")
    st.stop()

# === PV / DCC 欄位 ===
dcc_col = next((c for c in ['DCC名稱', 'DCC名称'] if c in layout_df.columns), None)
pv_col = next((c for c in ['看板點位', '看板点位'] if c in layout_df.columns), None)
if not dcc_col or not pv_col:
    st.error("Layout 缺少 DCC 或 PV 欄位")
    st.stop()

related_dccs = layout_df[layout_df['溫溼度看板'].isin(selected_panels)][dcc_col].dropna().unique().tolist()
pv_bases = layout_df[layout_df['溫溼度看板'].isin(selected_panels)][pv_col].dropna().tolist()

# === PV 欄位對應（QA02/QA08 用精準對應，其它維持原本）===
pv_cols = []
if selected_area == "CHD" and selected_factory in ("QA02", "QA08"):
    for base in pv_bases:
        pv_name = f"{base}.PV"
        if pv_name in air_df.columns:
            pv_cols.append(pv_name)
else:
    for base in pv_bases:
        base = str(base).strip()
        if floor_cfg.get("pv_replace"):
            base = base.replace(*floor_cfg["pv_replace"])
        matches = [c for c in air_df.columns if c.upper().endswith('.PV') and base.upper().replace("-", "_") in c.upper().replace("-", "_")]
        pv_cols.extend([m for m in matches if m not in pv_cols])
pv_cols = list(dict.fromkeys(pv_cols))  # 去重、保序

if not pv_cols:
    st.error("找不到對應 PV 欄位")
    st.write("TT檔案所有欄位：", air_df.columns.tolist())
    st.write("Layout指定的點位：", pv_bases)
    st.stop()

# === 合併資料（不重取樣）===
merged = pd.merge(dcc_df, air_df[["DateTime"] + pv_cols], on="DateTime", how="outer").sort_values("DateTime")

# === 從 ComparisonTable 讀取：Max correlation 包含目前看板的列 → 取 DCC_name，對應欄位並加入作圖 ===
extra_cols = []  # (MC) 會加在圖例前綴

cmp_path = factory_cfg.get("comparison_file")
if cmp_path and os.path.exists(cmp_path):
    try:
        cmp_book = pd.read_excel(cmp_path, sheet_name=None)

        # 分頁：先精確匹配樓層，否則退回第一張
        if selected_floor in cmp_book:
            cmp_df = cmp_book[selected_floor].copy()
            sheet_used = selected_floor
        else:
            sheet_used = next(iter(cmp_book.keys()))
            cmp_df = cmp_book[sheet_used].copy()

        cmp_df.columns = cmp_df.columns.astype(str).str.strip()

        # 欄位名稱容錯（擴充）
        dcc_col_cmp = next((c for c in cmp_df.columns if c.strip().lower() in [
            "dcc name","dcc_name","dcc名稱","dcc名称","dccname","dcc","dcc點位","dcc欄位"
        ]), None)
        panel_col_cmp = next((c for c in cmp_df.columns if c.strip().lower() in [
            "max correlation","max_correlation","maxcorr","mc","mc targets","targets",
            "溫溼度看板","看板","看板群","panel","panels","對應看板","看板清單","看板列表"
        ]), None)

        # 側欄偵錯
        st.sidebar.write("【CMP】使用工作表:", sheet_used)
        st.sidebar.write("【CMP】欄位列表:", list(cmp_df.columns))
        st.sidebar.write("【CMP】DCC欄/Panel欄:", dcc_col_cmp, "/", panel_col_cmp)

        if dcc_col_cmp and panel_col_cmp:
            def _norm(s: str) -> str:
                return re.sub(r"\s+", "", str(s)).upper()

            panel_set = {_norm(p) for p in selected_panels}

            def contains_any_panel(cell):
                tokens = re.split(r"[、,;|/\\\s]+", str(cell))
                tokens = [_norm(t) for t in tokens if t.strip()]
                return any(t in panel_set for t in tokens)

            matched = cmp_df[cmp_df[panel_col_cmp].apply(contains_any_panel)]
            base_names = (
                matched[dcc_col_cmp]
                .dropna().astype(str).str.strip().unique().tolist()
            )
            st.sidebar.write("【CMP】命中列數:", len(matched))
            st.sidebar.write("【CMP】抓到 base DCC 名稱:", base_names)

            # 嘗試把 base 名稱映射成實際欄位：exact / _CV5 / _TT4 / .PV
            resolved_cols = []
            for base in base_names:
                candidates = [base, f"{base}_CV5", f"{base}_TT4", f"{base}.PV"]
                for c in candidates:
                    if c in merged.columns or c in air_df.columns:
                        resolved_cols.append(c)

            # 併入 air_df 中存在但 merged 尚未有的
            air_extra = [c for c in resolved_cols if c in air_df.columns and c not in merged.columns]
            if air_extra:
                merged = pd.merge(merged, air_df[["DateTime"] + air_extra], on="DateTime", how="outer")

            # 最終 extra_cols
            extra_cols = list(dict.fromkeys(resolved_cols))
            st.sidebar.write("【CMP】實際映射到欄位:", extra_cols)

        else:
            st.sidebar.write("【CMP】找不到 DCC/看板欄位，已略過。")

    except Exception as e:
        st.warning(f"讀取 ComparisonTable 失敗：{e}")
# 若沒提供 comparison_file，則不做 MaxCorr 額外曲線


# === 濾出要畫的欄位 ===
value_cols = []
for d in related_dccs:
    for suf in ['_CV5', '_TT4']:
        col = d + suf
        if col in merged.columns:
            value_cols.append(col)
value_cols += pv_cols + extra_cols
value_cols = list(dict.fromkeys(value_cols))  # 去重

# === 修補：處理欄位重名 / 全空，避免 to_numeric 出錯 ===
if merged.columns.duplicated().any():
    dup_names = pd.unique(merged.columns[merged.columns.duplicated()])
    for name in dup_names:
        same = merged.loc[:, merged.columns == name]
        same_num = same.apply(pd.to_numeric, errors='coerce')
        merged[name] = same_num.bfill(axis=1).iloc[:, 0]
        merged.drop(columns=same.columns[1:], inplace=True, errors='ignore')

# 數值化（若不小心仍是 DataFrame，壓成第一欄）
for c in value_cols:
    if c in merged.columns:
        if isinstance(merged[c], pd.DataFrame):
            merged[c] = merged[c].apply(pd.to_numeric, errors='coerce').bfill(axis=1).iloc[:, 0]
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

# === 依 TT 檔的時間粒度（例如每分鐘）限制顯示，不做平均 ===
# 以 TT 的 DateTime 作為基準單位：若 TT 全部是整分（秒=0），就對齊到每分鐘；
# 若 TT 是 15 秒或 5 秒等倍數，也會自動對齊。否則 fallback 為秒級。
plot_df = merged[["DateTime"] + value_cols].sort_values("DateTime").copy()

# 推斷 TT 的最小顯示單位（基於 TT，而不是 DCC），避免圖上出現 00:01:15 這種細度
_tt_sec = air_df["DateTime"].dt.second
_tt_micro = air_df["DateTime"].dt.microsecond
if (_tt_sec.eq(0) & _tt_micro.eq(0)).all():
    _base_unit = 'T'      # 每分鐘
elif (_tt_sec.mod(15).eq(0) & _tt_micro.eq(0)).all():
    _base_unit = '15S'    # 每 15 秒
elif (_tt_sec.mod(5).eq(0) & _tt_micro.eq(0)).all():
    _base_unit = '5S'     # 每 5 秒
else:
    _base_unit = 'S'      # 秒級

# 將所有資料時間對齊到 TT 的粒度，並在同一時段保留最後一筆（不做平均）
plot_df["DateTime"] = plot_df["DateTime"].dt.floor(_base_unit)
plot_df = plot_df.groupby("DateTime", as_index=False).last()

# === 畫圖 ===
fig = make_subplots(specs=[[{"secondary_y": True}]])
for col in value_cols:
    if col not in plot_df.columns:
        continue

    is_extra = col in extra_cols
    label = col.replace(".PV", "")
    if is_extra:
        label = f"(MC) {label}"

    if col.endswith("_CV5"):
        style = dict(dash="solid", width=2)
        sec_y = False
    else:
        style = dict(dash="dashdot", width=3) if is_extra else dict(dash="dot", width=2)
        sec_y = True

    fig.add_trace(
        go.Scatter(x=plot_df["DateTime"], y=plot_df[col], name=label, line=style),
        secondary_y=sec_y,
    )

fig.update_layout(title=f"{selected_area} / {selected_factory} / {selected_floor}",
                  xaxis=dict(rangeslider=dict(visible=True)))
fig.update_yaxes(title_text="風門開度 (%)", secondary_y=False)
fig.update_yaxes(title_text="溫度 (°C)", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)
