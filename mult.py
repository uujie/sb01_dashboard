import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 自動掃描 QA02 RAW 中所有 DCC 檔案（CSV）===
def scan_qa02_dcc(base_path="CHD/QA02_NOW/QA02 RAW"):
    dcc_map = {"1F": [], "2F": [], "3F": [], "4F": []}
    if not os.path.exists(base_path):
        return dcc_map
    for file in os.listdir(base_path):
        if file.endswith(".csv") and "DCC" in file.upper():
            f_upper = file.upper()
            for floor in dcc_map.keys():
                if f"-{floor}-" in f_upper:
                    dcc_map[floor].append(os.path.join(base_path, file))
    return dcc_map

# === 通用掃描（支援 .xlsx/.csv，並自動偵測樓層）===
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
    return {k: v for k, v in dcc_map.items() if v}

# === 小工具：讀表（自動判斷 CSV / Excel）===
def read_table(path, **kwargs):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, **kwargs)
    else:
        return pd.read_excel(path, sheet_name=0, **kwargs)

# === Huai_an 專用功能：生成看板與 DCC 對應、欄位重命名（支援一對多） ===
def build_huai_an_mapping(layout_df, dcc_df_cols):
    """
    從 Huai_an layout 建立「溫溼度看板 -> DCC(list)」對應表，
    並根據 DCC檔案實際欄位生成重命名字典。
    自動支援繁體 / 簡體 / 空白混用。
    """
    # ---- 容忍繁簡與空白 ----
    col_map = {re.sub(r"\s+", "", c): c for c in layout_df.columns}
    dcc_col = None
    panel_col = None
    for c in layout_df.columns:
        name = re.sub(r"\s+", "", c)
        if name in ["DCC名稱", "DCC名称"]:
            dcc_col = c
        elif name in ["溫溼度看板", "温湿度看板"]:
            panel_col = c

    mapping = {}
    if dcc_col and panel_col:
        for _, row in layout_df.iterrows():
            dcc = str(row[dcc_col]).strip()
            panel = str(row[panel_col]).strip()
            if panel and dcc and panel.lower() != "nan" and dcc.lower() != "nan":
                mapping.setdefault(panel, []).append(dcc)  # ✅ 支援多個 DCC

    # ---- 產生重命名對照 ----
    rename_map = {}
    for col in dcc_df_cols:
        if not isinstance(col, str):
            continue
        new_name = col
        if col.endswith(".PV"):
            new_name = col.replace(".PV", "_TT4")
        elif col.endswith(".PID_CV"):
            new_name = col.replace(".PID_CV", "_CV5")
        new_name = new_name.strip()
        if new_name != col:
            rename_map[col] = new_name

    return mapping, rename_map

# === 預設路徑 ===
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

#==== Huai_an ====
hb01_raw_path = "Huai_an/HB01_NOW/RAW"
hb01_layout_path = "Huai_an/HB01_NOW/HB01_layout_path_20250313V1.xlsx"
hb01_dcc_map = scan_dcc_any(base_path=hb01_raw_path, floors_hint=("1F", "2F", "3F", "4F"), exts=(".xlsx", ".csv"))


# === 區域設定 ===
area_config = {
    "SJ": {
        "factories": {
            "SB01": {
                "layout_file": "./SJ/SB01_NOW/SB01 layout_20250421.xlsx",
                "comparison_file": "./SJ/SB01_NOW/SB01_ComparisonTable.xlsx",  # Max correlation / INAP 對應表
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
                    "2F": {"tt": "./SJ/SB02_NOW/RAW/SB02_2F_TT_202412.csv", "dcc": "./SJ/SB02_NOW/RAW/SB02-2F-DCC1201-1231.csv", "pv_replace": None},
                    "3F": {"tt": "./SJ/SB02_NOW/RAW/SB02_3F_TT_202412.csv", "dcc": "./SJ/SB02_NOW/RAW/SB02-3F-DCC1201-1231.csv", "pv_replace": None},
                    "4F": {"tt": "./SJ/SB02_NOW/RAW/SB02_4F_TT_202412.csv", "dcc": "./SJ/SB02_NOW/RAW/SB02-4F-DCC1201-1231.csv", "pv_replace": None},
                    "5F": {"tt": "./SJ/SB02_NOW/RAW/SB02_5F_TT_202412.csv", "dcc": "./SJ/SB02_NOW/RAW/SB02-5F-DCC1201-1231.csv", "pv_replace": None},
                    "6F": {"tt": "./SJ/SB02_NOW/RAW/SB02_6F_TT_202412.csv", "dcc": "./SJ/SB02_NOW/RAW/SB02-6F-DCC1201-1231.csv", "pv_replace": None},
                    "7F": {"tt": "./SJ/SB02_NOW/RAW/SB02_7F_TT_202412.csv", "dcc": "./SJ/SB02_NOW/RAW/SB02-7F-DCC1201-1231.csv", "pv_replace": None},
                },
            },
            "SL01": {
                "layout_file": "SJ/SL01_NOW/SL01 layout - 20250421.xlsx",
                "floors": {
                    floor: {
                        "tt": os.path.join("SJ/SL01_NOW/SL-RAW_20240418", f"SL_{floor}_TT_202305.csv"),
                        "dcc_multi": dcc_list,
                        "pv_replace": None if floor in ["1F"] else ("_TTHTPT_TT", "_TTHTPT_HT"),
                    }
                    for floor, dcc_list in scan_dcc_any("SJ/SL01_NOW/SL-RAW_20240418", ("1F","2F","3F","4F","5F","6F"), (".csv",)).items()
                },
            },
        },
    },
    "CHD": {
        "factories": {
            "QA02": {
                "layout_file": qa02_layout_path,
                "comparison_file": "./CHD/QA02_NOW/A02_ComparisonTable.xlsx",
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

# === Huai_an 自動設定 ===
huai_an_factories = ["A8", "A9", "B6", "B7", "HB01", "HB04", "HB06", "HC01"]
huai_an_config = {}

for f in huai_an_factories:
    base = f"Huai_an/{f}_NOW"
    raw_path = os.path.join(base, "RAW")

    if not os.path.exists(raw_path):
        print(f"⚠️ 找不到 {raw_path}，略過 {f}")
        continue

    # 自動找 layout 檔案
    layout_file = next(
        (os.path.join(base, file)
         for file in os.listdir(base)
         if "layout" in file.lower() and file.endswith((".xlsx", ".xls"))),
        None
    )
    if not layout_file:
        print(f"⚠️ {f} 缺少 layout 檔案，略過。")
        continue

    cmp_path = os.path.join(base, f"{f}_ComparisonTable.xlsx")

    # 掃 RAW 裡的 DCC 檔案
    dcc_map = scan_dcc_any(
        base_path=raw_path,
        floors_hint=("1F", "2F", "3F", "4F"),
        exts=(".xlsx", ".csv")
    )

    floors_cfg = {}
    for floor, dcc_list in dcc_map.items():
        # 找樓層對應的 TT 檔案
        tt_file = next(
            (os.path.join(raw_path, file)
             for file in os.listdir(raw_path)
             if floor.lower() in file.lower() and "tt" in file.lower()),
            None
        )

        if not tt_file:
            print(f"⚠️ {f} {floor} 未找到 TT 檔案，略過。")
            continue

        floors_cfg[floor] = {
            "tt": tt_file,
            "dcc_multi": dcc_list,
            "pv_replace": ("_TTHT_TT", "_TTHT_HT") if floor != "1F" else None,
        }

    # 加入設定
    if floors_cfg:
        huai_an_config[f] = {
            "layout_file": layout_file,
            "comparison_file": cmp_path if os.path.exists(cmp_path) else None,
            "floors": floors_cfg,
        }

# ✅ 最後這一行：把 Huai_an 加進 area_config
area_config["Huai_an"] = {"factories": huai_an_config}
# === Streamlit UI ===
st.set_page_config(page_title="DCC/TT 每分鐘真實曲線", layout="wide")
selected_area = st.sidebar.selectbox("選擇廠區", list(area_config.keys()))
selected_factory = st.sidebar.selectbox("選擇工廠", list(area_config[selected_area]["factories"].keys()))
factory_cfg = area_config[selected_area]["factories"][selected_factory]

def sort_floors(floors):
    order = {'1F':1,'2F':2,'3F':3,'4F':4,'5F':5,'6F':6}
    return sorted(floors, key=lambda f: order.get(f,999))

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
tt_time_col = next((c for c in air_df.columns if c.strip().lower() in ["datetime","startdatetime"]), None)
if not tt_time_col:
    st.error("❌ TT 檔案中找不到 DateTime / StartDateTime 欄位")
    st.stop()
air_df["DateTime"] = pd.to_datetime(air_df[tt_time_col])

# === 讀 DCC ===
dcc_df = None
if "dcc_multi" in floor_cfg:
    parts = []
    for p in floor_cfg["dcc_multi"]:
        df = read_table(p)
        df.columns = df.columns.str.strip()
        dt_col = next((c for c in df.columns if c.strip().lower() in ["datetime","startdatetime"]), None)
        if dt_col:
            df["DateTime"] = pd.to_datetime(df[dt_col])
            parts.append(df)
    dcc_df = pd.concat(parts, ignore_index=True)
else:
    dcc_df = read_table(floor_cfg["dcc"])
    if "DateTime" not in dcc_df.columns:
        dt = next((c for c in dcc_df.columns if str(c).lower()=="datetime"), None)
        if dt: dcc_df.rename(columns={dt:"DateTime"}, inplace=True)
    dcc_df["DateTime"] = pd.to_datetime(dcc_df["DateTime"])

# === Huai_an 專用欄位重命名 ===
if selected_area == "Huai_an":
    mapping, rename_map = build_huai_an_mapping(layout_df, dcc_df.columns)
    if rename_map:
        dcc_df.rename(columns=rename_map, inplace=True)
        st.sidebar.write("【Huai_an】已自動重命名欄位：", rename_map)
    if mapping:
        st.sidebar.write("【Huai_an】看板-DCC 對應表：")
        st.sidebar.dataframe(pd.DataFrame(list(mapping.items()), columns=["溫溼度看板","DCC名稱"]))

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

# === PV 欄位對應（加強：同時嘗試「原始」與「替換後」兩種 base）===
def _norm(s: str) -> str:
    return re.sub(r"[-\s]+", "_", str(s)).strip().upper()

pv_cols = []
for base in map(lambda x: str(x).strip(), pv_bases):
    candidates = {base}
    if floor_cfg.get("pv_replace"):
        candidates.add(base.replace(*floor_cfg["pv_replace"]))

    # 💡 特殊處理 Huai_an：ROOM → DCC
    if selected_area == "Huai_an":
        new_candidates = set()
        for c in candidates:
            new_candidates.add(c.replace("_ROOM_", "_DCC_"))
        candidates |= new_candidates  # 合併新候選

    # 搜索欄位
    for cand in candidates:
        exacts = [cand, f"{cand}.PV", f"{cand}.PID_CV"]
        for exact in exacts:
            if exact in air_df.columns and exact not in pv_cols:
                pv_cols.append(exact)
                continue
        # 模糊搜尋
        for c in air_df.columns:
            if any(_norm(cand) in _norm(cand2) for cand2 in [c]):
                if c not in pv_cols:
                    pv_cols.append(c)

pv_cols = list(dict.fromkeys(pv_cols))  # 去重、保序


pv_cols = []
if selected_area == "CHD" and selected_factory in ("QA02", "QA08"):
    # CHD QA02/QA08 維持嚴格的 base+".PV" 對應
    for base in pv_bases:
        pv_name = f"{base}.PV"
        if pv_name in air_df.columns:
            pv_cols.append(pv_name)
else:
    for base in map(lambda x: str(x).strip(), pv_bases):
        candidates = {base}
        if floor_cfg.get("pv_replace"):
            candidates.add(base.replace(*floor_cfg["pv_replace"]))
        # 先試精確，再做包含式
        for cand in candidates:
            exact = f"{cand}.PV"
            if exact in air_df.columns and exact not in pv_cols:
                pv_cols.append(exact)
                continue
            for c in air_df.columns:
                if str(c).upper().endswith(".PV") and _norm(cand) in _norm(c):
                    if c not in pv_cols:
                        pv_cols.append(c)

pv_cols = list(dict.fromkeys(pv_cols))  # 去重、保序


if not pv_cols:
    st.error("找不到對應 PV 欄位")
    st.write("TT檔案所有欄位：", air_df.columns.tolist())
    st.write("Layout指定的點位：", pv_bases)
    st.stop()

# === 合併資料（不重取樣）===
merged = pd.merge(dcc_df, air_df[["DateTime"] + pv_cols], on="DateTime", how="outer").sort_values("DateTime")

# === 從 ComparisonTable 讀取：同時支援多欄位的 MC 與 INAP ===
# === 從 ComparisonTable 讀取（統一所有廠區的 MC / INAP 邏輯）===
extra_cols = []   # 額外要畫的欄位
extra_tags = {}   # 欄位 -> "MC" 或 "INAP"

cmp_path = factory_cfg.get("comparison_file")
if cmp_path and os.path.exists(cmp_path):
    try:
        cmp_book = pd.read_excel(cmp_path, sheet_name=None)

        # 精確匹配樓層，否則退回第一張
        if selected_floor in cmp_book:
            cmp_df = cmp_book[selected_floor].copy()
            sheet_used = selected_floor
        else:
            sheet_used = next(iter(cmp_book.keys()))
            cmp_df = cmp_book[sheet_used].copy()

        cmp_df.columns = cmp_df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)

        # 找 DCC 欄位
        dcc_col_cmp = next((
            c for c in cmp_df.columns
            if any(k in c.lower() for k in ["dcc_name", "dcc", "dcc名稱", "dcc名称"])
        ), None)

        # 找 MC / INAP 欄位
        mc_cols   = [c for c in cmp_df.columns if "max correlation" in c.lower()]
        inap_cols = [c for c in cmp_df.columns if "inappropriate" in c.lower() or "不適當" in c.lower()]

        st.sidebar.write("【CMP】使用工作表:", sheet_used)
        st.sidebar.write("【CMP】DCC欄:", dcc_col_cmp)
        st.sidebar.write("【CMP】MC欄位清單:", mc_cols)
        st.sidebar.write("【CMP】INAP欄位清單:", inap_cols)

        # === 核心邏輯 ===
        tagged_bases = []
        def _split_tokens(cell):
            return [t.strip() for t in re.split(r"[、,;|/\\\s]+", str(cell)) if t.strip()]
        def _keyize(s):
            return re.sub(r"[-\s]+", "_", str(s)).upper()
        def _normalize_panel_name(name: str, area: str) -> str:
            """
            將看板名稱正規化：
            - 移除空白、底線
            - Huai_an: 將 TTHT_01 → TTHT201 類型轉換
            """
            name = str(name).strip().upper().replace(" ", "")
            if area == "Huai_an" and re.match(r"TTHT_\d{2}", name):
                num = re.findall(r"\d{2}", name)[0]
                return f"TTHT2{num}"  # ✅ 轉換邏輯：_01 → 201, _02 → 202, ...
            return re.sub(r"[^A-Z0-9]", "", name)

        # --- MC ---
        for col in mc_cols:
            for _, row in cmp_df.iterrows():
                tokens = [_keyize(t) for t in _split_tokens(row[col])]
                if any(_normalize_panel_name(t, selected_area) == _normalize_panel_name(sp, selected_area)for t in tokens for sp in selected_panels):
                    dcc_bases = _split_tokens(row[dcc_col_cmp])
                    for b in dcc_bases:
                        tagged_bases.append((b, "MC"))

        # --- INAP ---
        for col in inap_cols:
            for _, row in cmp_df.iterrows():
                tokens = [_keyize(t) for t in _split_tokens(row[col])]
                if any(_normalize_panel_name(t, selected_area) == _normalize_panel_name(sp, selected_area)for t in tokens for sp in selected_panels):
                    dcc_bases = _split_tokens(row[dcc_col_cmp])
                    for b in dcc_bases:
                        tagged_bases.append((b, "INAP"))

        # === 映射到實際欄位 ===
        def _match_col(base, df_cols):
            key = _keyize(base)
            for c in df_cols:
                if key in _keyize(c):
                    return c
            return None

        merged_cols_u = list(merged.columns)
        air_cols_u = list(air_df.columns)

        resolved = []
        for base, tag in tagged_bases:
            c = _match_col(base, merged_cols_u) or _match_col(base, air_cols_u)
            if c:
                resolved.append((c, tag))

        extra_cols = []
        extra_tags = {}
        for c, tag in resolved:
            if c not in extra_cols:
                extra_cols.append(c)
            extra_tags[c] = tag

        st.sidebar.write("【CMP】實際映射到欄位:", extra_cols)
        st.sidebar.write("【CMP】欄位標籤:", extra_tags)

    except Exception as e:
        st.warning(f"讀取 ComparisonTable 失敗：{e}")


# 若沒提供 comparison_file，則不做 MC/INAP 額外曲線
""""""
if selected_area == "Huai_an":
    huai_an_mapping, _ = build_huai_an_mapping(layout_df, dcc_df.columns)
    st.sidebar.subheader("【DEBUG 6】開始找 DCC 欄位對應")
    for panel in selected_panels:
        st.sidebar.write(f"🔍 看板: {panel}")
        if panel in huai_an_mapping:
            dcc_list = huai_an_mapping[panel]
            st.sidebar.write(f"  → 對應 DCC: {dcc_list}")
            all_hits = []
            for dcc_base in dcc_list:
                local_hits = [c for c in merged.columns if re.search(dcc_base.replace('_', ''), c.replace('_', ''), re.I)]
                all_hits.extend(local_hits)
            st.sidebar.write(f"  → 匹配到欄位: {all_hits}")
# === 濾出要畫的欄位（Huai_an 進階版：支援 TT28 / TCV28 雙關聯） ===
value_cols = []

if selected_area == "Huai_an":
    # 使用 mapping 建立反查：看板 → DCC
    huai_an_mapping, _ = build_huai_an_mapping(layout_df, dcc_df.columns)

    for panel in selected_panels:
        if panel in huai_an_mapping:
            dcc_list = huai_an_mapping[panel]  # e.g. ["TCV_02", "TCV_03"]
            for dcc_base in dcc_list:
                dcc_variants = {
                    dcc_base,
                    dcc_base.replace("_", ""),
                    dcc_base.replace("TCV_", "TT"),
                    dcc_base.replace("TCV_", "TT").replace("_", "")
                }
                for c in merged.columns:
                    for variant in dcc_variants:
                        if re.search(variant, c.replace("_", ""), re.I):
                            if any(suf in c for suf in ["_CV5", "_TT4", ".PID_CV", ".PV"]):
                                value_cols.append(c)
                                break

else:
    for d in related_dccs:
        for suf in ['_CV5', '_TT4', '.PID_CV', '.PV']:
            col = d + suf
            if col in merged.columns:
                value_cols.append(col)

value_cols += pv_cols + extra_cols
value_cols = list(dict.fromkeys(value_cols))  # 去重、保序

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

# === 依 TT 檔的時間粒度限制顯示（不做平均）===
plot_df = merged[["DateTime"] + value_cols].sort_values("DateTime").copy()

# 推斷 TT 的最小顯示單位
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

# 對齊到 TT 的粒度，在同一時段保留最後一筆（不做平均）
plot_df["DateTime"] = plot_df["DateTime"].dt.floor(_base_unit)
plot_df = plot_df.groupby("DateTime", as_index=False).last()

st.sidebar.write("【DEBUG】extra_tags內容：", extra_tags)

# === 畫圖 ===
# === 【新增】偵測未被匹配到的 MC/INAP 欄位 ===
pattern_fix = lambda s: re.sub(r"[_\s]", "", str(s)).upper()

unmatched_cols = []
for col in extra_cols:
    found = any(
        pattern_fix(col) in pattern_fix(c)
        or pattern_fix(c).endswith(pattern_fix(col))
        or pattern_fix(col).endswith(pattern_fix(c))
        for c in merged.columns
    )
    if not found:
        unmatched_cols.append(col)

if unmatched_cols:
    st.sidebar.write("⚠️ 以下 ComparisonTable 偵測到但無法對應到實際欄位（未畫出）:")
    for u in unmatched_cols:
        st.sidebar.write("　🔸", u)


# === 【改進】欄位搜尋函式 ===
def find_draw_col(colname, df_columns):
    """放寬欄位匹配條件，允許略有差異的命名"""
    norm_target = pattern_fix(colname)
    for c in df_columns:
        norm_col = pattern_fix(c)
        # 結尾相同、完全相同、或其中之一包含另一個 → 即視為匹配
        if (
            norm_col == norm_target
            or norm_col.endswith(norm_target)
            or norm_target.endswith(norm_col)
            or norm_target in norm_col
        ):
            return c
    return None


# === 【修改後】畫圖邏輯 ===
fig = make_subplots(specs=[[{"secondary_y": True}]])

for col in value_cols:
    draw_col = find_draw_col(col, plot_df.columns)  # 🆕 改用放寬比對
    if not draw_col:
        continue

    # === 標籤判定 ===
    tag = extra_tags.get(col) or extra_tags.get(draw_col)
    if not tag:
        for k, t in extra_tags.items():
            if re.search(k.replace("_", ""), draw_col.replace("_", ""), re.I):
                tag = t
                break

    prefix = f"({tag}) " if tag else ""
    label = prefix + draw_col.replace(".PV", "_TT4").replace(".PID_CV", "_CV5")

    # === 判斷主軸 (CV5) or 副軸 (TT4) ===
    if any(x in draw_col for x in ["_CV5", ".PID_CV"]):
        style = dict(dash="solid", width=2 if tag != "INAP" else 3)
        sec_y = False
    else:
        if tag == "INAP":
            style = dict(dash="dashdot", width=3)
        elif tag == "MC":
            style = dict(dash="dash", width=2)
        else:
            style = dict(dash="dot", width=2)
        sec_y = True

    # === 加入曲線 ===
    fig.add_trace(
        go.Scatter(
            x=plot_df["DateTime"],
            y=plot_df[draw_col],
            mode="lines",
            name=label,
            line=style,
            yaxis="y2" if sec_y else "y1",
        )
    )

# === 設定圖表樣式 ===
fig.update_layout(
    hovermode="x",
    xaxis=dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="black",
        spikethickness=1.5
    ),
    hoverdistance=50,
    spikedistance=50,
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_color="black"
    )
)


# ⭐ 最重要的：把 hover box 固定在右側
fig.update_layout(hoverlabel_align='right')


fig.update_yaxes(title_text="風門開度 (%)", secondary_y=False)
fig.update_yaxes(title_text="溫度 (°C)", secondary_y=True)
st.plotly_chart(fig, use_container_width=True)
