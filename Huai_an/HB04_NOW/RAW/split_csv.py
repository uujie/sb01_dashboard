import os
import re
import pandas as pd

# === 當前資料夾 ===
BASE_PATH = os.getcwd()

# === 日期區間設定 ===
RANGES = [
    ("0501", "0531"),
    ("0601", "0630"),
    ("0701", "0731"),
    ("0801", "0901"),
]

# === 讀取函式 ===
def load_table(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig")
    else:
        return pd.read_excel(path, sheet_name=0)

# === 主拆分函式 ===
def split_file(file_path):
    print(f"\n📂 正在處理：{os.path.basename(file_path)}")

    try:
        df = load_table(file_path)
    except Exception as e:
        print(f"❌ 無法讀取 {file_path}\n  原因：{e}")
        return

    if "DateTime" not in df.columns:
        print("⚠️ 找不到 DateTime 欄位，略過。")
        return

    # 日期處理
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    if df.empty:
        print("⚠️ DateTime 欄位為空，略過。")
        return

    year = int(df["DateTime"].dt.year.mode()[0])

    # 檔名處理：移除像 0501-0901 的字樣
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    clean_name = re.sub(r"[-_]?0[5-9]01[-_]0[5-9]0[1-9]", "", base_name)  # 移除 0501-0901 或類似字樣

    # 依區間輸出
    for start, end in RANGES:
        start_date = pd.Timestamp(f"{year}-{start[:2]}-{start[2:]}")
        end_date = pd.Timestamp(f"{year}-{end[:2]}-{end[2:]}")
        mask = (df["DateTime"] >= start_date) & (df["DateTime"] <= end_date)
        sub = df.loc[mask].copy()

        if sub.empty:
            print(f"  ⚠️ {start}-{end} 無資料，略過。")
            continue

        out_name = f"{clean_name}_{start}-{end}.xlsx"
        out_file = os.path.join(BASE_PATH, out_name)
        sub.to_excel(out_file, index=False)
        print(f"  ✅ 已輸出：{out_name} ({len(sub):,} 筆資料)")

# === 掃描 DCC 檔 ===
all_files = [
    os.path.join(BASE_PATH, f)
    for f in os.listdir(BASE_PATH)
    if f.lower().endswith((".csv", ".xlsx")) and "dcc" in f.lower()
]

if not all_files:
    print("❌ 找不到任何 DCC 檔案，請確認此程式是否放在 RAW 資料夾內。")
else:
    print(f"🔍 偵測到 {len(all_files)} 個 DCC 檔案，開始拆分...\n")
    for file in all_files:
        split_file(file)

print("\n🎉 全部樓層拆分完成！")
print("📁 所有新檔案已放在原本資料夾中。")
