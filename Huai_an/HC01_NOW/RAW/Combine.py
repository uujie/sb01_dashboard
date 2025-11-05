import os
import re
import pandas as pd

# === 設定資料夾 ===
folder_path = os.getcwd()  # 將 convert_csv.py 放在同資料夾中執行即可

# === 正則：偵測像 DCC0701-0708 的日期模式 ===
date_pattern = re.compile(r"DCC(\d{4})-(\d{4})")

# 樓層清單
floors = ["1F", "2F", "3F", "4F"]

for floor in floors:
    combined_df = pd.DataFrame()

    # 篩選：包含樓層、DCC、且有日期格式的 Excel 檔案
    floor_files = [
        f for f in os.listdir(folder_path)
        if (f.endswith(".xlsx") or f.endswith(".xls"))
        and (f"-{floor}-" in f or f"_{floor}_" in f)
        and date_pattern.search(f)
    ]
    if not floor_files:
        print(f"⚠️ 沒有找到 {floor} 的符合條件的檔案，略過。")
        continue

    # === 依日期排序 ===
    def extract_start_date(filename):
        match = date_pattern.search(filename)
        return int(match.group(1)) if match else 0

    floor_files = sorted(floor_files, key=extract_start_date)

    print(f"\n📂 處理 {floor} 的 {len(floor_files)} 個檔案...")

    # === 合併資料 ===
    for file in floor_files:
        excel_path = os.path.join(folder_path, file)
        try:
            df = pd.read_excel(excel_path, sheet_name=0)
            df["SourceFile"] = os.path.splitext(file)[0]  # 標示來源檔
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f"✅ 已加入：{file}")
        except Exception as e:
            print(f"⚠️ 讀取失敗 {file}：{e}")

    # === 輸出結果 ===
    output_csv = os.path.join(folder_path, f"HC01-{floor}-DCC.csv")
    combined_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"🎉 已匯出：{output_csv}")

print("\n✅ 全部樓層合併完成並依日期排序！")
