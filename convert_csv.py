import os
import pandas as pd

folder_path = r"C:\Users\User\Documents\sb01_dashboard-main\huai_an\HB04_NOW\RAW"


# 逐一轉換所有 Excel 檔
for file in os.listdir(folder_path):
    if file.endswith(".xlsx") or file.endswith(".xls"):
        excel_path = os.path.join(folder_path, file)
        try:
            # ✅ 只抓第一個工作表（即使有多個 sheet 也一律選第一個）
            df = pd.read_excel(excel_path, sheet_name=0)

            # ✅ 保留原檔名，只改副檔名
            csv_name = os.path.splitext(file)[0] + ".csv"
            csv_path = os.path.join(folder_path, csv_name)

            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 已轉換：{file} → {csv_name}")
        except Exception as e:
            print(f"⚠️ 轉換失敗：{file} → {e}")

print("\n🎉 全部轉換完成！")
