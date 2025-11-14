import pandas as pd
import os

# === 你的資料夾路徑 ===
folder_path = r"C:\Users\User\Documents\sb01_dashboard-main\Huai_an\HB04_NOW\RAW"

# 要處理的樓層
floors = ["1F", "2F", "3F", "4F"]

# 日期區間
start_date = "2024-08-01 00:00:00"
end_date   = "2024-09-01 00:00:00"

# 切割每個樓層
for floor in floors:

    # 找符合該樓層的 CSV 檔
    floor_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".csv") and f"_{floor}_" in f
    ]

    if not floor_files:
        print(f"⚠️ 找不到 {floor} 的檔案，略過。")
        continue

    print(f"\n📂 處理 {floor}：找到 {len(floor_files)} 個檔案")

    # 合併該樓層所有 CSV
    combined_df = pd.DataFrame()

    for file in sorted(floor_files):
        csv_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(csv_path)
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df["SourceFile"] = file
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            print(f"✅ 已加入：{file}")
        except Exception as e:
            print(f"⚠️ 無法讀取 {file}：{e}")

    # 篩選 8/1～9/1
    mask = (combined_df["DateTime"] >= start_date) & (combined_df["DateTime"] < end_date)
    df_cut = combined_df[mask]

    # 輸出檔名
    output_name = f"HB04_{floor}_20240801_20240901.csv"
    output_path = os.path.join(folder_path, output_name)

    df_cut.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"🎉 已輸出：{output_path}（共 {len(df_cut)} 筆資料）")

print("\n✨ 全部樓層處理完成！")
