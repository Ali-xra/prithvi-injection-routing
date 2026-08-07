"""
00_test_cds.py — تست اتصال به Copernicus CDS
=============================================
نسخه: v1 · تاریخ: 2026-07-28

این اسکریپت چه می‌کند:
    یک دانلود خیلی کوچک از ERA5-Land انجام می‌دهد تا سه چیز را با هم ثابت کند —
    (۱) کلید API خوانده می‌شود  (۲) مجوز دیتاست پذیرفته شده  (۳) اتصال برقرار است.

چرا لازم است:
    قبل از ساختن خط لولهٔ ۸۰۴ نمونه‌ای، باید مطمئن شویم زنجیره سالم است.
    اگر اینجا خطا بدهد، آنجا هم می‌دهد — ولی اینجا فهمیدنش ارزان است.

خروجی:
    یک فایل NetCDF کوچک در big-files/wind-prithvi/data/era5/

اجرا:
    python 00_test_cds.py
"""

from pathlib import Path
import cdsapi

# ---------------------------------------------------------------
# خروجی بیرون از proje می‌رود — گوگل‌درایو سینک می‌کند
# ---------------------------------------------------------------
OUT_DIR = Path.home() / "Desktop" / "big-files" / "injection-routing" / "data" / "era5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "00_test_wind_2018-07-15.nc"

# ---------------------------------------------------------------
# درخواست: یک ساعت، یک مربع کوچک در کالیفرنیای شمالی
# area به‌ترتیب [شمال، غرب، جنوب، شرق] است
# ---------------------------------------------------------------
REQUEST = {
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    "year": "2018",
    "month": "07",
    "day": "15",
    "time": ["12:00"],
    "area": [40, -122, 39, -121],
    "data_format": "netcdf",
    "download_format": "unarchived",
}


def main():
    print(f"خروجی در: {OUT_FILE}")
    print("در حال اتصال به CDS...\n")

    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-land", REQUEST, str(OUT_FILE))

    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"\n✅ موفق — {size_kb:.1f} کیلوبایت")
    print(f"   {OUT_FILE}")


if __name__ == "__main__":
    main()
