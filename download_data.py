"""Загрузка Hotel Booking Demand CSV из открытого источника TidyTuesday."""
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
DEST = Path(__file__).resolve().parent / "data" / "hotel_bookings.csv"

if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Источник:", URL)
    try:
        with urlopen(Request(URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as response, DEST.open("wb") as file:
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                file.write(data)
        import pandas as pd
        table = pd.read_csv(DEST)
        print("Загружено:", table.shape, "->", DEST)
        if table.shape != (119390, 32):
            raise ValueError(f"Ожидаются 119390 строк, 32 столбца; получено {table.shape}")
    except Exception as exc:
        DEST.unlink(missing_ok=True)
        raise SystemExit(f"Загрузка не удалась: {exc}. Можно вручную положить hotel_bookings.csv в data/")
