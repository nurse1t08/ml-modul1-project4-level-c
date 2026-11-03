"""Генератор исключительно синтетических данных для демонстрации без интернета."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(n=3500, seed=42):
    g = np.random.default_rng(seed)
    hotel = g.choice(["City Hotel", "Resort Hotel"], n, p=[.67, .33])
    lead = g.integers(0, 450, n)
    segment = g.choice(["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate"], n,
                       p=[.48, .20, .17, .10, .05])
    deposit = g.choice(["No Deposit", "Non Refund", "Refundable"], n, p=[.82, .12, .06])
    repeated = g.binomial(1, .05, n)
    previous = g.poisson(.22, n)
    years = g.choice([2015, 2016, 2017], n, p=[.28, .44, .28])
    children = g.poisson(.12, n).astype(float)
    children[g.choice(n, min(n, 4), replace=False)] = np.nan
    country = g.choice(["PRT", "GBR", "FRA", "ESP", "DEU", "BRA", None], n,
                       p=[.35, .24, .14, .09, .09, .07, .02])
    adults = g.choice([0, 1, 2, 3, 4], n, p=[.015, .14, .68, .14, .025])
    babies = g.binomial(1, .01, n)
    logit = (-1.32 + .0025 * lead + .65 * (hotel == "City Hotel")
             + .8 * (deposit == "Non Refund") + .3 * (segment == "Online TA")
             - .6 * repeated + .15 * previous)
    p = 1 / (1 + np.exp(-logit))
    canceled = g.binomial(1, p)
    data = pd.DataFrame({
        "hotel": hotel, "is_canceled": canceled, "lead_time": lead,
        "arrival_date_year": years,
        "arrival_date_month": g.choice(["January", "March", "June", "August", "October"], n),
        "arrival_date_week_number": g.integers(1, 53, n),
        "arrival_date_day_of_month": g.integers(1, 29, n),
        "stays_in_weekend_nights": g.integers(0, 4, n),
        "stays_in_week_nights": g.integers(0, 8, n),
        "adults": adults, "children": children, "babies": babies,
        "meal": g.choice(["BB", "HB", "SC", "FB"], n), "country": country,
        "market_segment": segment,
        "distribution_channel": g.choice(["TA/TO", "Direct", "Corporate"], n),
        "is_repeated_guest": repeated, "previous_cancellations": previous,
        "previous_bookings_not_canceled": g.poisson(.3, n),
        "reserved_room_type": g.choice(list("ABCDEF"), n),
        "assigned_room_type": g.choice(list("ABCDEFG"), n),
        "booking_changes": g.poisson(.4 + canceled * .2),
        "deposit_type": deposit, "agent": g.choice([1., 2., 5., 10., np.nan], n),
        "company": g.choice([np.nan, 1., 4.], n, p=[.94, .03, .03]),
        "days_in_waiting_list": g.poisson(.3, n),
        "customer_type": g.choice(["Transient", "Contract", "Group"], n),
        "adr": g.lognormal(4.5, .35, n),
        "required_car_parking_spaces": g.binomial(1, .08, n),
        "total_of_special_requests": g.poisson(.5, n),
        "reservation_status": np.where(canceled == 1, "Canceled", "Check-Out"),
        "reservation_status_date": g.choice(["2016-01-01", "2017-04-15", "2017-08-01"], n),
    })
    if n > 0:
        data.loc[0, "adr"] = -4
    if n > 1:
        data.loc[1, "adr"] = 2000
    if n > 2:
        data.loc[2, ["adults", "children", "babies"]] = [0, 0, 0]
    return data

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "data" / "hotel_bookings_demo.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(path, index=False)
    print("СИНТЕТИЧЕСКИЕ ДАННЫЕ:", df.shape, "->", path)
