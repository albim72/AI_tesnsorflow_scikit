#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from typing import List, Tuple

from faker import Faker


def weighted_choice(options: List[Tuple[str, float]]) -> str:
    labels = [o[0] for o in options]
    weights = [o[1] for o in options]
    return random.choices(labels, weights=weights, k=1)[0]


def random_date_between(year_from: int, year_to: int) -> date:
    start = date(year_from, 1, 1)
    end = date(year_to, 12, 31)
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def random_weight_kg() -> int:
    band = weighted_choice([
        ("40-60", 25),
        ("61-85", 30),
        ("86-100", 15),
        ("101-130", 10),
        (">130", 20),
    ])
    if band == "40-60":
        return random.randint(40, 60)
    if band == "61-85":
        return random.randint(61, 85)
    if band == "86-100":
        return random.randint(86, 100)
    if band == "101-130":
        return random.randint(101, 130)
    return random.randint(131, 180)


def random_favorite_color() -> str:
    base = weighted_choice([
        ("czerwony", 5),
        ("niebieski", 9),
        ("biały", 11),
        ("czarny", 18),
        ("żółty", 13),
        ("brązowy", 4),
        ("inne", 40),
    ])
    if base != "inne":
        return base

    inne_kolory = [
        "zielony", "szary", "fioletowy", "różowy", "pomarańczowy",
        "granatowy", "turkusowy", "bordowy", "beżowy", "złoty", "srebrny",
    ]
    return random.choice(inne_kolory)


def random_gender() -> str:
    return random.choice(["kobieta", "mężczyzna"])


def generate_csv(
    output_path: str,
    n: int = 50_000,
    seed: int | None = 42,
    locale: str = "pl_PL",
) -> None:
    if seed is not None:
        random.seed(seed)

    faker = Faker(locale)
    if seed is not None:
        faker.seed_instance(seed)

    fieldnames = [
        "id",
        "imie",
        "nazwisko",
        "lata_pracy",
        "data_urodzenia",
        "miasto",
        "liczba_certyfikatow",
        "ulubiony_kolor",
        "wzrost_cm",
        "plec",
        "waga_kg",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1, n + 1):
            writer.writerow({
                "id": i,
                "imie": faker.first_name(),
                "nazwisko": faker.last_name(),
                "lata_pracy": random.randint(1, 23),
                "data_urodzenia": random_date_between(1950, 2005).isoformat(),
                "miasto": faker.city(),
                "liczba_certyfikatow": random.randint(1, 10),
                "ulubiony_kolor": random_favorite_color(),
                "wzrost_cm": random.randint(120, 210),
                "plec": random_gender(),
                "waga_kg": random_weight_kg(),
            })


# --- URUCHOMIENIE (Datalore/Jupyter) ---
OUT = "synthetic_employees_50000.csv"
N = 50_000
SEED = 42          # ustaw None jeśli chcesz losowo
LOCALE = "pl_PL"

generate_csv(output_path=OUT, n=N, seed=SEED, locale=LOCALE)
print(f"OK: zapisano {N} rekordów do: {OUT}")
