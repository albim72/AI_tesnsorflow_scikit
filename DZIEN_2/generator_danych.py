#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generuje plik CSV z danymi syntetycznymi (50 000 rekordów).

Pola:
- id (1..50000)
- imię (Faker)
- nazwisko (Faker)
- lata_pracy (1..23)
- data_urodzenia (1950..2005)
- miasto (Faker)
- liczba_certyfikatow (1..10)
- ulubiony_kolor (wagowane)
- wzrost_cm (120..210)
- plec (kobieta/mężczyzna)
- waga_kg (wagowane przedziały)
"""

from __future__ import annotations

import csv
import random
import argparse
from datetime import date, timedelta
from typing import List, Tuple

from faker import Faker


def weighted_choice(options: List[Tuple[str, float]]) -> str:
    """Losuje element wg wag (wagi nie muszą sumować się do 1)."""
    labels = [o[0] for o in options]
    weights = [o[1] for o in options]
    return random.choices(labels, weights=weights, k=1)[0]


def random_date_between(year_from: int, year_to: int) -> date:
    """Losuje datę między 1.01.year_from a 31.12.year_to."""
    start = date(year_from, 1, 1)
    end = date(year_to, 12, 31)
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def random_weight_kg() -> int:
    """
    Waga wg rozkładu:
    - 40-60  -> 25%
    - 61-85  -> 30%
    - 86-100 -> 15%
    - 101-130 -> 10%
    - >130 -> pozostałe (20%), przyjmujemy 131-180
    """
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
    # >130
    return random.randint(131, 180)


def random_favorite_color() -> str:
    """
    Kolor wg rozkładu:
    czerwony 5%, niebieski 9%, biały 11%, czarny 18%, żółty 13%, brązowy 4%, inne 40%
    """
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
            row = {
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
            }
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator danych syntetycznych do CSV (50k rekordów).")
    parser.add_argument("--out", default="synthetic_employees_50000.csv", help="Ścieżka do pliku CSV wyjściowego.")
    parser.add_argument("--n", type=int, default=50_000, help="Liczba rekordów (domyślnie 50000).")
    parser.add_argument("--seed", type=int, default=42, help="Seed RNG (domyślnie 42). Ustaw -1 by wyłączyć.")
    parser.add_argument("--locale", default="pl_PL", help="Locale dla Faker (domyślnie pl_PL).")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    generate_csv(output_path=args.out, n=args.n, seed=seed, locale=args.locale)
    print(f"OK: zapisano {args.n} rekordów do: {args.out}")


if __name__ == "__main__":
    main()
