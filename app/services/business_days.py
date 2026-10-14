"""Brazilian business-day calendar (national holidays) used to compute settlement dates."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _easter(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=32)
def national_holidays(year: int) -> frozenset[date]:
    easter = _easter(year)
    fixed = {
        date(year, 1, 1),   # Confraternização Universal
        date(year, 4, 21),  # Tiradentes
        date(year, 5, 1),   # Dia do Trabalho
        date(year, 9, 7),   # Independência
        date(year, 10, 12),  # Nossa Senhora Aparecida
        date(year, 11, 2),  # Finados
        date(year, 11, 15),  # Proclamação da República
        date(year, 11, 20),  # Consciência Negra (national since 2024)
        date(year, 12, 25),  # Natal
    }
    movable = {
        easter - timedelta(days=48),  # Carnaval (Monday) - bank holiday
        easter - timedelta(days=47),  # Carnaval (Tuesday) - bank holiday
        easter - timedelta(days=2),  # Sexta-feira Santa
        easter + timedelta(days=60),  # Corpus Christi - bank holiday
    }
    return frozenset(fixed | movable)


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in national_holidays(d.year)


def add_business_days(start: date, days: int) -> date:
    """Return the date ``days`` business days after ``start`` (D+N convention)."""
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if is_business_day(current):
            added += 1
    return current


def next_business_day_on_or_after(d: date) -> date:
    while not is_business_day(d):
        d += timedelta(days=1)
    return d
