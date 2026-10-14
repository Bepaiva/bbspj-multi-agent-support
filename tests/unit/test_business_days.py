from datetime import date

from app.services.business_days import add_business_days, is_business_day, national_holidays


def test_weekend_is_skipped():
    friday = date(2026, 10, 9)
    assert add_business_days(friday, 1) == date(2026, 10, 13)  # Mon 12/10 is N. Sra. Aparecida


def test_movable_holidays_2026():
    holidays = national_holidays(2026)
    assert date(2026, 4, 3) in holidays   # Good Friday (Easter 05/04/2026)
    assert date(2026, 2, 16) in holidays  # Carnival Monday
    assert date(2026, 6, 4) in holidays   # Corpus Christi


def test_consciencia_negra_is_holiday():
    assert not is_business_day(date(2026, 11, 20))


def test_d_plus_2():
    assert add_business_days(date(2026, 10, 13), 2) == date(2026, 10, 15)
