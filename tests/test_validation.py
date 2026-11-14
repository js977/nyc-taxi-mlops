import pandas as pd
import pytest

from nyc_taxi.data.validate import (
    validate_required_columns,
    validate_missing_values,
    validate_duplicate_rows,
    validate_numeric_value,
    validate_timestamps,
    str_validate_day_week,
    number_validate_day_week,
    validate_hours,
    compare_timestamp_pick_hour,
    validate_nonnegative_counts,
    validate_dataframe,
    validate_fare_prediction_training_data,
    validate_demand_training_data,
)


# Individual validation functions

def test_required_columns_reports_missing_columns_in_requested_order():
    df = pd.DataFrame({
        "fare_amount": [10],
        "PULocationID": [161],
    })

    missing = validate_required_columns(
        df,
        ["fare_amount", "DOLocationID", "pickup_hour"],
    )

    assert missing == ["DOLocationID", "pickup_hour"]


def test_required_columns_accepts_extra_columns():
    df = pd.DataFrame({
        "fare_amount": [10],
        "extra": [1],
    })

    assert validate_required_columns(df, ["fare_amount"]) == []


def test_missing_values_counts_per_column():
    df = pd.DataFrame({
        "fare_amount": [10, None, 20],
        "PULocationID": [None, 161, None],
    })

    missing = validate_missing_values(
        df,
        ["fare_amount", "PULocationID"],
    )

    assert missing == {
        "fare_amount": 1,
        "PULocationID": 2,
    }


def test_duplicate_rows_counts_repeated_rows_after_first_occurrence():
    df = pd.DataFrame({
        "a": [1, 1, 1, 2],
        "b": [3, 3, 3, 4],
    })

    assert validate_duplicate_rows(df) == 2


@pytest.mark.parametrize(
    ("values", "positive_only", "expected"),
    [
        ([10, 0, -2, "bad", None, float("inf")], False, 3),
        ([10, 0, -2, "bad", None, float("inf")], True, 5),
    ],
)
def test_numeric_value_general(values, positive_only, expected):
    df = pd.DataFrame({"fare_amount": values})

    assert validate_numeric_value(
        df,
        "fare_amount",
        positive_only=positive_only,
    ) == expected


@pytest.mark.parametrize("column", ["PULocationID", "DOLocationID"])
def test_location_ids_must_be_positive_integers(column):
    df = pd.DataFrame({
        column: [1, "2", 0, -1, 3.5, "bad"],
    })

    assert validate_numeric_value(df, column) == 4


def test_timestamps_detect_unparseable_and_missing_values():
    df = pd.DataFrame({
        "time": [
            "2025-01-01 10:00:00",
            "not-a-date",
            None,
        ],
    })

    assert validate_timestamps(df, "time") == 2


def test_weekday_names_are_exact_and_case_sensitive():
    df = pd.DataFrame({
        "day_of_week": [
            "Monday",
            "Sunday",
            "monday",
            "Friday ",
            None,
        ],
    })

    assert str_validate_day_week(df) == 3


def test_numeric_weekdays_must_be_integers_between_zero_and_six():
    df = pd.DataFrame({
        "day_of_week": [0, 6, "3", 7, -1, 2.5, "Monday", None],
    })

    assert number_validate_day_week(df) == 5


def test_pickup_hours_must_be_integers_between_zero_and_twenty_three():
    df = pd.DataFrame({
        "pickup_hour": [0, 12, 23, 24, -1, 10.5, "bad", None],
    })

    assert validate_hours(df) == 5


def test_timestamp_hour_mismatches_only_count_comparable_rows():
    df = pd.DataFrame({
        "timestamp": [
            "2025-01-01 10:30",
            "2025-01-01 11:30",
            "bad",
            "2025-01-01 13:00",
        ],
        "pickup_hour": [10, 12, 9, None],
    })

    assert compare_timestamp_pick_hour(
        df,
        "timestamp",
        "pickup_hour",
    ) == 1


def test_nonnegative_counts_reject_negative_fractional_and_nonfinite_values():
    df = pd.DataFrame({
        "pickup_count": [0, 2, "3", -1, 1.5, "bad", None, float("inf")],
    })

    assert validate_nonnegative_counts(df, "pickup_count") == 5


# General DataFrame validation

def test_validate_dataframe_returns_error_for_empty_dataset():
    report = validate_dataframe(
        pd.DataFrame(),
        ["fare_amount"],
    )

    assert report == {
        "errors": ["Dataset is empty!"],
        "warnings": [],
    }


def test_validate_dataframe_returns_early_for_missing_columns():
    df = pd.DataFrame({
        "fare_amount": [10],
    })

    report = validate_dataframe(
        df,
        ["fare_amount", "PULocationID"],
    )

    assert "Missing columns" in report["errors"][0]
    assert report["warnings"] == []


def test_validate_dataframe_reports_duplicate_warning_and_invalid_values():
    df = pd.DataFrame({
        "fare_amount": ["bad", "bad"],
        "PULocationID": [161, 161],
    })

    report = validate_dataframe(
        df,
        ["fare_amount", "PULocationID"],
    )

    assert any(
        "duplicate rows" in warning
        for warning in report["warnings"]
    )
    assert any(
        "fare_amount: 2 invalid numeric values" in error
        for error in report["errors"]
    )


# Example datasets

def make_fare_df():
    return pd.DataFrame({
        "fare_amount": [12.5, 15.0],
        "PULocationID": [161, 162],
        "DOLocationID": [163, 164],
        "pickup_hour": [10, 11],
        "day_of_week": ["Wednesday", "Wednesday"],
        "tpep_pickup_datetime": [
            "2025-01-01 10:00",
            "2025-01-01 11:30",
        ],
    })


def make_demand_df():
    times = pd.date_range(
        "2025-01-01 00:00",
        periods=169,
        freq="h",
    )
    counts = pd.Series(range(1, 170), dtype="int64")

    df = pd.DataFrame({
        "pickup_datetime_hour": times,
        "PULocationID": [161] * len(times),
        "pickup_count": counts,
        "pickup_hour": times.hour,
        "day_of_week": times.dayofweek,
        "lag_1h": counts.shift(1),
        "lag_24h": counts.shift(24),
        "lag_168h": counts.shift(168),
    })

    return df.iloc[168:].reset_index(drop=True)


# Fare prediction training data

def test_fare_training_data_accepts_valid_rows():
    report = validate_fare_prediction_training_data(make_fare_df())

    assert report["errors"] == []


def test_fare_training_data_rejects_bad_weekday_hour_and_nonpositive_fare():
    df = make_fare_df()
    df.loc[0, "day_of_week"] = "Notaday"
    df.loc[0, "pickup_hour"] = 25
    df.loc[0, "fare_amount"] = 0

    report = validate_fare_prediction_training_data(df)

    assert any(
        "invalid weekday names" in error
        for error in report["errors"]
    )
    assert any(
        "invalid hours" in error
        for error in report["errors"]
    )
    assert any(
        "timestamp and pickup-hour mismatches" in error
        for error in report["errors"]
    )
    assert any(
        "fares are zero or negative" in error
        for error in report["errors"]
    )


# Demand training data

def test_demand_training_data_accepts_valid_single_row():
    report = validate_demand_training_data(make_demand_df())

    assert report["errors"] == []
    assert any(
        "could not be checked" in warning
        for warning in report["warnings"]
    )


def test_demand_training_data_detects_duplicate_zone_hour_pairs():
    df = pd.concat(
        [make_demand_df(), make_demand_df()],
        ignore_index=True,
    )

    report = validate_demand_training_data(df)

    assert any(
        "duplicate zone-hour pairs" in error
        for error in report["errors"]
    )
    assert any(
        "Lag consistency checks skipped" in warning
        for warning in report["warnings"]
    )


def test_demand_training_data_detects_wrong_one_hour_lag():
    times = pd.date_range(
        "2025-01-01",
        periods=170,
        freq="h",
    )
    counts = pd.Series(range(1, 171))

    df = pd.DataFrame({
        "pickup_datetime_hour": times,
        "PULocationID": [161] * len(times),
        "pickup_count": counts,
        "pickup_hour": times.hour,
        "day_of_week": times.dayofweek,
        "lag_1h": counts.shift(1),
        "lag_24h": counts.shift(24),
        "lag_168h": counts.shift(168),
    })

    df = df.iloc[168:].reset_index(drop=True)
    df.loc[1, "lag_1h"] = 999

    report = validate_demand_training_data(df)

    assert any(
        "lag_1h: 1 lag mismatches" in error
        for error in report["errors"]
    )


def test_demand_training_data_rejects_invalid_counts():
    df = make_demand_df()
    df.loc[0, "pickup_count"] = -1

    report = validate_demand_training_data(df)

    assert any(
        "pickup_count: 1 invalid counts" in error
        for error in report["errors"]
    )