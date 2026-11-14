import pandas as pd


FARE_REQUIRED_COLUMNS = [
    "fare_amount",
    "PULocationID",
    "DOLocationID",
    "pickup_hour",
    "day_of_week",
    "tpep_pickup_datetime",
]

DEMAND_REQUIRED_COLUMNS = [
    "PULocationID",
    "tpep_pickup_datetime",
]

DEMAND_TRAINING_REQUIRED_COLUMNS = [
    "pickup_datetime_hour",
    "PULocationID",
    "pickup_count",
    "pickup_hour",
    "day_of_week",
    "lag_1h",
    "lag_24h",
    "lag_168h",
]


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> list[str]:

    missing_columns = []

    for column in required_columns:
        if column not in df.columns:
            missing_columns.append(column)

    return missing_columns


def validate_missing_values(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, int]:

    result = {}

    for column in columns:
        result[column] = int(df[column].isna().sum())

    return result


def validate_duplicate_rows(df: pd.DataFrame) -> int:

    return int(df.duplicated().sum())


def validate_numeric_value(
    df: pd.DataFrame,
    column: str,
    positive_only: bool = False,
) -> int:

    values = pd.to_numeric(df[column], errors="coerce")

    invalid = (
        values.isna()
        | values.isin([float("inf"), -float("inf")])
    )

    if positive_only:
        invalid = invalid | (values <= 0)

    # Location IDs must be positive integers.
    if column in ["PULocationID", "DOLocationID"]:
        invalid = invalid | (values <= 0) | (values % 1 != 0)

    return int(invalid.sum())


def validate_timestamps(
    df: pd.DataFrame,
    column: str,
) -> int:

    timestamps = pd.to_datetime(df[column], errors="coerce")

    return int(timestamps.isna().sum())


def str_validate_day_week(df: pd.DataFrame) -> int:

    weekdays = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    invalid = ~df["day_of_week"].isin(weekdays)

    return int(invalid.sum())


def number_validate_day_week(df: pd.DataFrame) -> int:

    days = pd.to_numeric(df["day_of_week"], errors="coerce")

    invalid = (
        days.isna()
        | ~days.between(0, 6)
        | (days % 1 != 0)
    )

    return int(invalid.sum())


def validate_hours(df: pd.DataFrame) -> int:

    hours = pd.to_numeric(df["pickup_hour"], errors="coerce")

    invalid = (
        hours.isna()
        | ~hours.between(0, 23)
        | (hours % 1 != 0)
    )

    return int(invalid.sum())


def compare_timestamp_pick_hour(
    df: pd.DataFrame,
    column_time: str,
    column_hour: str,
) -> int:

    timestamps = pd.to_datetime(df[column_time], errors="coerce")
    hours = pd.to_numeric(df[column_hour], errors="coerce")

    comparable = timestamps.notna() & hours.notna()

    invalid = comparable & (timestamps.dt.hour != hours)

    return int(invalid.sum())


def validate_nonnegative_counts(
    df: pd.DataFrame,
    column: str,
) -> int:

    values = pd.to_numeric(df[column], errors="coerce")

    invalid = (
        values.isna()
        | values.isin([float("inf"), -float("inf")])
        | (values < 0)
        | (values % 1 != 0)
    )

    return int(invalid.sum())


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
) -> dict[str, list[str]]:

    errors = []
    warnings = []

    if df.empty:
        errors.append("Dataset is empty!")
        return {"errors": errors, "warnings": warnings}

    missing_columns = validate_required_columns(df, required_columns)

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
        return {"errors": errors, "warnings": warnings}

    duplicate_rows = validate_duplicate_rows(df)

    if duplicate_rows > 0:
        warnings.append(f"Found {duplicate_rows} duplicate rows.")

    missing_values = validate_missing_values(df, required_columns)

    for column, count in missing_values.items():
        if count > 0:
            errors.append(f"{column}: {count} missing values.")

    numeric_columns = [
        "fare_amount",
        "PULocationID",
        "DOLocationID",
        "pickup_hour",
        "pickup_count",
        "lag_1h",
        "lag_24h",
        "lag_168h",
    ]

    for column in numeric_columns:
        if column in required_columns:
            invalid_count = validate_numeric_value(df, column)

            if invalid_count > 0:
                errors.append(
                    f"{column}: {invalid_count} invalid numeric values."
                )

    timestamp_columns = [
        "tpep_pickup_datetime",
        "pickup_datetime_hour",
    ]

    for column in timestamp_columns:
        if column in required_columns:
            invalid_count = validate_timestamps(df, column)

            if invalid_count > 0:
                errors.append(
                    f"{column}: {invalid_count} invalid timestamps."
                )

    return {"errors": errors, "warnings": warnings}


def validate_fare_prediction_training_data(
    df: pd.DataFrame,
) -> dict[str, list[str]]:

    report = validate_dataframe(df, FARE_REQUIRED_COLUMNS)

    if df.empty or validate_required_columns(df, FARE_REQUIRED_COLUMNS):
        return report

    errors = report["errors"]

    invalid_days = str_validate_day_week(df)

    if invalid_days > 0:
        errors.append(
            f"day_of_week: {invalid_days} invalid weekday names."
        )

    invalid_hours = validate_hours(df)

    if invalid_hours > 0:
        errors.append(
            f"pickup_hour: {invalid_hours} invalid hours."
        )

    mismatches = compare_timestamp_pick_hour(
        df,
        "tpep_pickup_datetime",
        "pickup_hour",
    )

    if mismatches > 0:
        errors.append(
            f"{mismatches} timestamp and pickup-hour mismatches."
        )

    # Training policy: accept only strictly positive fares.
    fares = pd.to_numeric(df["fare_amount"], errors="coerce")
    nonpositive_fares = int((fares <= 0).sum())

    if nonpositive_fares > 0:
        errors.append(
            f"fare_amount: {nonpositive_fares} fares are zero or negative."
        )

    return report


def validate_demand_training_data(
    df: pd.DataFrame,
) -> dict[str, list[str]]:

    report = validate_dataframe(
        df,
        DEMAND_TRAINING_REQUIRED_COLUMNS,
    )

    if df.empty or validate_required_columns(
        df,
        DEMAND_TRAINING_REQUIRED_COLUMNS,
    ):
        return report

    errors = report["errors"]
    warnings = report["warnings"]

    invalid_days = number_validate_day_week(df)

    if invalid_days > 0:
        errors.append(
            f"day_of_week: {invalid_days} invalid numeric weekdays."
        )

    invalid_hours = validate_hours(df)

    if invalid_hours > 0:
        errors.append(
            f"pickup_hour: {invalid_hours} invalid hours."
        )

    mismatches = compare_timestamp_pick_hour(
        df,
        "pickup_datetime_hour",
        "pickup_hour",
    )

    if mismatches > 0:
        errors.append(
            f"{mismatches} timestamp and pickup-hour mismatches."
        )

    count_columns = [
        "pickup_count",
        "lag_1h",
        "lag_24h",
        "lag_168h",
    ]

    invalid_counts = {}

    for column in count_columns:
        invalid_count = validate_nonnegative_counts(df, column)
        invalid_counts[column] = invalid_count

        if invalid_count > 0:
            errors.append(
                f"{column}: {invalid_count} invalid counts."
            )

    # Use a copy so validation does not modify the input.
    ordered = df.copy()

    ordered["pickup_datetime_hour"] = pd.to_datetime(
        ordered["pickup_datetime_hour"],
        errors="coerce",
    )

    ordered["PULocationID"] = pd.to_numeric(
        ordered["PULocationID"],
        errors="coerce",
    )

    duplicate_zone_hours = int(
        ordered.duplicated(
            subset=["PULocationID", "pickup_datetime_hour"]
        ).sum()
    )

    if duplicate_zone_hours > 0:
        errors.append(
            f"Found {duplicate_zone_hours} duplicate zone-hour pairs."
        )

    invalid_zones = validate_numeric_value(df, "PULocationID")

    if (
        ordered["pickup_datetime_hour"].isna().any()
        or invalid_zones > 0
        or duplicate_zone_hours > 0
        or invalid_counts["pickup_count"] > 0
    ):
        warnings.append(
            "Lag consistency checks skipped because timestamps, "
            "zones, pickup counts, or zone-hour uniqueness are invalid."
        )
        return report

    ordered["pickup_count"] = pd.to_numeric(
        ordered["pickup_count"],
        errors="coerce",
    )

    ordered = ordered.sort_values(
        ["PULocationID", "pickup_datetime_hour"]
    )

    grouped = ordered.groupby("PULocationID")

    lag_columns = {
        1: "lag_1h",
        24: "lag_24h",
        168: "lag_168h",
    }

    for hours, lag_column in lag_columns.items():

        expected_lag = grouped["pickup_count"].shift(hours)

        previous_time = grouped["pickup_datetime_hour"].shift(hours)

        actual_lag = pd.to_numeric(
            ordered[lag_column],
            errors="coerce",
        )

        correct_time_difference = (
            ordered["pickup_datetime_hour"] - previous_time
            == pd.Timedelta(hours=hours)
        )

        comparable = (
            expected_lag.notna()
            & actual_lag.notna()
            & correct_time_difference
        )

        mismatches = comparable & (expected_lag != actual_lag)

        mismatch_count = int(mismatches.sum())

        if mismatch_count > 0:
            errors.append(
                f"{lag_column}: {mismatch_count} lag mismatches."
            )

        skipped_count = int((~comparable).sum())

        if skipped_count > 0:
            warnings.append(
                f"{lag_column}: {skipped_count} rows could not be "
                "checked because suitable history or a lag value "
                "was unavailable."
            )

    return report