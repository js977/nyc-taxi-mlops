import argparse
from nyc_taxi.data.audit import audit_dataset
import pandas as pd
import json
from pathlib import Path

def main():

    parser=argparse.ArgumentParser(
        description="Audit a NYC Taxi dataset"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the csv dataset"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_audit.json")
    )

    args=parser.parse_args()

    df=pd.read_csv(args.input)

    report=audit_dataset(df)

    args.output.parent.mkdir(

        parents=True,
        exist_ok=True
    )

    #args.output.write_text(
    #    json.dumps(report,indent=2)
    #    encoding="utf-8"
    #)
    args.output.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

    print(f"Audit competed:{args.output}")

if __name__=="__main__":
    main()
