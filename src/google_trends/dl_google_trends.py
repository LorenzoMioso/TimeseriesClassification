import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from pytrends.request import TrendReq
from tqdm import tqdm


def get_google_trends(
    ft_list,
    least_recent_date,
    most_recent_date,
    geo,
    hl,
    category=-1,
    gprop="",
    N=2,
    sleep_time=30,
):
    results = pd.DataFrame()

    # Create timeframe for information download
    min_date = datetime.strptime(least_recent_date, "%Y-%m-%d %H:%M:%S.%f").strftime(
        "%Y-%m-%d"
    )
    max_date = datetime.strptime(most_recent_date, "%Y-%m-%d %H:%M:%S.%f").strftime(
        "%Y-%m-%d"
    )
    gtrends_timeframe = min_date + " " + max_date

    # Create google trends request
    pytrends = TrendReq(hl=hl, tz=360, timeout=50, retries=2)

    # Download the trend for the past five years of all colors from Google Trends
    # We download each trend N times in order to then create a mean signal
    print("Downloading Google Trends...")
    p_bar = tqdm(total=len(ft_list))

    for val in ft_list:
        feature_trends = []
        for _ in range(N):
            # Wait a bit before downloading the next trend
            time.sleep(sleep_time)

            # Searching for all categories or in a particular one
            if category == -1:
                pytrends.build_payload(
                    kw_list=[val],  # add search keyword
                    gprop=gprop,
                    geo=geo,
                    timeframe=gtrends_timeframe,
                )
            else:
                pytrends.build_payload(
                    kw_list=[val],
                    cat=category,  # filter by category
                    gprop=gprop,
                    geo=geo,
                    timeframe=gtrends_timeframe,
                )

            data = pytrends.interest_over_time()

            if not data.empty:
                data.drop(labels=["isPartial"], axis="columns", inplace=True)
                feature_trends.append(data[val])

        # Append average of series as the final trend
        results[val] = pd.DataFrame(feature_trends).T.mean(axis=1)
        p_bar.update()
    p_bar.close()
    results = results.resample(
        "W-MON"
    ).sum()  # Resample to the same frequency as our sales data
    return results


def run(args):
    # Load data
    train_df = pd.read_csv("visuelle/train.csv", parse_dates=[args.parse_dates_col])
    test_df = pd.read_csv("visuelle/test.csv", parse_dates=[args.parse_dates_col])
    # data_df = train_df.append(test_df)
    data_df = pd.concat([train_df, test_df])

    # Download data from Gtrends and save them to a pandas df
    least_recent_date = data_df[args.parse_dates_col].min()  # 2015-10-05
    most_recent_date = data_df[args.parse_dates_col].max()  # 2019-12-16

    # print(least_recent_date)
    # Or more hard coded (if you prefer)
    # least_recent_date, most_recent_date = pd.Timestamp('2015-10-05'), pd.Timestamp('2019-12-16')
    # features = data_df[args.feature].unique().tolist()
    # features = args.feature
    # print(features)

    with open(args.topics_file) as topics_file:
        topics_list = [line.strip() for line in topics_file]

    gtrends_series = get_google_trends(
        topics_list, args.start_date, args.end_date, geo=args.geo, hl=args.hl
    )

    # Save to a csv file
    save_dir = "downloaded_trends/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = "gtrend_"
    save_file += "_".join(topics_list)

    gtrends_series.to_csv(Path(save_dir + save_file + ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Google Trends.")
    parser.add_argument("--gtrend_save_path", type=str, default="gtrends.csv")
    parser.add_argument("--geo", type=str, default="")
    parser.add_argument("--hl", type=str, default="en-US")
    parser.add_argument("--feature", type=str, default="color")
    parser.add_argument("--index_col_nr", type=int, default=0)
    parser.add_argument("--parse_dates_col", type=str, default="release_date")
    parser.add_argument("--topics_file", type=str, default="topics.txt")
    parser.add_argument(
        "--start_date",
        type=str,
        default=str(datetime.today() - relativedelta(years=5)),
    )
    parser.add_argument("--end_date", type=str, default=str(datetime.today()))

    args = parser.parse_args()
    run(args)
