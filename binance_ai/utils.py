import pandas as pd

from manage import REF_LOCAL

if not REF_LOCAL:
    from aws import S3
    s3 = S3()


def path_exists(p_path):
    if REF_LOCAL:
        return p_path.exists()
    else:
        return s3.key_exists(str(p_path))


def read_csv(p_path):
    if REF_LOCAL:
        return pd.read_csv(p_path)
    else:
        return s3.read_csv(p_path)


def df_to_csv(p_save_file, df, index=True):
    if REF_LOCAL:
        p_save_file.parent.mkdir(exist_ok=True, parents=True)
        return df.to_csv(p_save_file, index=index)
    else:
        return s3.to_csv(str(p_save_file), df, index=index)


def series_unix_to_tz(series, unit=None, utc=None, convert=True, region='Asia/Tokyo'):
    series = pd.to_datetime(series, unit=unit, utc=utc)
    if convert:
        series = series.dt.tz_convert(region)
    else:
        series = series.dt.tz_localize(region)
    return series
