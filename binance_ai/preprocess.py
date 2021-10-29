import datetime
import os
import time
from logging import getLogger
from pathlib import Path

import pandas as pd
from binance.spot import Spot
# from bitflyer_api import get_executions
from dateutil.relativedelta import relativedelta

from manage import EXECUTION_HISTORY_DIR, LOCAL, REF_LOCAL
from utils import df_to_csv, path_exists, read_csv, series_unix_to_tz

logger = getLogger(__name__)

if LOCAL:
    from dotenv import load_dotenv
    load_dotenv()

if not REF_LOCAL:
    from aws import S3
    s3 = S3()


def get_aggregate_trades_list(symbol, df_pre_latest_aggregate_trades, from_aggregate_tradeId=None, limit=1000, region='Asia/Tokyo'):
    logger.debug('取引履歴ダウンロード中...')
    aggregate_trades_list = []

    spot_client = Spot(key=os.environ['API_KEY'], secret=os.environ['API_SECRET'])
    if not from_aggregate_tradeId:
        from_aggregate_tradeId = df_pre_latest_aggregate_trades.iat[-1, 0]
    elif df_pre_latest_aggregate_trades.empty and not from_aggregate_tradeId:
        raise Exception('df_pre_latest_aggregate_trades and from_aggregate_tradeId are vacant.')

    start_timestamp = -1
    if not df_pre_latest_aggregate_trades.empty:
        start_timestamp = int(df_pre_latest_aggregate_trades.index[0].timestamp() * 1000)

    start_time = time.time()
    request_count = 0
    while True:
        new_aggregate_trades_list = spot_client.agg_trades(symbol=symbol, limit=limit, fromId=from_aggregate_tradeId)
        request_count += 1
        aggregate_trades_list += new_aggregate_trades_list
        if new_aggregate_trades_list[-1]['a'] - new_aggregate_trades_list[0]['a'] + 1 < limit:
            break
        elif start_timestamp == -1:
            start_timestamp = new_aggregate_trades_list[0]['T']
        elif new_aggregate_trades_list[-1]['T'] - start_timestamp > 259200000:
            break
        from_aggregate_tradeId += limit
        if request_count == 1000:
            process_time = time.time() - start_time
            if process_time < 60:
                time.sleep(60 - process_time)
            start_time = time.time()
            request_count = 0

    df_latest_aggregate_trades = pd.DataFrame(aggregate_trades_list)
    df_latest_aggregate_trades.columns = ['aggregate_tradeId', 'price', 'size', 'first_tradeId', 'last_tradeId', 'timestamp', 'maker', 'best_price_match']

    df_latest_aggregate_trades['timestamp'] = pd.to_datetime(df_latest_aggregate_trades['timestamp'], unit='ms', utc=True)
    df_latest_aggregate_trades['timestamp'] = df_latest_aggregate_trades['timestamp'].dt.tz_convert(region)
    df_latest_aggregate_trades = df_latest_aggregate_trades.set_index('timestamp', drop=True)

    df_latest_aggregate_trades = df_latest_aggregate_trades.astype({'price': float, 'size': float})

    df_latest_aggregate_trades = pd.concat([df_pre_latest_aggregate_trades, df_latest_aggregate_trades])
    return df_latest_aggregate_trades


def save_execution_history(
        product_code,
        df_latest_aggregate_trades,
        p_exe_history_dir,
        target_datetime_list):

    for target_datetime in target_datetime_list:
        p_exe_history_day_dir = p_exe_history_dir.joinpath(
            product_code,
            target_datetime.strftime('%Y'),
            target_datetime.strftime('%m'),
            target_datetime.strftime('%d'),
        )

        target_date_st = target_datetime.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        next_day = target_date_st + datetime.timedelta(days=1)

        target_date_ed = next_day.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        logger.debug(f'[{target_datetime}] データ抽出中...')
        df_target_date_trades = df_latest_aggregate_trades.query('@target_date_st <= index < @target_date_ed')
        logger.debug(f'[{target_datetime}] データ抽出完了')

        p_save_row_file = p_exe_history_day_dir.joinpath('row.csv')
        p_save_1h_file = p_exe_history_day_dir.joinpath('1h.csv')
        p_save_1m_file = p_exe_history_day_dir.joinpath('1m.csv')
        p_save_10m_file = p_exe_history_day_dir.joinpath('10m.csv')

        logger.debug(f'[{target_datetime}] リサンプリング中...')

        df_to_csv(p_save_row_file, df_target_date_trades, index=True)
        resampling(df_target_date_trades, p_save_1m_file, 'T')
        resampling(df_target_date_trades, p_save_10m_file, '10T')
        resampling(df_target_date_trades, p_save_1h_file, 'H')

        logger.debug(f'[{target_datetime}] リサンプリング完了')


def resampling(df, p_save_file='', freq='T'):
    df_price = df[['price']]
    df_size = df[['size']]
    df_price_ohlc = df_price.resample(freq).ohlc()
    df_price_ohlc.columns = [
        f'{col_name[1]}_{col_name[0]}' for col_name in df_price_ohlc.columns.tolist()]
    df_price_ohlc.interpolate()
    df_price_ohlc.dropna(how='any')

    df_size = df_size.resample(freq).sum()
    df_size.columns = ['total_size']
    df_resampled = pd.concat([df_price_ohlc, df_size], axis=1)
    if not p_save_file == '':
        df_to_csv(p_save_file, df_resampled, index=True)

    return df_resampled


def make_summary_from_scratch(p_file):
    logger.debug(f'[{p_file.parent}] 集計データ作成中...')
    p_summary_path = p_file.parent.joinpath('summary.csv')

    df = pd.DataFrame()

    if path_exists(p_file):
        df = read_csv(str(p_file))

    if df.empty:
        logger.debug(f'[{p_file}] データが存在しなかったため集計データ作成を中断します。')
        return pd.DataFrame()

    df_summary = pd.DataFrame({
        'open_price': [float(df['open_price'].values[0])],
        'high_price': [int(df['high_price'].max())],
        'low_price': [int(df['low_price'].min())],
        'close_price': [float(df['close_price'].values[-1])],
        'total_size': [float(df['total_size'].sum())]
    })

    df_to_csv(p_summary_path, df_summary, index=False)
    logger.debug(f'[{p_file.parent}] 集計データ作成完了')
    return df_summary


def make_summary_from_csv(
        product_code,
        p_dir='',
        summary_path_list=[],
        save=True):
    if p_dir != '':
        logger.debug(f'[{p_dir}] 集計データ更新中...')
        p_summary_save_path = p_dir.joinpath('summary.csv')
    else:
        if len(summary_path_list) == 0:
            logger.debug('対象となる集計データが存在しないため更新を終了します。')
            return
        elif len(summary_path_list) == 1:
            logger.debug(
                f'[{summary_path_list[0]}] 集計データ更新中...'
            )
        else:
            summary_path_list = sorted(summary_path_list)
            logger.debug(
                f'[{summary_path_list[0]} - {summary_path_list[-1]}] 集計データ更新中...'
            )
    df_summary = pd.DataFrame()
    if p_dir != '':
        if path_exists(p_summary_save_path):
            df_summary = read_csv(str(p_summary_save_path))
    if summary_path_list == [] and p_dir != '':
        if REF_LOCAL:
            p_summary_path_list = p_dir.glob('*/summary.csv')
            p_summary_path_list = sorted(p_summary_path_list)
            if len(p_summary_path_list) == 1:
                summary_path_list = [
                    str(p_summary_path_list[0])
                ]
            else:
                summary_path_list = [
                    str(p_summary_path_list[-2]), str(p_summary_path_list[-1])
                ]
        else:
            day_dir_list = s3.listdir(str(p_dir))
            if len(day_dir_list) <= 2:
                target_day_dir_list = day_dir_list
            else:
                target_day_dir_list = day_dir_list[-2:]
            for day_dir in target_day_dir_list:
                summary_path_tmp = day_dir + 'summary.csv'
                if path_exists(summary_path_tmp):
                    summary_path_list.append(summary_path_tmp)

    for summary_path in summary_path_list:
        if product_code not in summary_path:
            logger.warning(
                f'[{summary_path}] 対象のproduct_codeとは違うパスが含まれているため、読み込み対象外にします。')
            continue

        df_summary_child = read_csv(summary_path)
        if df_summary.empty:
            df_summary = df_summary_child.copy()
        else:
            if len(summary_path_list) == 1:
                df_summary.at[0, 'open_price'] = df_summary_child.at[0, 'open_price']

            if df_summary.at[0, 'high_price'] < df_summary_child.at[0, 'high_price']:
                df_summary.at[0, 'high_price'] = df_summary_child.at[0, 'high_price']

            if df_summary.at[0, 'low_price'] > df_summary_child.at[0, 'low_price']:
                df_summary.at[0, 'low_price'] = df_summary_child.at[0, 'low_price']

            df_summary['close_price'] = df_summary_child['close_price']

    if save and p_dir != '':
        df_to_csv(p_summary_save_path, df_summary, index=False)

    if p_dir != '':
        logger.debug(f'[{p_dir}] 集計データ更新完了')
    else:
        if len(summary_path_list) == 1:
            logger.debug(
                f'[{summary_path_list[0]}] 集計データ更新完了'
            )
        else:
            logger.debug(
                f'[{summary_path_list[0]} - {summary_path_list[-1]}] 集計データ更新完了'
            )
    return df_summary


def make_summary(product_code, p_dir, daily=False):
    if daily:
        p_1m_file = p_dir.joinpath('1m.csv')
        df_summary = make_summary_from_scratch(p_1m_file)
    else:
        df_summary = make_summary_from_csv(
            product_code=product_code,
            p_dir=p_dir,
            summary_path_list=[],
            save=True
        )
    return not df_summary.empty


def obtain_execution_history_from_scratch(product_code, from_aggregate_tradeId, limit=1000):
    p_exe_history_dir = Path(EXECUTION_HISTORY_DIR)

    df_trades_list = get_aggregate_trades_list(product_code, pd.DataFrame(), from_aggregate_tradeId=from_aggregate_tradeId, limit=limit)
    prev_trades_num = len(df_trades_list)

    while True:
        start_time = time.time()
        df_trades_list = get_aggregate_trades_list(product_code, df_trades_list, limit=limit)

        target_datetime = df_trades_list.index[0]
        target_datetime = target_datetime.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        ed_date = df_trades_list.index[-1].replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        ed_after_1d_datetime = ed_date + datetime.timedelta(days=1)

        target_datetime_list = []

        while target_datetime != ed_after_1d_datetime:
            target_datetime_list.append(target_datetime)
            target_datetime += datetime.timedelta(days=1)

        save_execution_history(product_code, df_trades_list, p_exe_history_dir, target_datetime_list)

        print(f"last trade datetime: {df_trades_list.index[-1]} tradeId: {df_trades_list.iat[-1,0]}")

        if len(df_trades_list) - prev_trades_num < limit:
            break

        df_trades_list = df_trades_list.query('@ed_date <= index')

        prev_trades_num = len(df_trades_list)
        process_time = time.time() - start_time
        time.sleep(60 - process_time % 60)

    logger.debug(f'[{product_code}] 集計データ作成中...')

    gen_execution_summaries(
        product_code,
        year=2021,
        month=-1,
        day=-1
    )

    logger.debug(f'[{product_code}] 集計データ作成完了')


def obtain_latest_summary(product_code, region='Asia/Tokyo'):
    logger.debug(f'[{product_code}] AI用集計データ取得中...')
    p_exe_history_dir = Path(EXECUTION_HISTORY_DIR)
    p_all_summary_file = p_exe_history_dir.joinpath(
        product_code, 'summary.csv')

    current_datetime = datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9)))

    # before_1h_datetime = current_datetime - datetime.timedelta(hours=1)
    before_6h_datetime = current_datetime - datetime.timedelta(hours=6)
    before_1d_datetime = current_datetime - datetime.timedelta(days=1)
    before_32d_datetime = current_datetime - datetime.timedelta(days=32)
    before_1m_datetime = current_datetime + relativedelta(months=-1)
    before_1y_datetime = current_datetime + relativedelta(years=-1)

    p_pre_latest_aggregate_trades_file = p_exe_history_dir.joinpath(
        product_code,
        current_datetime.strftime('%Y'),
        current_datetime.strftime('%m'),
        current_datetime.strftime('%d'),
        'row.csv'
    )

    target_datetime_list = [current_datetime]
    target_datetime = current_datetime
    while not path_exists(p_pre_latest_aggregate_trades_file):
        target_datetime -= datetime.timedelta(days=1)
        p_pre_latest_aggregate_trades_file = p_exe_history_dir.joinpath(
            product_code,
            target_datetime.strftime('%Y'),
            target_datetime.strftime('%m'),
            target_datetime.strftime('%d'),
            'row.csv'
        )
        target_datetime_list.append(target_datetime)

    df_pre_latest_aggregate_trades = read_csv(str(p_pre_latest_aggregate_trades_file))
    df_pre_latest_aggregate_trades['timestamp'] = series_unix_to_tz(df_pre_latest_aggregate_trades['timestamp'], unit=None, utc=None, region=region)
    df_pre_latest_aggregate_trades = df_pre_latest_aggregate_trades.set_index('timestamp', drop=True)
    df_pre_latest_aggregate_trades = df_pre_latest_aggregate_trades.astype({'price': float, 'size': float})

    logger.debug(f'[{product_code}] 最新データ取得中...')
    df_latest_aggregate_trades = get_aggregate_trades_list(product_code, df_pre_latest_aggregate_trades)
    logger.debug(f'[{product_code}] 最新データ取得完了')

    logger.debug(f'[{product_code}] 最新データ保存中...')
    save_execution_history(product_code, df_latest_aggregate_trades, p_exe_history_dir, target_datetime_list)
    logger.debug(f'[{product_code}] 最新データ保存完了')

    df_resampled_s = resampling(df_latest_aggregate_trades[['price', 'size']], freq='S')
    df_before_6h = df_resampled_s.query('index > @before_6h_datetime')
    df_before_1d = df_resampled_s.query('index > @before_1d_datetime')

    for target_datetime in target_datetime_list:
        gen_execution_summaries(
            product_code,
            year=target_datetime.strftime('%Y'),
            month=target_datetime.strftime('%m'),
            day=target_datetime.strftime('%d')
        )

    # monthly summary
    p_target_monthly_summary_path_list = [
        p_exe_history_dir.joinpath(
            product_code,
            current_datetime.strftime('%Y'),
            current_datetime.strftime('%m'),
            'summary.csv'
        ),
        p_exe_history_dir.joinpath(
            product_code,
            before_32d_datetime.strftime('%Y'),
            before_32d_datetime.strftime('%m'),
            'summary.csv'
        )
    ]

    target_monthly_summary_path_list = []
    for p_target_monthly_summary_path in p_target_monthly_summary_path_list:
        if path_exists(p_target_monthly_summary_path):
            target_monthly_summary_path_list.append(
                str(p_target_monthly_summary_path)
            )
    df_monthly_summary = make_summary_from_csv(
        product_code=product_code,
        p_dir='',
        summary_path_list=target_monthly_summary_path_list,
        save=False
    )

    # weekly summary
    target_weekly_summary_path_list = []
    for i in range(8):
        if i == 0:
            target_datetime = current_datetime
        else:
            target_datetime = current_datetime - datetime.timedelta(days=i)
        p_target_weekly_summary_path = p_exe_history_dir.joinpath(
            product_code,
            target_datetime.strftime('%Y'),
            target_datetime.strftime('%m'),
            target_datetime.strftime('%d'),
            'summary.csv'
        )
        if path_exists(p_target_weekly_summary_path):
            target_weekly_summary_path_list.append(
                str(p_target_weekly_summary_path)
            )
    df_weekly_summary = make_summary_from_csv(
        product_code=product_code,
        p_dir='',
        summary_path_list=target_weekly_summary_path_list,
        save=False
    )

    # yearly summary
    before_365d_datetime = current_datetime - datetime.timedelta(days=365)
    p_target_yearly_summary_path_list = [
        p_exe_history_dir.joinpath(
            product_code,
            current_datetime.strftime('%Y'),
            current_datetime.strftime('%m'),
            current_datetime.strftime('%d'),
            'summary.csv'
        ),
        p_exe_history_dir.joinpath(
            product_code,
            before_365d_datetime.strftime('%Y'),
            before_365d_datetime.strftime('%m'),
            before_365d_datetime.strftime('%d'),
            'summary.csv'
        )
    ]
    target_yearly_summary_path_list = []
    for p_target_yearly_summary_path in p_target_yearly_summary_path_list:
        if path_exists(p_target_yearly_summary_path):
            target_yearly_summary_path_list.append(
                str(p_target_yearly_summary_path)
            )

    df_yearly_summary = make_summary_from_csv(
        product_code=product_code,
        p_dir='',
        summary_path_list=target_yearly_summary_path_list,
        save=False
    )

    df_all_summary = read_csv(str(p_all_summary_file))

    latest_summary = {
        'now': {
            'price': df_before_1d['close_price'].values[-1],
        },
        '6h': {
            'price': {
                'open': df_before_6h['open_price'].values[0],
                'high': df_before_6h['high_price'].max(),
                'low': df_before_6h['low_price'].min(),
                'close': df_before_6h['close_price'].values[-1],
            },
            'trend': 'DOWN',
        },
        '1d': {
            'price': {
                'open': df_before_1d['open_price'].values[0],
                'high': df_before_1d['high_price'].max(),
                'low': df_before_1d['low_price'].min(),
                'close': df_before_1d['close_price'].values[-1],
            },
            'trend': 'DOWN',
        },
        '1w': {
            'price': {
                'open': df_weekly_summary.at[0, 'open_price'],
                'high': df_weekly_summary.at[0, 'high_price'],
                'low': df_weekly_summary.at[0, 'low_price'],
                'close': df_weekly_summary.at[0, 'close_price'],
            },
            'trend': 'DOWN',
        },
        '1m': {
            'price': {
                'open': df_monthly_summary.at[0, 'open_price'],
                'high': df_monthly_summary.at[0, 'high_price'],
                'low': df_monthly_summary.at[0, 'low_price'],
                'close': df_monthly_summary.at[0, 'close_price'],
            },
            'trend': 'DOWN',
        },
        '1y': {
            'price': {
                'open': df_yearly_summary.at[0, 'open_price'],
                'high': df_yearly_summary.at[0, 'high_price'],
                'low': df_yearly_summary.at[0, 'low_price'],
                'close': df_yearly_summary.at[0, 'close_price'],
            },
            'trend': 'DOWN',
        },
        'all': {
            'price': {
                'open': df_all_summary.at[0, 'open_price'],
                'high': df_all_summary.at[0, 'high_price'],
                'low': df_all_summary.at[0, 'low_price'],
                'close': df_all_summary.at[0, 'close_price'],
            },
            'trend': 'DOWN',
        },

    }

    # load summaries
    p_yesterday_summary_path = p_exe_history_dir.joinpath(
        product_code,
        before_1d_datetime.strftime('%Y'),
        before_1d_datetime.strftime('%m'),
        before_1d_datetime.strftime('%d'),
        'summary.csv'
    )
    p_last_month_summary_path = p_exe_history_dir.joinpath(
        product_code,
        before_1m_datetime.strftime('%Y'),
        before_1m_datetime.strftime('%m'),
        'summary.csv'
    )
    p_last_year_summary_path = p_exe_history_dir.joinpath(
        product_code,
        before_1y_datetime.strftime('%Y'),
        'summary.csv'
    )
    df_yesterday_summary = pd.DataFrame()
    df_last_month_summary = pd.DataFrame()
    df_last_year_summary = pd.DataFrame()

    if path_exists(p_yesterday_summary_path):
        df_yesterday_summary = read_csv(str(p_yesterday_summary_path))

    if path_exists(p_last_month_summary_path):
        df_last_month_summary = read_csv(str(p_last_month_summary_path))

    if path_exists(p_last_year_summary_path):
        df_last_year_summary = read_csv(str(p_last_year_summary_path))

    if not df_yesterday_summary.empty:
        latest_summary['yesterday'] = {
            'open': df_yesterday_summary.at[0, 'open_price'],
            'high': df_yesterday_summary.at[0, 'high_price'],
            'low': df_yesterday_summary.at[0, 'low_price'],
            'close': df_yesterday_summary.at[0, 'close_price'],
        }

    if not df_last_month_summary.empty:
        latest_summary['last_month'] = {
            'open': df_last_month_summary.at[0, 'open_price'],
            'high': df_last_month_summary.at[0, 'high_price'],
            'low': df_last_month_summary.at[0, 'low_price'],
            'close': df_last_month_summary.at[0, 'close_price'],
        }

    if not df_last_year_summary.empty:
        latest_summary['last_year'] = {
            'open': df_last_year_summary.at[0, 'open_price'],
            'high': df_last_year_summary.at[0, 'high_price'],
            'low': df_last_year_summary.at[0, 'low_price'],
            'close': df_last_year_summary.at[0, 'close_price'],
        }

    logger.debug(f'[{product_code}] AI用集計データ取得完了')

    return latest_summary


def gen_execution_summaries(product_code, year=2021, month=-1, day=-1):
    logger.debug(f'[{product_code} {year} {month} {day}] 集計データ作成開始')
    p_save_base_dir = Path(EXECUTION_HISTORY_DIR)
    p_product_dir = p_save_base_dir.joinpath(product_code)
    p_year_dir = p_product_dir.joinpath(str(year))
    if month == -1:
        if REF_LOCAL:
            for p_target_month_dir in p_year_dir.glob('*'):
                if not p_target_month_dir.is_dir():
                    continue
                p_month_dir = p_year_dir.joinpath(str(p_target_month_dir.name))
                if day == -1:
                    for p_target_day_dir in p_month_dir.glob('*'):
                        if p_target_day_dir.is_dir():
                            p_day_dir = p_month_dir.joinpath(
                                str(p_target_day_dir.name))
                            success = make_summary(product_code, p_day_dir, daily=True)
                            if not success:
                                logger.debug(f'[{p_day_dir}] データが存在しないため、集計を作成できませんでした。')
                                return
                else:
                    p_day_dir = p_month_dir.joinpath(str(day))
                    success = make_summary(product_code, p_day_dir, daily=True)
                    if not success:
                        logger.debug(f'[{p_day_dir}] データが存在しないため、集計を作成できませんでした。')
                        return
                make_summary(product_code, p_month_dir)
        else:
            target_month_dir_list = s3.listdir(str(p_year_dir))
            for target_month_dir in target_month_dir_list:
                if target_month_dir.endswith('summary.csv'):
                    continue
                dir_list = target_month_dir.split('/')
                dir_list.remove('')
                p_month_dir = p_year_dir.joinpath(
                    dir_list[-1]
                )
                if day == -1:
                    target_day_dir_list = s3.listdir(str(p_month_dir))
                    for target_day_dir in target_day_dir_list:
                        if not target_day_dir.endswith('summary.csv'):
                            dir_list = target_day_dir.split('/')
                            dir_list.remove('')
                            p_day_dir = p_month_dir.joinpath(
                                dir_list[-1]
                            )
                            success = make_summary(product_code, p_day_dir, daily=True)
                            if not success:
                                logger.debug(f'[{p_day_dir}] データが存在しないため、集計を作成できませんでした。')
                                return
                else:
                    p_day_dir = p_month_dir.joinpath(str(day))
                    success = make_summary(product_code, p_day_dir, daily=True)
                    if not success:
                        logger.debug(f'[{p_day_dir}] データが存在しないため、集計を作成できませんでした。')
                        return
                make_summary(product_code, p_month_dir)

    else:
        p_month_dir = p_year_dir.joinpath(str(month))
        if day == -1:
            if REF_LOCAL:
                for p_target_day_dir in p_month_dir.glob('*'):
                    if p_target_day_dir.is_dir():
                        p_day_dir = p_month_dir.joinpath(
                            str(p_target_day_dir.name)
                        )
                        success = make_summary(product_code, p_day_dir, daily=True)
                        if not success:
                            logger.debug(f'[{p_day_dir}] データが存在しないため、集計を作成できませんでした。')
                            return
            else:
                target_day_dir_list = s3.listdir(str(p_month_dir))
                for target_day_dir in target_day_dir_list:
                    if not target_day_dir.endswith('summary.csv'):
                        p_day_dir = p_month_dir.joinpath(
                            target_day_dir.split('/')[-2]
                        )
                        success = make_summary(product_code, p_day_dir, daily=True)
                        if not success:
                            logger.debug(f'[{p_day_dir}] データが存在しないため、集計を作成できませんでした。')
                            return
        else:
            p_day_dir = p_month_dir.joinpath(str(day))
            success = make_summary(product_code, p_day_dir, daily=True)
            if not success:
                logger.debug(f'[{p_day_dir}] データが存在しないため、集計を作成できませんでした。')
                return
        success = make_summary(product_code, p_month_dir)
    success = make_summary(product_code, p_year_dir)
    success = make_summary(product_code, p_product_dir)

    logger.debug(f'[{product_code} {year} {month} {day}] 集計データ作成終了')


def delete_row_data(product_code, current_datetime, days):
    before_7d_datetime = current_datetime - datetime.timedelta(days=days)
    p_dir = Path(EXECUTION_HISTORY_DIR)
    p_row_file = p_dir.joinpath(
        product_code,
        before_7d_datetime.strftime('%Y'),
        before_7d_datetime.strftime('%m'),
        before_7d_datetime.strftime('%d'),
        'row.csv'
    )

    if REF_LOCAL:
        p_row_file.unlink(missing_ok=True)
    else:
        s3.delete_file(str(p_row_file))
