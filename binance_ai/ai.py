import datetime
import os
import time
from logging import getLogger
from pathlib import Path

import pandas as pd
from binance.spot import Spot

from manage import CHILD_ORDERS_DIR, LOCAL, REF_LOCAL
from utils import df_to_csv, path_exists, read_csv, rm_file, series_unix_to_tz

if LOCAL:
    from dotenv import load_dotenv
    load_dotenv()
if not REF_LOCAL:
    from aws import S3
    s3 = S3()

logger = getLogger(__name__)


class AI:
    """自動売買システムのアルゴリズム

    """

    def __init__(self, product_code, exchange_info, latest_summary, region='Asia/Tokyo'):

        self.spot_client = Spot(key=os.environ['API_KEY'], secret=os.environ['API_SECRET'])

        self.product_code = product_code
        self.size_round_digits = int(os.environ.get(f'{product_code}_SIZE_ROUND_DIGITS', 0))
        self.region = region
        self.exchange_info = exchange_info

        account_info = self.spot_client.account()
        self.commission_rate = account_info['makerCommission'] / 10000

        self.free_balances = {}
        for balance in account_info['balances']:
            self.free_balances.update({balance['asset']: float(balance['free'])})

        self.price_round_digits = int(os.environ[f"{self.exchange_info['symbols'][0]['quoteAsset']}_PRICE_ROUND_DIGITS"])

        p_child_orders_dir = Path(CHILD_ORDERS_DIR)
        p_child_orders_dir = p_child_orders_dir.joinpath(self.product_code)
        self.p_child_orders_path = {
            'long': p_child_orders_dir.joinpath('long_term.csv'),
            'short': p_child_orders_dir.joinpath('short_term.csv')
        }
        self.child_orders = {
            'long': pd.DataFrame(),
            'short': pd.DataFrame()
        }

        self.latest_summary = latest_summary

        for term in ['long', 'short']:
            if path_exists(self.p_child_orders_path[term]):
                self.child_orders[term] = read_csv(
                    str(self.p_child_orders_path[term])
                )
                if len(self.child_orders[term]) == 0:
                    self.child_orders[term] = pd.DataFrame()
                else:
                    self.child_orders[term] = self.child_orders[term].set_index(
                        'orderId',
                        drop=True,
                    )
                    self.child_orders[term]['time'] = series_unix_to_tz(self.child_orders[term]['time'], unit=None, utc=None, region=region)

        self.datetime_references = {
            'now': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))),
        }
        self.datetime_references['hourly'] = self.datetime_references['now'] - datetime.timedelta(hours=6)

        self.datetime_references['daily'] = self.datetime_references['now'] - datetime.timedelta(days=1)

        self.datetime_references['weekly'] = self.datetime_references['now'] - datetime.timedelta(days=7)

        self.datetime_references['monthly'] = self.datetime_references['now'] - datetime.timedelta(days=31)

        self.min_volume = {
            'long': float(os.environ.get(f'{product_code}_LONG_MIN_VOLUME', 10)),
            'short': float(os.environ.get(f'{product_code}_SHORT_MIN_VOLUME', 10)),
        }

        self.max_volume = {
            'long': float(os.environ.get(f'{product_code}_LONG_MAX_VOLUME', 100)),
            'short': float(os.environ.get(f'{product_code}_SHORT_MAX_VOLUME', 100)),
        }

        self.cut_loss_rate = {
            'long': float(os.environ.get(f'{product_code}_CUT_LOSS_RATE_LONG', 0.95)),
            'short': float(os.environ.get(f'{product_code}_CUT_LOSS_RATE_SHORT', 0.95)),
        }

        self.max_buy_prices_rate = {
            'long': float(os.environ.get('MAX_BUY_PRICE_RATE_IN_LONG')),
            'short': float(os.environ.get('MAX_BUY_PRICE_RATE_IN_SHORT')),
        }

    def _delte_order(self, term, order_id):
        self.child_orders[term].drop(
            index=[order_id],
            inplace=True
        )
        # csvファイルを更新
        if len(self.child_orders[term]) == 0:
            rm_file(self.p_child_orders_path[term])
        else:
            df_to_csv(self.p_child_orders_path[term], self.child_orders[term], index=True)
        logger.debug(f'{str(self.p_child_orders_path[term])} が更新されました。')

    def load_latest_child_orders(self,
                                 term,
                                 child_order_cycle,
                                 order_id,
                                 related_order_id='no_id'):
        logger.debug(f'order_id: {order_id}')
        # get a child order from api
        df_child_orders_tmp = pd.DataFrame()
        start_time = time.time()
        while df_child_orders_tmp.empty:
            response = self.spot_client.get_order(symbol=self.product_code, orderId=order_id)

            df_child_orders_tmp = pd.DataFrame([response])
            df_child_orders_tmp = df_child_orders_tmp.rename(columns={'origQty': 'size'})

            df_child_orders_tmp = df_child_orders_tmp.astype({
                'symbol': str,
                'orderId': int,
                'clientOrderId': str,
                'orderListId': int,
                'price': float,
                'size': float,
                'executedQty': float,
                'cummulativeQuoteQty': float,
                'status': str,
                'timeInForce': str,
                'type': str,
                'side': str,
                'stopPrice': float,
                'icebergQty': float,
                'isWorking': bool,
                'origQuoteOrderQty': float
            })

            df_child_orders_tmp['time'] = series_unix_to_tz(df_child_orders_tmp['time'], unit='ms', utc=True, region=self.region)
            df_child_orders_tmp['updateTime'] = series_unix_to_tz(df_child_orders_tmp['updateTime'], unit='ms', utc=True, region=self.region)

            if time.time() - start_time > 5:
                logger.warning(f'{order_id} はすでに存在しないため、ファイルから削除します。')
                self._delte_order(
                    term=term,
                    order_id=order_id
                )
                return

        df_child_orders_tmp['orderCycle'] = child_order_cycle
        df_child_orders_tmp['relatedOrderId'] = related_order_id
        df_child_orders_tmp['profit'] = 0
        df_child_orders_tmp['volume'] = df_child_orders_tmp['price'] * df_child_orders_tmp['size']
        df_child_orders_tmp['commission'] = df_child_orders_tmp['volume'] * self.commission_rate

        df_child_orders_tmp = df_child_orders_tmp.set_index('orderId', drop=True)

        if self.child_orders[term].empty:
            self.child_orders[term] = df_child_orders_tmp
        else:
            self.child_orders[term].loc[order_id] = df_child_orders_tmp.loc[order_id]

        if self.child_orders[term].at[order_id, 'status'] == 'FILLED':
            if self.child_orders[term].at[order_id, 'relatedOrderId'] == 'no_id' \
                    or self.child_orders[term].at[order_id, 'side'] == 'SELL':
                logger.info(
                    f'[{self.product_code} {term} {child_order_cycle} {self.child_orders[term].at[order_id, "side"]}  {order_id}] 約定しました!'
                )

            if self.child_orders[term].at[order_id, 'side'] == 'SELL':
                sell_price = self.child_orders[term].at[order_id, 'price']
                sell_size = self.child_orders[term].at[order_id, 'size']
                sell_commission = self.child_orders[term].at[order_id, 'commission']

                buy_price = self.child_orders[term].at[related_order_id, 'price']
                buy_size = self.child_orders[term].at[related_order_id, 'size']
                buy_commission = self.child_orders[term].at[related_order_id, 'commission']

                profit = sell_price * sell_size - buy_price * buy_size
                profit -= sell_commission + buy_commission

                logger.info(f'[{self.product_code} {term} {child_order_cycle}] {profit}円の利益が発生しました。')

                self.child_orders[term].at[order_id, 'profit'] = profit
                self.child_orders[term]['cumsumProfit'] = self.child_orders[term]['profit'].cumsum()

        # csvファイルを更新
        df_to_csv(self.p_child_orders_path[term], self.child_orders[term], index=True)
        logger.debug(f'{str(self.p_child_orders_path[term])} が更新されました。')

    def update_child_orders(self,
                            term,
                            order_id="",
                            child_order_cycle="",
                            related_order_id="no_id"):

        # --------------------------------
        # 既存の注文における約定状態を更新
        # --------------------------------
        for child_order_acceptance_id_tmp in self.child_orders[term].index.tolist():
            if self.child_orders[term].at[child_order_acceptance_id_tmp,
                                          'status'] == 'NEW':
                self.load_latest_child_orders(
                    term=term,
                    child_order_cycle=self.child_orders[term].at[child_order_acceptance_id_tmp,
                                                                 'orderCycle'],
                    order_id=child_order_acceptance_id_tmp,
                    related_order_id=self.child_orders[term].at[child_order_acceptance_id_tmp,
                                                                'relatedOrderId']
                )
        # --------------------------------
        # related_order_idを指定して、注文情報を更新
        # --------------------------------
        if not order_id == "":
            if child_order_cycle == "":
                raise ValueError("child_order_cycle must be setted")
            self.load_latest_child_orders(
                term=term,
                child_order_cycle=child_order_cycle,
                order_id=order_id,
                related_order_id=related_order_id
            )

    def _cancel(self,
                term,
                child_order_cycle,
                order_id,
                child_order_type='buy'):
        # ----------------------------------------------------------------
        # キャンセル処理
        # ----------------------------------------------------------------
        # cancel
        response = self.spot_client.cancel_order(symbol=self.product_code, orderId=order_id)

        if response['status'] == 'CANCELED':
            self._delte_order(term, order_id)
            print('================================================================')
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle}  {child_order_type} {order_id}] のキャンセルに成功しました。'
            )
            print('================================================================')

    def _buy(self, term, child_order_cycle, local_prices):
        global_prices = self.latest_summary['all']['price']
        if 1 - local_prices['low'] / global_prices['high'] > 1 / 2:
            price_rate = 1
        else:
            price_rate = -4 * (1 - self.max_buy_prices_rate[term]) * (
                1 - local_prices['low'] / global_prices['high']) ** 2 + 1
            # price_rate = 2 * (1 - self.max_buy_prices_rate[term]) * (
            #     1 - local_prices['low'] / global_prices['high']) + self.max_buy_prices_rate[term]
            # price_rate = 4 * (1 - self.max_buy_prices_rate[term]) * (
            # 1 - local_prices['low'] / global_prices['high']) ** 2 +
            # self.max_buy_prices_rate[term]

        price = round(local_prices['low'] * price_rate, self.price_round_digits)
        if price >= global_prices['high'] * self.max_buy_prices_rate[term]:
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle} {price}] 過去最高価格に近いため、購入できません。'
            )
            return
        elif price <= self.latest_summary['now']['price'] * 0.21:
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle} {price}] 注文価格が低すぎるため、購入できません。'
            )
            return

        # size_rate = 100 * (self.max_buy_prices_rate[term] - price / global_prices['high']) ** 2 + 1
        # size = self.min_size[term] * size_rate

        volume_rate = 100 * (self.max_buy_prices_rate[term] - price / global_prices['high']) ** 2 + 1
        volume = self.min_volume[term] * volume_rate
        if volume < self.min_volume[term]:
            volume = self.min_volume[term]
        elif volume > self.max_volume[term]:
            volume = self.max_volume[term]

        if volume > self.free_balances[self.exchange_info['symbols'][0]['quoteAsset']]:
            logger.info(
                f"[{self.product_code} {term} {child_order_cycle} {price} {volume}] {self.exchange_info['symbols'][0]['quoteAsset']} が不足しているため、購入できません。"
            )
            return

        size = volume / price
        size = round(size, self.size_round_digits)

        if size == 0:
            size = round(0.1**self.size_round_digits, self.size_round_digits)

        buy_active_same_price = pd.DataFrame()
        target_buy_history = pd.DataFrame()
        target_buy_history_active = pd.DataFrame()
        target_buy_history_completed = pd.DataFrame()
        same_category_order = pd.DataFrame()
        target_datetime = self.datetime_references[child_order_cycle]
        if not self.child_orders[term].empty:
            buy_active_same_price = self.child_orders[term].query(
                'side == "BUY" and status == "NEW" and price == @price'
            )
            target_buy_history = self.child_orders[term].query(
                'side == "BUY" and time > @target_datetime and orderCycle == @child_order_cycle'
            )
            target_buy_history_active = target_buy_history.query(
                'status == "NEW"'
            )
            target_buy_history_completed = target_buy_history.query(
                'status == "FILLED"'
            )
            same_category_order = self.child_orders[term].query(
                'side == "BUY" and status == "NEW" and orderCycle == @child_order_cycle'
            ).copy()

        if not buy_active_same_price.empty:
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle}] 同じ価格での注文がすでにあるため、購入できません。'
            )
            return

        if not target_buy_history_completed.empty:
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle}] 約定済みの注文から十分な時間が経過していないため、新規の買い注文はできません。'
            )
            return

        if same_category_order.empty:
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle}] 同じサイクルを持つACTIVEな買い注文が存在しないため、買い注文を行います。'
            )
        else:
            if len(same_category_order) >= 2:
                logger.error(
                    f'[{term} {child_order_cycle}]同じサイクルを持つACTIVEな買い注文が2つ以上あります。'
                )
            if target_buy_history_active.empty:
                logger.info(
                    f'[{self.product_code} {term} {child_order_cycle} {same_category_order.index[0]}] 前回の注文からサイクル時間以上の間約定しなかったため、買い注文を更新します。'
                )
            else:
                if price == same_category_order['price'].values[0]:
                    logger.info(
                        f'[{self.product_code} {term} {child_order_cycle}] すでに注文済みのため、購入できません。'
                    )
                    return
                else:
                    logger.info(
                        f'[{self.product_code} {term} {child_order_cycle}] 価格が変動したため、買い注文を更新します。'
                    )

            logger.info(
                f'[{self.product_code} {term} {child_order_cycle} {same_category_order.index[0]} {same_category_order["price"].values[0]} {same_category_order["size"].values[0]}] 買い注文をキャンセルします。'
            )
            self._cancel(
                term=term,
                child_order_cycle=child_order_cycle,
                order_id=same_category_order.index[0],
                child_order_type='buy'
            )

        # ----------------------------------------------------------------
        # 買い注文
        # ----------------------------------------------------------------
        response = self.spot_client.new_order(
            symbol=self.product_code,
            side='BUY',
            type='LIMIT',
            timeInForce='GTC',
            quantity=size,
            price=price
        )

        if response['status'] == 'NEW':
            print('================================================================')
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle} ${price} {size} ${volume}'
                + f'{response["orderId"]}] 買い注文に成功しました!!'
            )
            print('================================================================')
            self.update_child_orders(
                term=term,
                order_id=response['orderId'],
                child_order_cycle=child_order_cycle,
            )

    def _sell(self, term, child_order_cycle, rate, cut_loss=False):
        if self.child_orders[term].empty:
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle}] 買い注文がないため、売り注文はできません。'
            )
            return

        related_buy_order = self.child_orders[term].query(
            'side=="BUY" and status == "FILLED" and orderCycle == @child_order_cycle and relatedOrderId == "no_id"').copy()
        if related_buy_order.empty:
            logger.info(
                f'[{self.product_code} {term} {child_order_cycle}] 約定済みの買い注文がないため、売り注文はできません。'
            )
        else:
            if len(related_buy_order) >= 2:
                logger.warning(
                    f'[{self.product_code} {term} {child_order_cycle}] 同じフラグを持つ約定済みの買い注文が2つ以上あります。'
                )
            for i in range(len(related_buy_order)):
                price = round(related_buy_order['price'].values[i] * rate, self.price_round_digits)
                if price < self.latest_summary['6h']['price']['high'] and not cut_loss:
                    price = round(self.latest_summary['6h']['price']['high'], self.price_round_digits)
                size = round(related_buy_order['size'].values[i] * (1 - self.commission_rate) * 0.999, self.size_round_digits)

                if not cut_loss:
                    response = self.spot_client.new_order(
                        symbol=self.product_code,
                        side='SELL',
                        type='LIMIT',
                        timeInForce='GTC',
                        quantity=size,
                        price=price
                    )
                else:
                    response = self.spot_client.new_order(
                        symbol=self.product_code,
                        side='SELL',
                        type='STOP_LOSS_LIMIT',
                        timeInForce='GTC',
                        quantity=size,
                        price=price,
                        stopPrice=price,
                        newOrderRespType='FULL'
                    )

                if response['status'] == 'NEW':
                    print('================================================================')
                    if not cut_loss:
                        logger.info(f'[{self.product_code} {term} {child_order_cycle} {price} {size} {response["orderId"]} '
                                    + f'{int(int(related_buy_order["price"].values[i]) * (rate-1)) * size}] 売り注文に成功しました！！')
                    else:
                        logger.info(f'[{self.product_code} {term} {child_order_cycle} {price} {size} {response["orderId"]} '
                                    + f'{int(int(related_buy_order["price"].values[i]) * (rate-1)) * size}] 損切り用の売り注文に成功しました！！')
                    print('================================================================')

                    self.update_child_orders(
                        term=term,
                        child_order_cycle=child_order_cycle,
                        related_order_id=related_buy_order.index[i],
                        order_id=response['orderId'],
                    )
                    self.update_child_orders(
                        term=term,
                        child_order_cycle=child_order_cycle,
                        related_order_id=response['orderId'],
                        order_id=related_buy_order.index[i],
                    )

    def update_long_term_profit(self):
        if not self.child_orders['long'].empty:
            self.child_orders['long']['profit'] = self.child_orders['long']['size'] \
                * (self.latest_summary['now']['price'] - self.child_orders['long']['price']) \
                - self.child_orders['long']['commission']

            self.child_orders['long'].loc[self.child_orders['long']
                                          ['status'] == 'NEW', 'profit'] = 0

            self.child_orders['long']['cumsumProfit'] = self.child_orders['long']['profit'].cumsum()

            # csvファイルを更新
            df_to_csv(self.p_child_orders_path['long'], self.child_orders['long'], index=True)
            logger.debug(f'{str(self.p_child_orders_path["long"])} が更新されました。')

    def long_term(self):
        # 最新情報を取得
        self.update_child_orders(term='long')

        if int(os.environ.get('LONG_DAILY', 0)):
            # daily
            self._buy(
                term='long',
                child_order_cycle='daily',
                local_prices=self.latest_summary['1d']['price']
            )
            self._sell(
                term='long',
                child_order_cycle='daily',
                rate=self.cut_loss_rate['long'],
                cut_loss=True)

        if int(os.environ.get('LONG_WEEKLY', 1)):
            # weekly
            self._buy(
                term='long',
                child_order_cycle='weekly',
                local_prices=self.latest_summary['1w']['price']
            )
            self._sell(
                term='long',
                child_order_cycle='weekly',
                rate=self.cut_loss_rate['long'],
                cut_loss=True)

        if int(os.environ.get('LONG_MONTHLY', 0)):
            # monthly
            self._buy(
                term='long',
                child_order_cycle='monthly',
                local_prices=self.latest_summary['1m']['price']
            )
            self._sell(
                term='long',
                child_order_cycle='monthly',
                rate=self.cut_loss_rate['long'],
                cut_loss=True)

    def short_term(self):
        # 最新情報を取得
        self.update_child_orders(term='short')

        if int(os.environ.get('SHORT_HOURLY', 1)):
            # hourly
            self._buy(
                term='short',
                child_order_cycle='hourly',
                local_prices=self.latest_summary['6h']['price']
            )

            self._sell(
                term='short',
                child_order_cycle='hourly',
                rate=float(os.environ.get('SELL_RATE_SHORT_HOURLY', 1.10))
            )

        if int(os.environ.get('SHORT_DAILY', 0)):
            # daily
            self._buy(
                term='short',
                child_order_cycle='daily',
                local_prices=self.latest_summary['1d']['price']
            )

            self._sell(
                term='short',
                child_order_cycle='daily',
                rate=float(os.environ.get('SELL_RATE_SHORT_DAILY', 1.10))
            )

        if int(os.environ.get('SHORT_WEEKLY', 0)):
            # weekly
            self._buy(
                term='short',
                child_order_cycle='weekly',
                local_prices=self.latest_summary['1w']['price']
            )
            self._sell(
                term='short',
                child_order_cycle='weekly',
                rate=float(os.environ.get('SELL_RATE_SHORT_WEEKLY', 1.10))
            )
