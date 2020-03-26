import time
import datetime
import pandas as pd
import configargparse
from core.bots.enums import TradeMode
from exchanges.base import Base
from strategies.enums import TradeState
from termcolor import colored
from json import JSONDecodeError
from core.bots.enums import BuySellMode

from .client import * 


class Huobi(Base):
    """
    Huobi interface
    """
    arg_parser = configargparse.get_argument_parser()
    arg_parser.add('--huobi_api_key', help='Huobi API key')
    arg_parser.add("--huobi_secret", help='Huobi secret key')
    arg_parser.add("--huobi_txn_fee", help='Huobi txn. fee')
    arg_parser.add("--huobi_buy_order", help='Poloniex buy order type ("market" or "limit")')
    arg_parser.add("--huobi_sell_order", help='Poloniex sell order type ("market" or "limit")')
    valid_candle_intervals = [300, 900, 1800, 7200, 14400, 86400]

    def __init__(self):
        super(Huobi, self).__init__()
        args = self.arg_parser.parse_known_args()[0]
        api_key = args.huobi_api_key
        secret = args.huobi_secret
        self.transaction_fee = float(args.huobi_txn_fee)
        self.huobi = HuobiClient(api_key, secret)
        self.buy_order_type = args.huobi_buy_order
        self.sell_order_type = args.huobi_sell_order
        self.verbosity = args.verbosity
        self.pair_delimiter = "_"
        self.account_id = int(self.huobi.get("/v1/account/accounts", {})["data"][0]["id"])

    def make_valid_pair(self, pair):
        return pair[:3] + "_" + pair[3:]

    def remove_delimiter(self, pair):
        return "".join([c for c in pair if c != "_"])

    def get_balances(self):
        """
        Return available account balances (function returns ONLY currencies > 0)
        """
        data = self.huobi.get("/v1/account/accounts/%d/balance" % self.account_id, {})["data"]["list"]
        # get only trade balances
        balance = {d["currency"]: float(d["balance"]) for d in data if d["type"] == "trade"} 
        only_non_zeros = {k: float(v) for k, v in balance.items() if float(v) > 0.0}
        return only_non_zeros

    def get_symbol_ticker(self, symbol, candle_size=5):
        data = self.huobi.http_get_request(self.huobi.MARKET_URL + "/market/detail", {
            "symbol": self.remove_delimiter(symbol)
        })["tick"]
        df = pd.DataFrame(data)
        df = df.rename(columns={"ts": "date", "vol": "volume"})
        df["pair"] = symbol
        df['date'] = int(datetime.datetime.utcnow().timestamp())
        return df


    def cancel_order(self, order_number):
        """
        Cancels order for given order number
        """
        self.huobi.post("/v1/order/orders/%d/submitcancel" % order_number, {})


    def get_open_orders(self, currency_pair="all"):
        if currency_pair == "all":
            return [self.get_symbol_open_orders(currency) for currency in self.get_pairs()]
        else:
            return self.get_symbol_open_orders(currency_pair)

    def get_symbol_open_orders(self, currency_pair):
        """
        Returns your open orders
        """
        currency_pair = currency_pair.replace("_", "")
        data = self.huobi.get("/v1/order/orders", {
            "symbol": self.remove_delimiter(currency_pair),
            "states": "pre-submitted,submitted",
        })["data"]
        return data


    def get_pairs(self):
        """
        Returns ticker pairs for all currencies
        """
        #for d in self.huobi.get("/v1/common/symbols", {})["data"]:
         #   print("format: {}".format(d))
        return ["{}_{}".format(d["base-currency"].lower(), d["quote-currency"].lower()) for d in self.huobi.get("/v1/common/symbols", {})["data"]]

    def trade(self, actions, wallet, trade_mode):
        if trade_mode == TradeMode.backtest:
            return Base.trade(actions, wallet, trade_mode)
        else:
            actions = self.life_trade(actions)
            return actions

    def life_trade(self, actions):
        """
        Places orders and returns order number
        !!! For now we are NOT handling postOnly type of orders !!!
        """
        for action in actions:

            
            if action.action == TradeState.none:
                actions.remove(action)
                continue

            # Handle buy_sell mode
            wallet = self.get_balances()
            if action.buy_sell_mode == BuySellMode.all:
                action.amount = self.get_buy_sell_all_amount(wallet, action)
            elif action.buy_sell_mode == BuySellMode.fixed:
                action.amount = self.get_buy_sell_all_amount(wallet, action)

            if self.verbosity:
                print('Processing live-action: ' + str(action.action) +
                      ', amount:', str(action.amount) +
                      ', pair:', action.pair+
                      ', rate:', "{0:.8f}".format(action.rate) +
                      ', buy_sell_mode:', action.buy_sell_mode)

            # If we don't have enough assets, just skip/remove the action
            if action.amount == 0.0 or action.amount == 0.00000000 :
                print(colored('No assets to buy/sell, ...skipping: ' + str(action.amount) + ' ' + action.pair, 'green'))
                actions.remove(action)
                continue

            # ** Buy Action **
            if action.action == TradeState.buy:
                print(colored('Setting buy order: ' + str(action.amount) + '' + action.pair, 'green'))
                action.order_number = self.buy(action.pair, action.rate, action.amount, self.buy_order_type)
                #amount_unfilled = action.order_number.get('amountUnfilled')
                if action.order_number != None:
                    actions.remove(action)
                    print(colored('Bought: ' + str(action.amount) + '' + action.pair, 'green'))
                #else:
                 #   action.amount = amount_unfilled
                  #  print(colored('Not filled 100% buy txn. Unfilled amount: ' + str(amount_unfilled) + '' + action.pair, 'red'))

            # ** Sell Action **
            elif action.action == TradeState.sell:
                print(colored('Setting sell order: ' + str(action.amount) + '' + action.pair, 'yellow'))
                action.order_number = self.sell(action.pair, action.rate,  action.amount, self.sell_order_type)
                #amount_unfilled = action.order_number.get('amountUnfilled')
                if action.order_number != None:
                    actions.remove(action)
                    print(colored('Sold: ' + str(action.amount) + '' + action.pair, 'yellow'))
                #else:
                #   action.amount = amount_unfilled
                #    print(colored('Not filled 100% sell txn. Unfilled amount: ' + str(amount_unfilled) + '' + action.pair, 'red'))
        return actions
    
    def buy(self, pair, rate, amount, order_type):
        params = {
                "account-id": str(self.account_id),
                "amount": str(amount),
                "source": "api",
                "symbol": self.remove_delimiter(pair),
                "type": "buy-%s" % self.buy_order_type,
                

        }
        return int(self.huobi.post("/v1/order/orders/place", params)["data"])

    def sell(self, pair, rate, amount, order_type):
        params = {
                "account-id": str(self.account_id),
                "amount": str(amount),
                "source": "api",
                "symbol": self.remove_delimiter(pair),
                "type": "sell-%s" % self.sell_order_type,
                
        }
        return int(self.huobi.post("/v1/order/orders/place", params)["data"])

    def get_candles(self, currency_pair, epoch_start, epoch_end, interval_in_sec=300):
        periods = [
            (60, "1min"),
            (300, "5min"),
            (900, "15min"),
            (1800, "30min"),
            (3600, "60min"),
            (3600*24, "1day"),
        ]
        if "_" in currency_pair:
            currency_pair = currency_pair.replace("_", "") 


        data = self.huobi.http_get_request(self.huobi.MARKET_URL + "/market/history/kline", {
            "symbol": self.remove_delimiter(currency_pair),
            "size": 2000,
            "period": min(periods, key=lambda x: abs(interval_in_sec - x[0]))[1]
        })

        #res = [d for d in data["data"]]

        tickers = data['data']
        raw_tickers = []

        if tickers is None:
            print(colored('\n! Got empty tickers for pair: ' + currency_pair, 'red'))
            return dict()

        # Parse tickers
        for ticker in tickers:

            raw_ticker = dict()
            raw_ticker['high'] = ticker['high']
            raw_ticker['low'] = ticker['low']
            raw_ticker['open'] = ticker['open']
            raw_ticker['close'] = ticker['close']
            raw_ticker['volume'] = ticker['vol']
            raw_ticker['quoteVolume'] = ticker['count']
            raw_ticker['date'] = ticker['id']
            raw_ticker['weightedAverage'] = 0.0

            raw_tickers.append(raw_ticker)


        
        return raw_tickers.copy()

    def get_candles_df(self, currency_pair, epoch_start, epoch_end, period=500):
        
        df = pd.DataFrame(self.get_candles(currency_pair, epoch_start, epoch_end, period))
        df = df.rename(columns={"vol": "volume"})
        df["pair"] = currency_pair
        df['date'] = int(datetime.datetime.utcnow().timestamp())
        return df.tail(1)
