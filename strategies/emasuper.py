import talib
import configargparse
from .base import Base
import core.common as common
from .enums import TradeState
from core.bots.enums import BuySellMode
from core.tradeaction import TradeAction
from lib.indicators.stoploss import StopLoss
from sklearn import tree
from termcolor import colored
import datetime as dt



acc = 10
Points = []
Highs = []
Lows = []
Volumes = []
dates = []
CashRecords = []

Cash = 100
days = 0
decision = 0
stockSymbol = 'AAPL'

def algot(t):

    features = []
    labels = []

    for i in range(len(t) - acc + 1):
        features.append(t[-1*acc:-1])

        #1 means price went down
        if t[-1] > t[-2]:
            labels.append(1)
        else:
            labels.append(0)
            
    clf = tree.DecisionTreeClassifier()
    clf.fit(features, labels)

    print(len(features))

    if clf.predict([t[-1*acc+1:]]) == 1:
        return 0
    else:
        return 1

def algo(t,h,l,v):

        features = []
        labels = []

        for i in range(len(t) - acc):
            
            temp_t = t[acc + i - 1]
            temp_h = h[acc + i - 1]
            temp_l = l[acc + i - 1]
            temp_v = v[acc + i - 1]
            
            features.append([temp_t, temp_h, temp_l, temp_v])
        
            #1 means price went up
            if t[acc + i] > t[acc + i - 1]:
                labels.append([1])
            else:
                labels.append([0])
                
        clf = tree.DecisionTreeClassifier()
        clf.fit(features, labels)
        temp_list = []
        
        for i in range(acc):
            temp_list.append([])
            temp_list[i].append(t[-1*(acc - i)])
            temp_list[i].append(h[-1*(acc - i)])
            temp_list[i].append(l[-1*(acc - i)])
            temp_list[i].append(v[-1*(acc - i)])
            
        if clf.predict(temp_list)[0] == 1:
            return 0
        else:
            return 1


class Emasuper(Base):
    """
    Ema strategy
    About: Buy when close_price > ema20, sell when close_price < ema20 and below death_cross
    """
    #fields    
    


    arg_parser = configargparse.get_argument_parser()

    def __init__(self):
        args = self.arg_parser.parse_known_args()[0]
        super(Emasuper, self).__init__()
        self.name = 'emasuper'
        self.min_history_ticks = 30
        self.pair = self.parse_pairs(args.pairs)[0]
        self.buy_sell_mode = BuySellMode.all
        self.stop_loss = StopLoss(int(args.ticker_size))
        self.Bought = False
        self.interval = int(args.ticker_size)


    def calculate(self, look_back, wallet):
        """
        Main strategy logic (the meat of the strategy)
        """
        (dataset_cnt, _) = common.get_dataset_count(look_back, self.group_by_field)

        # Wait until we have enough data
        if dataset_cnt < self.min_history_ticks:
            print('dataset_cnt:', dataset_cnt)
            return self.actions

        self.actions.clear()
        new_action = TradeState.none

        # Calculate indicators
        df = look_back.tail(self.min_history_ticks)
        
        for i in df[['close']]:
            for j in df[i]:
                last_price = round(j,10)
                Points.append(last_price)

        for i in df[['high']]:
            for j in df[i]:
                Highs.append(round(j,8))

        for i in df[['low']]:
            for j in df[i]:
                Lows.append(round(j,8))
                
        for i in df[['volume']]:
            for j in df[i]:
                Volumes.append(round(j,8))

        for i in df[['date']]:
            for j in df[i]:
                last_time = dt.datetime.fromtimestamp(j)
                dates.append(last_time)
                
        
        days = len(df[['close']])
        
        print("Last time: "+str(last_time)+" "+str(last_price))

        if days > acc:
            decision = algot(Points[:days])

        new_last_time = last_time + dt.timedelta(minutes = self.interval) 

        print('-----------------------------------------------------------------------------------------')
        print('|                                                                                       |')
        print('|                                                                                       |')
        print('|                                                                                       |')

        #if self.Bought == True:
        if decision == 0:
                self.Bought = False
                print(colored("*           Buy now or wait, price will went UP at {}          *".format(new_last_time), 'green'))
                
                #new_action = TradeState.sell
        #else:
        elif decision == 1:
                self.Bought = True
                #new_action = TradeState.buy
                print(colored("*           Sell now or wait, price will went DOWN at {}          *".format(new_last_time), 'red'))

        print('|                                                                                       |')
        print('|                                                                                       |')
        print('|                                                                                       |')
        print('-----------------------------------------------------------------------------------------')


        
        
        trade_price = self.get_price(new_action, df.tail(), self.pair)


        
        # Get stop-loss
        #if new_action == TradeState.buy and self.stop_loss.calculate(close):
        #    print('stop-loss detected,..selling')
        #    new_action = TradeState.sell

        action = TradeAction(self.pair,
                             new_action,
                             amount=None,
                             rate=trade_price,
                             buy_sell_mode=self.buy_sell_mode)

        self.actions.append(action)
        return self.actions


    

