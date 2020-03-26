import talib
import configargparse
import pandas as pd
import core.common as common
import datetime as dt
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns

from .base import Base
from .enums import TradeState
from core.bots.enums import BuySellMode
from core.tradeaction import TradeAction
from lib.indicators.stoploss import StopLoss
from sklearn import tree
from termcolor import colored
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



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


class Emabond(Base):
    """
    Ema strategy
    About: Buy when close_price > ema20, sell when close_price < ema20 and below death_cross
    """
    #fields    
    


    arg_parser = configargparse.get_argument_parser()

    def __init__(self):
        args = self.arg_parser.parse_known_args()[0]
        super(Emabond, self).__init__()
        self.name = 'emabond'
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

        self.emabondfunction(df)
        
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

    def series_to_supervised(self, data, n_in = 1, n_out = 1, dropnan = True):
        if type(data) == list:
            n_vars = 1
        else:
            n_vars = data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def emabondfunction(self, data):
        #Make np array - float
        values = data[['open'] + ['high'] + ['low'] + ['close']].values
        values = values.astype('float64')

        print(values)

        #Normalise features
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaled = scaler.fit_transform(values)

        #Convert to supervise learning
        reframed = self.series_to_supervised(scaled, 1, 1)

        #Keep only necessary columns
        reframed.drop(reframed.columns[[4, 5, 6]], axis = 1, inplace = True)
        print(reframed)

        #Split data into 70/30 training/test
        values = reframed.values
        n_train = int(len(values) * 0.9)
        train = values[:n_train, :]
        test = values[n_train:, :]

        #Split into inputs and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        print("Test full: "+str(test))
        print("Test x: "+str(test_X))
        print("Test y: "+str(test_y))

        #reshape input to be 3d [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        print(str(train_X.shape))

        #Build LSTM - 300 epochs
        mdl = Sequential()
        mdl.add(LSTM(100, input_shape = (train_X.shape[1], train_X.shape[2])))
        mdl.add(Dense(1))
        mdl.compile(loss = 'mae', optimizer = 'adamax')
        mdl_hist = mdl.fit(train_X, train_y,
                            epochs = self.interval*60,
                            batch_size = 100,
                            validation_data = (test_X, test_y),
                            verbose = 0,
                            shuffle = False)

        #Make prediction using test_X
        yhat = mdl.predict(test_X)
        print(yhat)

        #De-normalise predictions back to original scale
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast

        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)

        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]

        #Evaluate performance using RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.12f' % rmse)

        #Plot predicted versus actual
        pyplot.plot(inv_y, label = 'Actual')
        pyplot.plot(inv_yhat, label = 'Predicted')
        pyplot.legend(loc = 'best')
        pyplot.show()    


            

