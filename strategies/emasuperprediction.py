import talib
import configargparse
import datetime as dt
import numpy as np
import pandas as pd
from .base import Base
import core.common as common
from .enums import TradeState
from core.bots.enums import BuySellMode
from core.tradeaction import TradeAction
from lib.indicators.stoploss import StopLoss
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from strategies.predictioncommon.feature_engineering import feature_engineering
from strategies.predictioncommon.model_visualization import model_visualization
#from strategies.predictioncommon.data_loading import read_co_data
from strategies.predictioncommon.model_evaluation import model_evaluation
from termcolor import colored



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



class Emasuperprediction(Base):
    """
    Ema strategy
    About: Buy when close_price > ema20, sell when close_price < ema20 and below death_cross
    """
    #fields    

    arg_parser = configargparse.get_argument_parser()

    def __init__(self):
        args = self.arg_parser.parse_known_args()[0]
        super(Emasuperprediction, self).__init__()
        self.name = 'emasuperprediction'
        self.min_history_ticks = 60
        self.pair = self.parse_pairs(args.pairs)[0]
        self.buy_sell_mode = BuySellMode.all
        self.stop_loss = StopLoss(int(args.ticker_size))
        self.Bought = False
        self.interval = int(args.ticker_size)
        self.standardizationFeatureFlag = True
        self.numStudyTrial = 50
        self.backTestInitialFund = 1000
        self.backTestDays = 15
        self.backTestSpread = 0
        self.marginTrade = False


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
        self.chartData_ = df


        
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

        self.appreciationRate_ = self.getAppreciationRate(self.chartData_.open)
        self.quantizer_ = self.quantizer(self.appreciationRate_)

        print("Format appreciation {}".format(self.appreciationRate_))
        print("Format quantizer {}".format(self.quantizer_))

        #self.prediction(self.appreciationRate_, self.quantizer_, 0, 30, 30)
        #bactTest_ = self.backTest(self.appreciationRate_, self.quantizer_, 30, 15, False)
        #print(bactTest_)

        fed_data = feature_engineering(df)
        # feature vector
        X = fed_data.take(list(range(fed_data.shape[1] - 1)), axis=1)
        # target
        y = np.ravel(fed_data.take([fed_data.shape[1] - 1], axis=1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        # 定义一个BP神经网络
        reg = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
        # 训练
        print("start training...")
        reg.fit(X_train, y_train)
        print("training finished.")
        # 预测
        print("start predicting...")
        y_pred = reg.predict(X_test)
        print("predicting finished.")

        y_pred = pd.DataFrame(y_pred)
        y_pred.index = X_test.index

        y_test = pd.DataFrame(y_test)
        y_test.index = X_test.index
        # 将结果写入文件
        # pd.DataFrame(y_pred).to_excel('y_pred.xlsx')
        # 模型评估
        model_evaluation(y_test, y_pred)
        # 可视化
        model_visualization(y_test, y_pred)

        print(type(X), type(y), type(X_train), type(X_test), type(y_train), type(y_test), type(y_pred))

                
        
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

    def prediction(self, sampleData, classData, trainStartIndex, numFeature, numTrainSample):
        """Return probability of price rise."""
        train_X, train_y = self.preparationTrainSample(sampleData, classData, trainStartIndex, numFeature, numTrainSample)

        X = np.array([sampleData[trainStartIndex:trainStartIndex + numFeature]])
        if self.standardizationFeatureFlag:
            train_X, X = self.standardizationFeature(train_X, X)
        y = []
        for i in range(0, self.numStudyTrial):
            clf = tree.DecisionTreeClassifier()
            clf.fit(train_X, train_y)
            y.append(clf.predict(X)[0])
        return sum(y) * 1.0 / len(y)

    def getAppreciationRate(self,price):
        """Transrate chart price to appreciation rate."""
        return np.append(-np.diff(price) / price[1:].values,0)

    def quantizer(self, y):
        """Transrate appreciation rate to -1 or 1 for preparing teacher data."""
        return np.where(np.array(y) >= 0.0, 1, -1)

    def preparationTrainSample(self,sampleData,classData,trainStartIndex, numFeature, numTrainSample):
        """Prepare training sample."""
        train_X = []
        train_y = []
        for i in range(numTrainSample):
            train_X.append(sampleData[trainStartIndex + i + 1:trainStartIndex + numFeature + i + 1])
            train_y.append(classData[trainStartIndex + i])
            print("Length of class data "+str(trainStartIndex + i + 1)+" : "+str(trainStartIndex + numFeature + i + 1))
        
        
        return np.array(train_X), np.array(train_y)

    def standardizationFeature(self, train_X, test_X):
        """Standarize feature data."""
        sc = StandardScaler()
        train_X_std = sc.fit_transform(train_X)
        test_X_std = sc.transform(test_X)
        return train_X_std, test_X_std

    def backTest(self, sampleData, classData, numFeature, numTrainSample, saveBackTestGraph):
        """Do back test and return the result."""
        Y = []
        YPrediction = []
        fund = [self.backTestInitialFund]
        pastDay = 0
        accuracyUp = 0
        accuracyDown = 0
        for trainStartIndex in range(self.backTestDays):
            print("Train start index: "+str(trainStartIndex))
            yPrediction = self.quantizer(self.prediction(sampleData, classData, trainStartIndex, numFeature, numTrainSample))
            y = self.quantizer(classData[trainStartIndex - 1])
            Y.append(y.tolist())
            YPrediction.append(yPrediction.tolist())
            pastDay += 1
            if yPrediction == y:
                if yPrediction == 1:
                    accuracyUp += 1
                    fund.append(fund[pastDay - 1] * (1 + abs(self.appreciationRate_[trainStartIndex - 1]) - self.backTestSpread))
                else:
                    accuracyDown += 1
                    if self.marginTrade:
                        fund.append(fund[pastDay - 1] * (1 + abs(self.appreciationRate_[trainStartIndex - 1]) - self.backTestSpread))
                    else:
                        fund.append(fund[pastDay - 1])
            else:
                if yPrediction == 1:
                    fund.append(fund[pastDay - 1] * (1 - abs(self.appreciationRate_[trainStartIndex - 1]) - self.backTestSpread))
                else:
                    if self.marginTrade:
                        fund.append(fund[pastDay - 1] * (1 - abs(self.appreciationRate_[trainStartIndex - 1]) - self.backTestSpread))
                    else:
                        fund.append(fund[pastDay - 1])

        backTestAccuracyRateUp = float(accuracyUp) / sum(np.array(YPrediction)[np.where(np.array(YPrediction) == 1)])
        backTestAccuracyRateDown = -float(accuracyDown) / sum(np.array(YPrediction)[np.where(np.array(YPrediction) == -1)])

        trainStartIndex = 0
        backTestCurrentPrice = self.chartData_.open[trainStartIndex:trainStartIndex + self.backTestDays + 1]
        backTestCurrentPrice = backTestCurrentPrice[::-1].tolist()
        backTestDate = self.chartData_.date[trainStartIndex:trainStartIndex + self.backTestDays + 1]
        backTestDate = backTestDate[::-1].tolist()

        backTestFinalFund = fund[-1]
        backTestInitialCurrentPrice = backTestCurrentPrice[0]
        backTestFinalCurrentPrice = backTestCurrentPrice[-1]
        backTestIncreasedFundRatio = (backTestFinalFund - self.backTestInitialFund) / self.backTestInitialFund
        backTestIncreasedCurrentPriceRatio = (backTestFinalCurrentPrice - backTestInitialCurrentPrice) / backTestInitialCurrentPrice

        columnNames = ["AccuracyRateUp", "AccuracyRateDown",
                       "InitialFund", "FinalFund", "IncreasedFundRatio",
                       "InitialCurrentPrice", "FinalCurrentPrice", "IncreasedCurrentPriceRatio"]
        columnValues = [backTestAccuracyRateUp, backTestAccuracyRateDown,
                        self.backTestInitialFund, backTestFinalFund, backTestIncreasedFundRatio,
                        backTestInitialCurrentPrice, backTestFinalCurrentPrice, backTestIncreasedCurrentPriceRatio]
        backTestResult = pd.DataFrame(np.array([columnValues]), columns=columnNames)

        if saveBackTestGraph:
            fig1, ax1 = plt.subplots(figsize=(11, 6))
            p1, = ax1.plot(backTestDate, fund, "-ob")
            ax1.set_title("Back test (" + self.currentPair + ")")
            ax1.set_xlabel("Day")
            ax1.set_ylabel("Fund")
            plt.grid(fig1)
            ax2 = ax1.twinx()
            p2, = ax2.plot(backTestDate, backTestCurrentPrice, '-or')
            ax2.set_ylabel("Price[" + self.currentPair + "]")
            ax1.legend([p1, p2], ["Fund", "Price_" + self.currentPair], loc="upper left")
            plt.savefig(self.workingDirPath + "/backTest_" + self.currentPair + ".png", dpi=50)
            plt.close()

            self.backTestResult_ = backTestResult

        return backTestResult


        

