""" 
 Project: Chapter14 Evaluating the RUL
 Class: EIT210 - Machine /Process Control, Factory Automation
 Author: Vasilije Mehandzic
 Original source: https://github.com/PacktPublishing/Hands-On-Industrial-Internet-of-Things/tree/master/Chapter14

 File: evaluating_the_RUL.py
 Description: 
 Date: 12/16/2020 """


import pyqtgraph
import pandas
import numpy
import math
import datetime
import seaborn
import matplotlib.pyplot
import sklearn.feature_selection
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.metrics
import keras.models
import keras.layers
import scipy.stats


def addToStatus(object, title = ''):
    global status_txt
    status_txt = "\n" + str(datetime.datetime.now()) + "  -----" + title + "-----"  + "\n" + str(object) + "\n" + status_txt
    label_status.setPlainText(status_txt)
    app.processEvents()
    f = open(log_file_name, "a")
    f.write(status_txt)
    f.close()

def act1():
    global columns, df, status_txt    
    columns = ['unitid', 'time', 'set_1','set_2','set_3']
    columns.extend(['sensor_' + str(i) for i in range(1,22)])
    df = pandas.read_csv('./data/train_FD001.txt', delim_whitespace=True, names=columns)
    addToStatus(df.head(), title = "READ THE DATASET, DF HEAD")

def act2():
    if 'df' in vars() or 'df' in globals() or 'df' in vars(__builtins__):
        global df, status_txt
        pandas.set_option('display.max_columns', None)
        pandas.set_option('display.max_rows', None)
        df_std = df.groupby('unitid').std()
        addToStatus(df_std==0, title="DF_STD")
        to_drop = ['set_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
        addToStatus(df.head(), title = "REMOVING UNUSEFUL DATA "+ str(to_drop))
        df = df.drop(to_drop, axis=1)

        # correlation
        def calculate_pvalues(df):
            df = df.dropna()._get_numeric_data()
            dfcols = pandas.DataFrame(columns=df.columns)
            pvalues = dfcols.transpose().join(dfcols, how='outer')
            for r in df.columns:
                for c in df.columns:
                    pvalues[r][c] = round(scipy.stats.pearsonr(df[r], df[c])[1], 4)
            return pvalues.replace(numpy.nan,0)

        addToStatus(calculate_pvalues(df[(df.unitid ==1)]), title = "CORRELATION 1") 
        addToStatus(calculate_pvalues(df[(df.unitid ==5)]), title = "CORRELATION 3")
        addToStatus(calculate_pvalues(df[(df.unitid ==10)]), title = "CORRELATION 10")
    
def act3():
    if 'df' in vars() or 'df' in globals() or 'df' in vars(__builtins__):    
        global df, status_txt
        seaborn.pairplot(df[(df.unitid <=5) ],  hue="unitid",  vars=['set_1', 'set_2','sensor_2','sensor_3','sensor_4'])
        addToStatus('', title = "PLOT FIRST 5 ENGINES AND VARIABLES")
        matplotlib.pyplot.show()

def act4():
    if 'df' in vars() or 'df' in globals() or 'df' in vars(__builtins__):    
        global df, status_txt
        df1=df[(df.unitid <5) ]
        i=0

        for column in df1:
            if ('sensor' in column):
                i=i+1
                ax = matplotlib.pyplot.subplot(4,4,i)
                ax = seaborn.tsplot(time="time", value=column, condition='unitid',
                            unit='unitid',legend=False,
                            data=df1, ax=ax)    

        addToStatus('', title = "PLOTING THE TIME SERIES")
        matplotlib.pyplot.show()

def act5():
    if 'df' in vars() or 'df' in globals() or 'df' in vars(__builtins__):    
        global df, status_txt
        addToStatus('', title = 'PERFORMING FEATURE SELECTION')
        rfg = sklearn.ensemble.RandomForestRegressor(n_estimators=30, random_state=1)
        rfe = sklearn.feature_selection.RFE(rfg, 4)
        addToStatus('', title = 'EXTRACTING VARS')

        array = df.values
        X = array[:,0:-1]
        y = array[:,-1]
        fit = rfe.fit(X, y)
        addToStatus('', title = 'RFE.FIT')
        # report selected features
        addToStatus('', title = 'REPORTING SELECTED FEATURES')
        names = df.columns.values[0:-1]

        for i in range(len(fit.support_)):
            if fit.support_[i]:
                addToStatus(names[i], title = str(i)+"> ")

def act6():
    if 'df' in vars() or 'df' in globals() or 'df' in vars(__builtins__):
        global df, status_txt
        def prepare_dataset(dataframe, columns):
            dataframe = dataframe[columns]
            dataset = dataframe.values
            dataset = dataset.astype('float32')

            # normalize the dataset
            scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            return dataset

        def build_model(input_dim):
            # create model
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(16, input_dim=input_dim, activation='relu'))
            model.add(keras.layers.Dense(32, activation='relu'))
            model.add(keras.layers.Dense(1, activation='sigmoid'))

            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model

        def create_train_dataset(dataset):
            dataX, dataY = [], []
            start=len(dataset)

            for i in range(len(dataset)):
                    a=dataset[i]
                    b=(start-i) / start
                    dataX.append(a)
                    dataY.append(b)
                    
            return numpy.array(dataX), numpy.array(dataY)

        def train_model(model, dataset):

            # create the dataset
            trainX, trainY = create_train_dataset(dataset)

            # Fit the model
            model.fit(trainX, trainY, epochs=150, batch_size=10, verbose=0)
            
            # make predictions
            trainPredict = model.predict(trainX)

                # calculate root mean squared error
            trainScore = math.sqrt(sklearn.metrics.mean_squared_error(trainY, trainPredict[:,0]))
            #print('Train Score: %.2f RMSE' % (trainScore))
            addToStatus('Train Score: %.2f RMSE' % (trainScore), title = "TRAIN SCORE")


        # prepare model
        columns_feature = ['set_1','set_2','sensor_4','sensor_7','sensor_11','sensor_12']
        columns_feature = ['sensor_4','sensor_7']
        addToStatus('', title = "preapare the model, columns_feature = " + str(columns_feature))
        

        numpy.random.seed(7)

        addToStatus('', title = "build model")
        model=build_model(len(columns_feature))

        i = int(combo_box.currentText())
        addToStatus('', title = "train the model, unitid = " + str(i))
        dataset= prepare_dataset(df[(df.unitid ==i)],columns_feature)
        train_model(model, dataset)

        addToStatus('', title = "test")
        df_test = pandas.read_csv('./data/test_FD001.txt', delim_whitespace=True,names=columns)
        expected = pandas.read_csv('./data/RUL_FD001.txt', delim_whitespace=True,names=['RUL'])

        n = len(dataset)
        addToStatus('', title = "dataset test i="+str(i)+ " dataset len=" +str(n))
        dataset_test = prepare_dataset(df_test[(df_test.unitid ==i)],columns_feature)
        testPredict = model.predict(dataset_test)
        testPredict = numpy.multiply(testPredict,n)
        addToStatus("RUL of Engine %s : predicted:%s expected:%s"%(1, testPredict[-1], expected['RUL'][i-1]), title = "BUILD, TRAIN, TEST")

app = pyqtgraph.Qt.mkQApp()
my_font = pyqtgraph.Qt.QtGui.QFont("Noto Mono", 10)
app.setFont(my_font)
pyqtgraph.setConfigOption('background', '#eee')
pyqtgraph.setConfigOption('foreground', 'k')
main_window = pyqtgraph.Qt.QtGui.QMainWindow()
main_window.setWindowIcon(pyqtgraph.Qt.QtGui.QIcon('progn.ico')) 
main_window.setWindowTitle("EIT210 - Evaluating the RUL") 
central_widget = pyqtgraph.Qt.QtGui.QWidget()
vbox_layout = pyqtgraph.Qt.QtGui.QVBoxLayout()
main_window.setCentralWidget(central_widget)
central_widget.setLayout(vbox_layout)
hbox_layout1 = pyqtgraph.Qt.QtGui.QHBoxLayout()
hbox_layout2 = pyqtgraph.Qt.QtGui.QHBoxLayout()
hbox_layout3 = pyqtgraph.Qt.QtGui.QHBoxLayout()
hbox_status_layout = pyqtgraph.Qt.QtGui.QHBoxLayout()
btn_1 = pyqtgraph.Qt.QtGui.QPushButton(' 1: read the dataset ')
btn_2 = pyqtgraph.Qt.QtGui.QPushButton(' 2: corelation engine ')
btn_3 = pyqtgraph.Qt.QtGui.QPushButton(' 3: plot first 5 engines and variables')
btn_4 = pyqtgraph.Qt.QtGui.QPushButton(' 4: plot timeseries ')
btn_5 = pyqtgraph.Qt.QtGui.QPushButton(' 5: fit')
u_id = pyqtgraph.Qt.QtGui.QLabel(' unitid:')
combo_box = pyqtgraph.Qt.QtGui.QComboBox()
for i in range(1,100):
    combo_box.addItem(str(i))
btn_6 = pyqtgraph.Qt.QtGui.QPushButton('              6: build, train, test ')
btn_6.setToolTip("6")
btn_1.clicked.connect(act1)
btn_2.clicked.connect(act2)
btn_3.clicked.connect(act3)
btn_4.clicked.connect(act4)
btn_5.clicked.connect(act5)
btn_6.clicked.connect(act6)
label_status = pyqtgraph.Qt.QtGui.QPlainTextEdit("stat here")
vbox_layout.addLayout(hbox_layout1)
vbox_layout.addLayout(hbox_layout2)
vbox_layout.addLayout(hbox_layout3)
vbox_layout.addLayout(hbox_status_layout)
hbox_layout1.addWidget(btn_1)
hbox_layout1.addWidget(btn_2)
hbox_layout2.addWidget(btn_3)
hbox_layout2.addWidget(btn_4)
hbox_layout3.addWidget(btn_5)
hbox_layout3.addWidget(btn_6)
hbox_layout3.addWidget(u_id)
hbox_layout3.addWidget(combo_box)
hbox_status_layout.addWidget(label_status)
main_window.show()

status_txt = ''
log_file_name = str(datetime.datetime.now())+".log"
log_file_name = log_file_name.replace(" ","_").replace(":","-")
f = open(log_file_name, "w+")
f.close()

app.exec_()