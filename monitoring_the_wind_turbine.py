""" 
 Project: Chapter14 - Monitoring Evaluating the RUL
 Class: EIT210 - Machine /Process Control, Factory Automation
 Author: Vasilije Mehandzic
 Original source: https://github.com/PacktPublishing/Hands-On-Industrial-Internet-of-Things/tree/master/Chapter14

 File: monitoring_the_wind_turbine.py
 Description: 
 Date: 12/16/2020 """


import numpy
import pandas
import pyqtgraph


def wind_turbine_model(x):
    # cut-in pseed vs cut-out speed
    if x < 4.5 or x > 21.5:
        return 0

    # standard operability
    return 376.936 - 195.8161*x + 33.75734*x**2 - 2.212492*x**3 + 0.06309095*x**4 - 0.0006533647*x**5

def loadData():
    global df
    df = pandas.read_csv('./data/wind_turbine.csv')  # load data


def plotDataFromCSV():
    global plot_window_01, wind_speed_ms, power_generated_kw, expected_power   
    wind_speed_ms = numpy.asarray(df.wind_speed_ms)
    power_generated_kw = numpy.asarray(df.power_generated_kw)
    expected_power = [wind_turbine_model(x) for x in df.wind_speed_ms]
    plot_window_01 = plot_widget.plot(wind_speed_ms,power_generated_kw, pen=pyqtgraph.mkPen('#eee', width=0), symbol='o',name = "power generated per wind_speed",symbolBrush=(200,0,0), symbolPen='w')
 
def plotReferencePower():
    global plot_window_02, reference_power
    reference_power = numpy.asarray([wind_turbine_model(x) for x in range(0, 30)])
    plot_window_02 = plot_widget.plot(reference_power, pen=pyqtgraph.mkPen('k', width=5), symbol='x',name = "reference power")

def plotExpectedPower():
    global plot_window_03, expected_power, df
    expected_power = [wind_turbine_model(x) for x in df.wind_speed_ms]
    plot_window_03 = plot_widget.plot(df.wind_speed_ms, expected_power, pen=pyqtgraph.mkPen('#eee', width=0), symbol='x',name = "expected power",symbolBrush=(0,0,200), symbolPen='w')

def evaluate():
    global expected_power
    # evaluate ratio
    de = []
    ts = []
    for i in range(0, len(expected_power)):
        mp = df.power_generated_kw[i]
        ep = expected_power[i]
        if ep > 0 and mp > 0:
            t = df.cycle_10_mins[i]
            de.append((mp-ep) / ep)
            ts.append(t)

    # predict degradation    
    year = int(combo_box.currentText())
    samples = 365*24*year
    z = numpy.polyfit(ts, de, 1)
    label_status.setText("Degradation in %s years will be %.2f %%" % (year, 100*(z[0]*(samples*year) + z[1])))



wind_speed_ms = []
power_generated_kw = []
app = pyqtgraph.Qt.mkQApp()
my_font= pyqtgraph.Qt.QtGui.QFont("Trebushet MS", 15)
app.setFont(my_font)
pyqtgraph.setConfigOption('background', '#eee')
pyqtgraph.setConfigOption('foreground', 'k')
main_window = pyqtgraph.Qt.QtGui.QMainWindow()
central_widget = pyqtgraph.Qt.QtGui.QWidget()
vbox_layout = pyqtgraph.Qt.QtGui.QVBoxLayout()
plot_widget = pyqtgraph.PlotWidget()
#main_window.setGeometry(120,180,1320,1080)
main_window.setWindowTitle("Degradation in %s years")
plot_widget.setXRange(0,30)
plot_widget.setYRange(0,300)
main_window.setCentralWidget(central_widget)
main_window.setWindowIcon(pyqtgraph.Qt.QtGui.QIcon('wind.ico')) 
main_window.setWindowTitle("EIT210 - Monitoring The Wind Turbine") 
central_widget.setLayout(vbox_layout)
vbox_layout.addWidget(plot_widget)
hbox_layout = pyqtgraph.Qt.QtGui.QHBoxLayout()
hbox_status_layout = pyqtgraph.Qt.QtGui.QHBoxLayout()
btn_load_reference_power = pyqtgraph.Qt.QtGui.QPushButton('   Load Reference Power   ')
btn_load_from_csv = pyqtgraph.Qt.QtGui.QPushButton('   Load Wind/Power from CSV   ')
btn_load_expected_power = pyqtgraph.Qt.QtGui.QPushButton('   Load Expected Power   ')
lbl_calculate = pyqtgraph.Qt.QtGui.QLabel('                        Degradation in')
btn_calculate = pyqtgraph.Qt.QtGui.QPushButton('Calculate')

combo_box = pyqtgraph.Qt.QtGui.QComboBox()
for i in range(1,100):
    combo_box.addItem(str(i))
ln_yrs = pyqtgraph.Qt.QtGui.QLabel('years')
btn_calculate.clicked.connect(evaluate)
btn_load_from_csv.clicked.connect(plotDataFromCSV)
btn_load_reference_power.clicked.connect(plotReferencePower)
btn_load_expected_power.clicked.connect(plotExpectedPower)

label_status = pyqtgraph.Qt.QtGui.QLabel('')
vbox_layout.addLayout(hbox_layout)
vbox_layout.addLayout(hbox_status_layout)

hbox_layout.addWidget(btn_load_reference_power)
hbox_layout.addWidget(btn_load_expected_power)
hbox_layout.addWidget(btn_load_from_csv)
hbox_layout.addWidget(lbl_calculate)
hbox_layout.addWidget(combo_box)
hbox_layout.addWidget(ln_yrs)
hbox_layout.addWidget(btn_calculate)
hbox_status_layout.addWidget(label_status)
plot_widget.setTitle('Wind speed[m/s] / Power Generated[kW]   &   Reference Power[m/s] / Power Generated[kW]')
plot_widget.showGrid(x = True, y = True, alpha = 0.33)
plot_widget.setLabel('bottom', "wind speed [m/s]")
plot_widget.setLabel('left', "power [kW]")
main_window.show()
loadData()
app.exec_()
