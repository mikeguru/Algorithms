#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Michael (Wen) Jiang"

import quandl
quandl.ApiConfig.api_key = 'obLWrsHwLyxBs1bxiqBB'

# matplotlib 	2.1.1
# scipy		    1.0.0
# numpy		    1.4.0
# pandas 		0.22.0

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fp
import scipy.signal as sc

def roll(data):
    short_rolling = data.rolling(window=20).mean()
    medium_rolling = data.rolling(window=30).mean()
    long_rolling = data.rolling(window=50).mean()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data.index, data, label=str(len(data)) + ' Datapoints of commodity prices')
    ax.plot(short_rolling.index, short_rolling, label='20 days rolling')
    ax.plot(medium_rolling.index, medium_rolling, label='30 days rolling')
    ax.plot(long_rolling.index, long_rolling, label='50 days rolling')
    ax.set_xlabel('Date')
    ax.set_ylabel('Commodity  price ($)')
    ax.legend()
    plt.show()


# Detrend to filter out noise
def fft(data):
    # Based on end-of-day stock data
    yearly_freq = 1 / 252
    qrty_freq = 1 / 63
    monthly_freq = 1 / 21
    weekly_freq = 1 / 5

    detrend = sc.detrend(data)
    plt.plot(detrend)
    plt.title(str(len(detrend)) + " Datapoints from Detrended Prices of Commodity")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.show()

    # Isolate repetitive cycles with largest magnitude – detect cycles
    w = np.blackman(20)
    y = np.convolve(w / w.sum(), detrend, mode='same')
    plt.plot(y)
    plt.title(str(len(y)) + " Datapoints from Blackman window of Detrended Prices")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.show()

    # FFT to identify cycles at certain frequencies and magnitudes for a commodity
    fft = abs(fp.rfft(y))
    plt.plot(fft)
    plt.title(str(len(fft)) + " Datapoints from FFT of Blackman Window")
    plt.xlabel("Frequency (1/" + str(len(y)) + "Days)")
    plt.ylabel("Magnitude (|Y(Frequency)|)")

    # Extra vertical spacer
    plt.axvline(x=yearly_freq * len(data), color='black', linestyle='solid', linewidth=1)
    plt.text(yearly_freq * len(data), 1, 'Yearly', ha='center', va='center', rotation='vertical')

    plt.axvline(x=qrty_freq * len(data), color='blue', linestyle='solid', linewidth=1)
    plt.text(qrty_freq * len(data), 1, 'Quarterly', ha='center', va='center', rotation='vertical')

    plt.axvline(x=monthly_freq * len(data), color='green', linestyle='solid', linewidth=1)
    plt.text(monthly_freq * len(data), 1, 'Monthly', ha='center', va='center', rotation='vertical')

    plt.axvline(x=weekly_freq * len(data), color='purple', linestyle='solid', linewidth=1)
    plt.text(weekly_freq * len(data), 1, 'Weekly', ha='center', va='center', rotation='vertical')

    plt.show()


def main():

    # FFT Time domain (1)   
    data4FFT = quandl.get('TFGRAIN/CORN', start_date="2000-12-1", end_date="2017-4-17", api_key='obLWrsHwLyxBs1bxiqBB')[
        'Cash Price']   
    # data4FFT = dt.DataReader("TFGRAIN/CORN", data_source='quandl', start=datetime.datetime(2000, 12, 1), end=datetime.datetime(2017, 4, 17), access_key= 'obLWrsHwLyxBs1bxiqBB')['Cash Price']
    roll(data4FFT)
    fft(data4FFT)

    # FFT Time domain (2)
    data4FFT = quandl.get('TFGRAIN/CORN', start_date="2016-4-9", end_date="2017-4-17", api_key='obLWrsHwLyxBs1bxiqBB')[
        'Cash Price']
    # data4FFT = dt.DataReader("TFGRAIN/CORN", data_source='quandl', start=datetime.datetime(2016, 4, 9), end=datetime.datetime(2017, 4, 17), access_key= 'obLWrsHwLyxBs1bxiqBB')['Cash Price']
    roll(data4FFT)
    fft(data4FFT)

    # FFT Time domain (3) 
    data4FFT = quandl.get('TFGRAIN/CORN', start_date="2016-10-11", end_date="2017-4-17", api_key='obLWrsHwLyxBs1bxiqBB')[
            'Cash Price']
    # data4FFT = dt.DataReader("TFGRAIN/CORN", data_source='quandl', start=datetime.datetime(2016, 10, 11), end=datetime.datetime(2017, 4, 17), access_key= 'obLWrsHwLyxBs1bxiqBB')['Cash Price']
    roll(data4FFT)
    fft(data4FFT)


# main of our program to prevent python script execution upon call
if __name__ == "__main__":
    main()
