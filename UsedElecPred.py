import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pycaret.regression import *
import jpholiday


class ClassGetData:
  # データ保存場所
  foldername = 'data/'

  def __init__(self, str_year):
    self.str_year = str_year

  # 電力使用量
  def load_used_elec(self):
    elec_filename = self.foldername + 'TEPCO' + self.str_year + '.csv'
    csvreader_elec = pd.read_csv(elec_filename, delimiter=',', dtype='str', skiprows=2)[1:]
    return csvreader_elec


  # 電力使用量
  def get_used_elec(self):
    # データ読込
    csvreader_elec = self.load_used_elec()

    # データ列の抜出
    dat_elec = csvreader_elec['実績(万kW)']
    pd_dat_elec = pd.DataFrame({'Used Elec (x10 MW)' : dat_elec})
    pd_dat_elec = pd_dat_elec.reset_index(drop=True)
    return pd_dat_elec


  # 気温
  def get_temp_data(self):
    # データ読込
    temp_filename = self.foldername + 'Tmp' + self.str_year + '.csv'
    csvreader_temp = pd.read_csv(temp_filename, encoding='Shift-JIS' ,delimiter=',', dtype='str', skiprows=4)

    # データ列の抜出
    dat_temp = csvreader_temp['Unnamed: 1']
    pd_dat_temp = pd.DataFrame({'Temp (Deg)' : dat_temp[:-1]})
    return pd_dat_temp


  # 風速
  def get_wind_data(self):
    # データ読込
    wind_filename = self.foldername + 'Wind' + self.str_year + '.csv'
    csvreader_wind = pd.read_csv(wind_filename, encoding='Shift-JIS' ,delimiter=',', dtype='str', skiprows=5)

    # データ列の抜出
    dat_wind = csvreader_wind['Unnamed: 1']
    pd_dat_wind = pd.DataFrame({'Wind (m/s)' : dat_wind[:-1]})
    return pd_dat_wind


  # 天気
  def get_weather_data(self):
    # データ読込
    weather_filename = self.foldername + 'We' + self.str_year + '.csv'
    csvreader_weather = pd.read_csv(weather_filename, encoding='Shift-JIS' ,delimiter=',', dtype='str', skiprows=4)

    # データ列の抜出
    dat_weather = csvreader_weather['Unnamed: 1']
    pd_dat_weather = pd.DataFrame({'Weather' : dat_weather[:-1]}, dtype='float')

    # 補完
    pd_dat_weather = pd_dat_weather.interpolate('ffill').interpolate('bfill')
    return pd_dat_weather


  # 日付データ分割と曜日祝日取得
  def get_date_data(self):
    # データ読込
    csvreader_elec = self.load_used_elec()
    dt_date = pd.to_datetime(list(csvreader_elec['DATE']))
    pd_dt_date = pd.Series(dt_date)

    # 曜日の取得
    dt_dayofweek = dt_date.dayofweek
    pd_dt_dayofweek = pd.Series(dt_dayofweek)

    # 祝日の取得
    list_holiday = [jpholiday.is_holiday(i) for i in dt_date]
    list_holiday = ['1' if i == True else '0' for i in list_holiday]

    # DataFrameに変換
    pd_dt_ymdw = pd.DataFrame({'year' : pd_dt_date.dt.year,
                               'month' : pd_dt_date.dt.month,
                               'day' : pd_dt_date.dt.day,
                               'DoW' : pd_dt_dayofweek,
                               'holiday' : list_holiday})
    return pd_dt_ymdw


  # 時刻データの取得
  def get_time_data(self):
    # データ読込
    csvreader_elec = self.load_used_elec()
    dt_time = pd.to_datetime(list(csvreader_elec['TIME']))

    # DataFrameに変換
    pd_dt_time = pd.DataFrame({'time' : pd.Series(dt_time).dt.hour})
    return pd_dt_time


  def get_data(self):
    # データ取得
    pd_dat_elec = self.get_used_elec()
    pd_dat_temp = self.get_temp_data()
    pd_dat_wind = self.get_wind_data()
    pd_dat_weather = self.get_weather_data()
    pd_dt_ymdw = self.get_date_data()
    pd_dt_time = self.get_time_data()

    # データの結合
    pd_dt_EYMTWDTW = pd.concat([pd_dat_elec, pd_dt_ymdw, pd_dt_time, pd_dat_temp, pd_dat_wind, pd_dat_weather], axis = 1)
    return pd_dt_EYMTWDTW

  # tsvデータ保存
  def save_tsv(self, save_pd_dt):
    savefilename = self.foldername + 'data.tsv'
    save_pd_dt.to_csv(savefilename, sep = '\t', index = False)


class ClassLearning:
  # データ保存場所
  foldername = 'data/'


  # tsvデータのロード
  def load_data(self):
    filename = self.foldername + 'data.tsv'
    tsvreader_data = pd.read_csv(filename, delimiter='\t', dtype='float')
    return tsvreader_data


  def data_split(self, tsvreader_data):
    train_test_indexlist = np.arange(11000, 11200)
    train_data = tsvreader_data.drop(train_test_indexlist)
    test_data = tsvreader_data.loc[train_test_indexlist]
    return train_data, test_data


  def pycaret_learning(self, tsvreader_data):
    exp = setup(tsvreader_data, target='Used Elec (x10 MW)')
    # compare_models()
    pycaret_create_model = create_model('lightgbm')
    pycaret_tune_model = tune_model(lgb)
    # pycaret_final_model = finalize_model(pycaret_tune_model)
    evaluate_model(pycaret_final_model)
    return pycaret_tune_model

  # pickelで学習モデル保存
  def save_learningmodel(self, model, modelname):
    savename_model = self.foldername + modelname + 'model.pkl'
    pickle.dump(model, open(savename_model, 'wb'))


  # pickelで学習モデルロード
  def load_learningmodel(self, modelname):
    loadname_model = self.foldername + modelname + 'model.pkl'
    model = pickle.load(open(loadname_model, 'rb'))
    return model


  def pycaret_predplot(self, pycaret_final_model, test_data):
    # 予測
    pred_ = predict_model(pycaret_final_model, data = test_data)

    # 図示
    fig, ax = plt.subplots()
    ax.plot(pred_['Used Elec (x10 MW)'], color='red', label='Prediction Value')
    ax.plot(pred_['Label'], color='blue', label='True Value')
    ax.axes.xaxis.set_visible(False)
    ax.legend()


  def learning_main(self):
    # 学習データの準備
    tsvreader_data = self.load_data()
    print(tsvreader_data)
    train_data, test_data = self.data_split(tsvreader_data)

    # pycaret学習
    pycaret_final_model = self.pycaret_learning(train_data)

    # pycaret予測
    self.pycaret_predplot(pycaret_final_model, test_data)

    # 学習モデルの保存
    self.save_learningmodel(pycaret_final_model, 'pycaret')

    # 学習モデルの読込
    pred_ = self.load_learningmodel('pycaret')


def main():
  pd_dt4save = pd.DataFrame()
  for int_year in range(2016, 2022):
    # データ年の設定
    str_year = str(int_year)
    print(str_year)
    class_get_data = ClassGetData(str_year)

    # データ読込と編集
    pd_dt_EYMTWDTW = class_get_data.get_data()

    # データ保管
    pd_dt4save = pd.concat([pd_dt4save, pd_dt_EYMTWDTW])

  # データ保存
  print(pd_dt4save)
  ClassGetData('2022').save_tsv(pd_dt4save)
  ClassLearning().learning_main()


if __name__ == '__main__':
  main()
