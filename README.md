# 日付データと気象データからPyCaretで使用電力を予測

## 概要

2016年から2021年の東京電力管内の使用電力のデータと気象庁の気象データをもとに、使用電力を予測します。

機械学習には`PyCaret`を使用し、`LightGBM`で学習しています。

ブログ記事は[こちら](https://snova301.hatenablog.com/entry/2022/03/28/182458)。


## 参考サイト

- [PyCaret](https://pycaret.readthedocs.io/en/latest/index.html)
- [東京電力 - 過去の電力使用実績データのダウンロード](https://www.tepco.co.jp/forecast/html/download-j.html)
- [気象庁 - 過去の気象データ・ダウンロード](https://www.data.jma.go.jp/gmd/risk/obsdl/index.php)