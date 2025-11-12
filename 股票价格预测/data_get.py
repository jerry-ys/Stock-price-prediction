import yfinance as yf
import datetime

# 设置股票代码和日期范围
ticker = 'TSLA'
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5*365)  # 过去5年

# 下载数据
tesla_data = yf.download(ticker, start=start_date, end=end_date,progress=True)

# 显示数据
print(tesla_data)


tesla_data.to_csv('TSLA.csv')
