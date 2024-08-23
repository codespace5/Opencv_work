from ib_insync import *

# Connect to the IB server
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # clientId should be different for every script you run

# Define a contract
contract = Stock('AAPL', 'SMART', 'USD')

# Specify the order type: LMT means "Limit"
# Change 'lmtPrice' to the price at which you want to buy the contract
order = LimitOrder('BUY', 100, lmtPrice=200.00)

# Place the order
trade = ib.placeOrder(contract, order)

# Define your target profit and stop loss percentages
profit_target_percentage = 50 / 100
stop_loss_percentage = 25 / 100

# Calculate the take profit and stop loss prices
take_profit_price = order.lmtPrice * (1 + profit_target_percentage)
stop_loss_price = order.lmtPrice * (1 - stop_loss_percentage)

# Create the take profit order
take_profit_order = LimitOrder('SELL', order.totalQuantity, lmtPrice=take_profit_price)
trade_take_profit = ib.placeOrder(contract, take_profit_order)

# Create the stop loss order
stop_loss_order = StopOrder('SELL', order.totalQuantity, stopPrice=stop_loss_price)
trade_stop_loss = ib.placeOrder(contract, stop_loss_order)

# Disconnect from the server
ib.disconnect()



/////////////////////////////////

from ib_insync import *

# Connect to the IB server
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # clientId should be different for every script you run

# Define a contract
contract = Stock('AAPL', 'SMART', 'USD')

# Specify the order type: LMT means "Limit"
# Change 'lmtPrice' to the price at which you want to buy the contract
order = LimitOrder('BUY', 100, lmtPrice=200.00)

# Place the order
trade = ib.placeOrder(contract, order)

# Define your target profit and stop loss percentages
profit_target_percentage = 50 / 100
stop_loss_percentage = 25 / 100

# Calculate the take profit and stop loss prices
take_profit_price = order.lmtPrice * (1 + profit_target_percentage)
stop_loss_price = order.lmtPrice * (1 - stop_loss_percentage)

# Create the take profit order
take_profit_order = LimitOrder('SELL', order.totalQuantity, lmtPrice=take_profit_price)
trade_take_profit = ib.placeOrder(contract, take_profit_order)

# Create the stop loss order
stop_loss_order = StopOrder('SELL', order.totalQuantity, stopPrice=stop_loss_price)
trade_stop_loss = ib.placeOrder(contract, stop_loss_order)

# Disconnect from the server
ib.disconnect()


