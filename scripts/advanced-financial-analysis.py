# advanced_financial_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ccxt
import talib

class AdvancedFinancialAnalysis:
    def __init__(self):
        self.data = None

    def load_data(self, symbol, exchange_id='binance', timeframe='1d', since=None, limit=None):
        """
        Загрузка данных криптовалюты с биржи.
        """
        exchange = getattr(ccxt, exchange_id)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        self.data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='ms')
        self.data.set_index('timestamp', inplace=True)
        return self.data

    def calc_volatility(self, window=30):
        """
        Расчет волатильности с использованием GARCH-подобного подхода.
        """
        returns = self.data['close'].pct_change().dropna()
        alpha, beta = 0.05, 0.9
        omega = (1 - alpha - beta) * returns.var()
        sigma2 = returns.ewm(alpha=alpha).var()
        for t in range(1, len(returns)):
            sigma2.iloc[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2.iloc[t-1]
        return np.sqrt(sigma2)

    def calc_asymmetry(self, q):
        """
        Расчет асимметрии для мультифрактального анализа.
        """
        return 1 + 0.1 * np.sign(q - 1) * np.log(np.abs(q))

    def multifractal_spectrum(self, q_range=np.arange(-5, 5, 0.5), r=0.1):
        """
        Расчет мультифрактального спектра.
        """
        prices = self.data['close'].values
        lambda_x, lambda_y = 1.0, 1.0
        volatility = self.calc_volatility()
        
        tau_q = []
        for q in q_range:
            N_q = np.sum(np.abs(np.diff(prices))**q)
            eta = self.calc_asymmetry(q)
            tau = (np.log(N_q) / np.log(1/np.sqrt(lambda_x * lambda_y) * r)) * (q - 1) * volatility.mean() * eta
            tau_q.append(tau)
        
        return q_range, np.array(tau_q)

    def correlation_dimension(self, epsilon=0.1):
        """
        Расчет корреляционной размерности.
        """
        prices = self.data['close'].values
        lambda_x, lambda_y = 1.0, 1.0
        N = len(prices)
        C = np.sum([np.sum(np.abs(prices[i] - prices[j]) < epsilon) for i in range(N) for j in range(i+1, N)])
        C = 2 * C / (N * (N - 1))
        return np.log(C) / np.log(1 / np.sqrt(lambda_x * lambda_y) * epsilon)

    def hurst_index(self):
        """
        Расчет индекса Херста с учетом асимметрии.
        """
        prices = self.data['close'].values
        lambda_t = 1.0
        N = len(prices)
        y = np.cumsum(prices - np.mean(prices))
        R = np.max(y) - np.min(y)
        S = np.std(prices)
        kappa = 1 + 0.1 * np.log(N)  # Учет асимметрии и временных зависимостей
        return (R / S) / np.power(lambda_t * N, 0.5) * kappa

    def mf_dfa(self, q_range=np.arange(-5, 5, 1), scale_range=np.logspace(1, 3, 20, base=2).astype(int)):
        """
        Мультифрактальный детрендированный флуктуационный анализ (MF-DFA).
        """
        y = np.cumsum(self.data['close'].pct_change().dropna())
        
        F_q = np.zeros((len(q_range), len(scale_range)))
        for i, scale in enumerate(scale_range):
            segments = len(y) // scale
            F_q_scale = np.zeros(len(q_range))
            for start in range(0, segments * scale, scale):
                segment = y[start:start+scale]
                coef = np.polyfit(np.arange(scale), segment, 1)
                trend = np.polyval(coef, np.arange(scale))
                F_squared = np.mean((segment - trend)**2)
                F_q_scale += np.power(F_squared, q_range/2)
            F_q[:, i] = np.power(F_q_scale / segments, 1/q_range)
        
        return q_range, scale_range, F_q

    def find_similar_pattern(self, pattern, threshold=0.9):
        """
        Поиск схожих паттернов в исторических данных.
        """
        pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern))
        data = self.data['close'].values
        data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        similarities = []
        for i in range(len(data) - len(pattern)):
            segment = data_normalized[i:i+len(pattern)]
            similarity = 1 - np.sqrt(np.mean((segment - pattern)**2))
            similarities.append(similarity)
        
        best_match = np.argmax(similarities)
        if similarities[best_match] >= threshold:
            return best_match, similarities[best_match]
        return None, None

    def forecast_pattern(self, pattern_length=50, forecast_length=20):
        """
        Прогнозирование на основе поиска схожих паттернов.
        """
        current_pattern = self.data['close'].values[-pattern_length:]
        match_index, similarity = self.find_similar_pattern(current_pattern)
        
        if match_index is not None:
            forecast = self.data['close'].values[match_index+pattern_length:match_index+pattern_length+forecast_length]
            return forecast, similarity
        return None, None

    def optimize_portfolio(self, returns, risk_free_rate=0.02):
        """
        Оптимизация портфеля с использованием модели Блэка-Литтермана.
        """
        n = returns.shape[1]
        mu = returns.mean()
        sigma = returns.cov()
        
        def objective(weights):
            portfolio_return = np.sum(mu * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            return -(portfolio_return - risk_free_rate) / portfolio_volatility

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        result = minimize(objective, n*[1./n], method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

    def backtest_strategy(self, strategy_func, initial_capital=10000):
        """
        Бэктестинг торговой стратегии.
        """
        portfolio = initial_capital
        positions = np.zeros(len(self.data))
        
        for i in range(1, len(self.data)):
            signal = strategy_func(self.data.iloc[:i])
            if signal == 1 and positions[i-1] == 0:
                positions[i] = portfolio / self.data['close'].iloc[i]
                portfolio = 0
            elif signal == -1 and positions[i-1] > 0:
                portfolio = positions[i-1] * self.data['close'].iloc[i]
                positions[i] = 0
            else:
                positions[i] = positions[i-1]
        
        final_portfolio = portfolio + positions[-1] * self.data['close'].iloc[-1]
        returns = (final_portfolio - initial_capital) / initial_capital
        sharpe_ratio = returns / (self.data['close'].pct_change().std() * np.sqrt(252))
        
        return {
            'Final Portfolio Value': final_portfolio,
            'Total Return': returns,
            'Sharpe Ratio': sharpe_ratio
        }

    def plot_multifractal_spectrum(self, q_range, tau_q):
        """
        Визуализация мультифрактального спектра.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(q_range, tau_q)
        plt.title('Multifractal Spectrum')
        plt.xlabel('q')
        plt.ylabel('tau(q)')
        plt.grid(True)
        plt.show()

    def plot_mfdfa(self, q_range, scale_range, F_q):
        """
        Визуализация результатов MF-DFA.
        """
        plt.figure(figsize=(12, 8))
        for i, q in enumerate(q_range):
            plt.loglog(scale_range, F_q[i], label=f'q={q}')
        plt.title('Multifractal Detrended Fluctuation Analysis')
        plt.xlabel('Scale')
        plt.ylabel('F(q)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_forecast(self, forecast, actual):
        """
        Визуализация прогноза и фактических данных.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual')
        plt.plot(np.arange(len(actual), len(actual) + len(forecast)), forecast, label='Forecast')
        plt.title('Price Forecast')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

# Пример использования
if __name__ == "__main__":
    analysis = AdvancedFinancialAnalysis()
    data = analysis.load_data('BTC/USDT', limit=1000)
    
    # Мультифрактальный анализ
    q_range, tau_q = analysis.multifractal_spectrum()
    analysis.plot_multifractal_spectrum(q_range, tau_q)
    
    # MF-DFA
    q_range, scale_range, F_q = analysis.mf_dfa()
    analysis.plot_mfdfa(q_range, scale_range, F_q)
    
    # Прогнозирование
    forecast, similarity = analysis.forecast_pattern()
    if forecast is not None:
        analysis.plot_forecast(forecast, data['close'].values[-len(forecast):])
    
    # Бэктестинг простой стратегии
    def simple_strategy(data):
        if data['close'].iloc[-1] > data['close'].iloc[-2]:
            return 1
        return -1
    
    results = analysis.backtest_strategy(simple_strategy)
    print("Backtest Results:", results)
