# generates buy/sell/hold signals from sentiment scores

def get_signal(score):
    """
    Takes a sentiment score in [-1, +1] and returns a trading signal.
    Scores above 0.2 suggest positive sentiment — buy signal.
    Scores below -0.2 suggest negative sentiment — sell or sit out.
    """
    if score > 0.2:
        return 'BUY'
    elif score < -0.2:
        return 'SELL'
    else:
        return 'HOLD'


def generate_signals(scores):
    """
    Applies get_signal to a pandas Series of sentiment scores.
    Returns a Series of BUY/SELL/HOLD strings.
    """
    return scores.apply(get_signal)