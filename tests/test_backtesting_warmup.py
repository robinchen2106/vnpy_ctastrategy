import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ
from vnpy.trader.object import BarData

from vnpy_ctastrategy.backtesting import BacktestingEngine


VT_SYMBOL = "000001.SZSE"


def make_bar(dt: datetime, close: float) -> BarData:
    return BarData(
        gateway_name="TEST",
        symbol="000001",
        exchange=Exchange.SZSE,
        datetime=dt,
        interval=Interval.DAILY,
        open_price=close,
        high_price=close,
        low_price=close,
        close_price=close,
    )


class DummyStrategy:
    def __init__(self) -> None:
        self.inited = False
        self.trading = False
        self.bar_dates: list[datetime] = []
        self.events: list[str] = []

    def on_init(self) -> None:
        self.events.append("init")

    def on_start(self) -> None:
        self.events.append("start")

    def on_stop(self) -> None:
        self.events.append("stop")

    def on_timer(self, dt: datetime) -> None:
        return

    def on_bar(self, bar: BarData) -> None:
        self.bar_dates.append(bar.datetime)


class BacktestingWarmupTest(unittest.TestCase):
    def test_warmup_parameters_change_load_bar_range(self) -> None:
        warmup = datetime(2024, 1, 1, tzinfo=DB_TZ)
        start = datetime(2024, 1, 11, tzinfo=DB_TZ)
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol=VT_SYMBOL,
            interval=Interval.DAILY,
            rate=0.0,
            slippage=0.0,
            size=1.0,
            pricetick=0.01,
            end=datetime(2024, 1, 31),
            warmup=warmup,
            start=start,
        )

        with patch("vnpy_ctastrategy.backtesting.load_bar_data", return_value=[]) as loader:
            engine.load_bar(
                VT_SYMBOL,
                days=1,
                interval=Interval.DAILY,
                callback=lambda bar: None,
                use_database=False,
            )

        self.assertEqual(loader.call_args.args[3], warmup)
        self.assertEqual(loader.call_args.args[4], start - timedelta(days=1))

    def test_warmup_skips_matching_and_daily_results(self) -> None:
        warmup_dt = datetime(2024, 1, 1, tzinfo=DB_TZ)
        trading_dt = datetime(2024, 1, 2, tzinfo=DB_TZ)
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol=VT_SYMBOL,
            interval=Interval.DAILY,
            rate=0.0,
            slippage=0.0,
            size=1.0,
            pricetick=0.01,
            end=datetime(2024, 1, 5),
            warmup=warmup_dt,
            start=trading_dt,
        )
        engine.history_data = [make_bar(warmup_dt, 100), make_bar(trading_dt, 120)]
        strategy = DummyStrategy()
        engine.strategy = strategy

        cross_calls: list[None] = []
        engine.cross_limit_order = lambda: cross_calls.append(None)

        engine.run_backtesting()

        self.assertEqual(strategy.bar_dates, [warmup_dt, trading_dt])
        self.assertEqual(len(cross_calls), 1)
        self.assertEqual(list(engine.daily_results), [trading_dt.date()])

        engine.calculate_result()
        statistics = engine.calculate_statistics(output=False)
        self.assertEqual(statistics["benchmark_return"], 0)

    def test_equal_warmup_and_start_skips_warmup(self) -> None:
        start = datetime(2024, 1, 2)
        bar_dt = start.replace(tzinfo=DB_TZ)
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol=VT_SYMBOL,
            interval=Interval.DAILY,
            rate=0.0,
            slippage=0.0,
            size=1.0,
            pricetick=0.01,
            end=datetime(2024, 1, 5),
            warmup=start,
            start=start,
        )
        engine.history_data = [make_bar(bar_dt, 100)]
        engine.strategy = DummyStrategy()

        cross_calls: list[None] = []
        engine.cross_limit_order = lambda: cross_calls.append(None)

        engine.run_backtesting()

        self.assertEqual(engine.strategy.bar_dates, [bar_dt])
        self.assertEqual(len(cross_calls), 1)
        self.assertEqual(list(engine.daily_results), [start.date()])


if __name__ == "__main__":
    unittest.main()
