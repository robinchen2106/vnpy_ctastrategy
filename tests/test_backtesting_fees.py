import unittest
from datetime import date, datetime

from pandas import DataFrame
from vnpy.trader.constant import Direction, Exchange, Interval, Offset
from vnpy.trader.object import TradeData

from vnpy_ctastrategy.backtesting import BacktestingEngine, DailyResult


class BacktestingFeeModelTest(unittest.TestCase):
    def make_trade(
        self,
        trade_id: str,
        direction: Direction,
        price: float,
        volume: float,
    ) -> TradeData:
        return TradeData(
            gateway_name="TEST",
            symbol="000001",
            exchange=Exchange.SZSE,
            orderid=f"order-{trade_id}",
            tradeid=trade_id,
            direction=direction,
            offset=Offset.OPEN,
            price=price,
            volume=volume,
            datetime=datetime(2024, 1, 2),
        )

    def test_commission_stamp_tax_and_percentage_slippage(self) -> None:
        result = DailyResult(datetime(2024, 1, 2).date(), 100.0)
        result.add_trade(self.make_trade("buy", Direction.LONG, 100.0, 1000))
        result.add_trade(self.make_trade("sell", Direction.SHORT, 100.0, 1000))

        result.calculate_pnl(
            pre_close=100.0,
            start_pos=0.0,
            size=1.0,
            rate=0.0001,
            slippage=0.2,
            min_commission=5.0,
            stamp_duty=0.0005,
            slippage_rate=0.0003,
        )

        self.assertEqual(result.turnover, 200_000.0)
        self.assertEqual(result.broker_commission, 20.0)
        self.assertEqual(result.stamp_tax, 50.0)
        self.assertEqual(result.stamp_duty, 50.0)
        self.assertEqual(result.commission, 70.0)
        self.assertEqual(result.slippage, 460.0)
        self.assertEqual(result.net_pnl, -530.0)

    def test_minimum_commission_and_canonical_parameter_names(self) -> None:
        engine = BacktestingEngine()
        engine.set_parameters(
            vt_symbol="000001.SZSE",
            interval=Interval.DAILY,
            start=datetime(2024, 1, 1),
            rate=0.0001,
            slippage=0.0,
            size=1.0,
            pricetick=0.01,
            minimum_commission=5.0,
            stamp_tax_rate=0.0005,
        )

        self.assertEqual(engine.min_commission, 5.0)
        self.assertEqual(engine.minimum_commission, 5.0)
        self.assertEqual(engine.stamp_duty, 0.0005)
        self.assertEqual(engine.stamp_tax_rate, 0.0005)
        self.assertEqual(engine.slippage_rate, 0.0003)

    def test_fixed_and_rate_slippage_are_combined(self) -> None:
        result = DailyResult(datetime(2024, 1, 2).date(), 100.0)
        result.add_trade(self.make_trade("fixed", Direction.LONG, 100.0, 1000))

        result.calculate_pnl(
            pre_close=100.0,
            start_pos=0.0,
            size=1.0,
            rate=0.0,
            slippage=0.2,
            slippage_rate=0.0003,
        )

        self.assertEqual(result.slippage, 230.0)

    def test_statistics_only_return_current_stamp_tax_fields(self) -> None:
        engine = BacktestingEngine()
        df = DataFrame(
            {
                "net_pnl": [-100.0, 200.0],
                "commission": [7.0, 8.0],
                "broker_commission": [2.0, 3.0],
                "stamp_tax": [5.0, 5.0],
                "slippage": [1.0, 1.0],
                "turnover": [10_000.0, 20_000.0],
                "trade_count": [1, 2],
            },
            index=[date(2024, 1, 2), date(2024, 1, 3)],
        )

        statistics = engine.calculate_statistics(df, output=False)

        self.assertEqual(statistics["total_broker_commission"], 5.0)
        self.assertEqual(statistics["total_stamp_tax"], 10.0)
        self.assertEqual(statistics["total_commission"], 15.0)
        self.assertEqual(statistics["total_transaction_cost"], 17.0)
        self.assertNotIn("total_stamp_duty", statistics)
        self.assertNotIn("daily_stamp_duty", statistics)


if __name__ == "__main__":
    unittest.main()
