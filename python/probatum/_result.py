"""What one run produced, whatever its policy was."""

import numpy as np


def _frozen(array):
    array = np.asarray(array)
    array.flags.writeable = False
    return array


class Result:
    """What one run recorded, on the rows its account saw from the first: ``time``,
    each row's open in int64 nanoseconds since 1970, with ``equity`` and ``returns``,
    each ``(T,)``; ``ledger``, ``fills``, ``orders`` and ``trades``, the ledger, the
    fill log, the order log and the round trips; ``stats``, the statistics at the
    Data's annualisation and the run's risk-free rate, None where the clock is too
    short to annualise; ``policy``, the copy of the policy the run used; ``bust``,
    whether the account's equity reached zero; ``liquidations``, the number of bars it
    was liquidated on; and ``provenance``, what the run was given. A row's return is
    its equity over the row before's, less one, and nothing once the equity is gone."""


    def __init__(self, ledger, fills, orders, trades, stats, policy, bust, provenance):
        self.ledger     = ledger
        self.fills      = fills
        self.orders     = orders
        self.trades     = trades
        self.stats      = stats
        self.policy     = policy
        self.bust       = bool(bust)
        self.provenance = provenance

        self.time   = _frozen(ledger["time"])
        self.equity = _frozen(ledger["equity"])
        before      = self.equity[:-1]
        grown       = np.divide(self.equity[1:], before, out=np.ones_like(before),
                                where=before > 0.0)
        self.returns      = _frozen(np.concatenate([[0.0], grown - 1.0]))
        self.liquidations = len({fill["bar"] for fill in fills
                                 if fill["reason"] == "liquidation"})


    def __repr__(self):
        given = self.provenance
        return (f"Result({given['policy']} on {given['venue']}, {len(self.time):,} "
                f"bars, equity {self.equity[0]:,.2f} to {self.equity[-1]:,.2f})")
