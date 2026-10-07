//! Small plain enums shared across the engine: order side, type, time in force,
//! market kind and wallet.

/// The direction of an order or a position change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// `+1.0` for a buy, `-1.0` for a sell: the sign a fill applies to a signed
    /// position.
    #[inline]
    pub const fn sign(self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }

    /// The other side, whose `sign()` is the negation of this one's.
    #[inline]
    pub const fn opposite(self) -> Side {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
}

/// How an order is matched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderType {
    /// Crosses the spread, fills at the next bar open, pays the taker fee.
    Market,
    /// Rests at a price, fills only when a bar reaches it, pays the maker fee.
    Limit,
    /// Becomes a market order once the trigger price is crossed.
    Stop,
}

/// How long an order stays live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good til canceled: rests until filled or canceled.
    Gtc,
    /// Immediate or cancel: fills what it can now, cancels the rest.
    Ioc,
    /// Fill or kill: fills completely now, or cancels entirely.
    Fok,
}

/// The market a position trades in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Market {
    /// Spot: hold the base asset; equity is quote plus base times price.
    Spot,
    /// Perpetual future: a signed position with margin, no expiry.
    Perp,
}

impl Market {
    /// The wallet a position in this market settles in: spot holdings in the spot
    /// wallet, a perpetual against the margin wallet (ADR 0008).
    #[inline]
    pub const fn wallet(self) -> Wallet {
        match self {
            Market::Spot => Wallet::Spot,
            Market::Perp => Wallet::Margin,
        }
    }
}

/// One of the account's two wallets (ADR 0008).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wallet {
    /// Cash and spot holdings. It never borrows and is never liquidated.
    Spot,
    /// The collateral behind the perpetuals, in cross margin.
    Margin,
}

impl Wallet {
    /// The other wallet, the one a transfer into this one comes out of.
    #[inline]
    pub const fn other(self) -> Wallet {
        match self {
            Wallet::Spot => Wallet::Margin,
            Wallet::Margin => Wallet::Spot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Side;

    #[test]
    fn side_sign() {
        assert_eq!(Side::Buy.sign(), 1.0);
        assert_eq!(Side::Sell.sign(), -1.0);
    }

    #[test]
    fn side_opposite() {
        assert_eq!(Side::Buy.opposite(), Side::Sell);
        assert_eq!(Side::Sell.opposite(), Side::Buy);
        assert_eq!(Side::Buy.opposite().opposite(), Side::Buy);
    }
}
