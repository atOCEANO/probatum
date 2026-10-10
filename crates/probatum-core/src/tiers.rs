//! A venue's tier table for one instrument: up to each bound, the most a position
//! may be levered and the maintenance it keeps, the tier chosen by the position's
//! notional or, on a quantity table, by its size (Specification, Margin).

/// One row of a tier table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tier {
    /// The tier's upper bound, of notional or of size.
    pub up_to: f64,
    /// The most a position in the tier may be levered, or zero for no cap.
    pub leverage: f64,
    /// The maintenance rate.
    pub rate: f64,
    /// The amount a notional table deducts from the maintenance.
    pub amount: f64,
}

/// What a table's bounds measure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TierBasis {
    #[default]
    Notional,
    Quantity,
}

/// Why a tier table was refused, by row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TierError {
    /// A maintenance rate of one or more.
    Rate { row: usize, rate: f64 },
    /// A maintenance rate that reaches the initial rate, one over the leverage.
    Initial { row: usize, rate: f64, initial: f64 },
    /// A notional table's amount where continuity across the bound asks another.
    Amount {
        row: usize,
        amount: f64,
        wanted: f64,
    },
    /// A bound not above the one before it, or not above zero in the first row.
    Bound { row: usize, up_to: f64, below: f64 },
    /// A leverage above the one before it, 0.0 standing for no cap.
    Leverage {
        row: usize,
        leverage: f64,
        below: f64,
    },
    /// A maintenance rate below the one before it.
    Falling { row: usize, rate: f64, below: f64 },
    /// A value that is not a finite number.
    NotFinite {
        row: usize,
        field: &'static str,
        value: f64,
    },
    /// A leverage or a rate below zero.
    Negative {
        row: usize,
        field: &'static str,
        value: f64,
    },
}

/// A tier table, checked as the Specification's loader checks it. An empty table
/// stands for none: the venue's leverage, and no maintenance.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tiers {
    basis: TierBasis,
    rows: Vec<Tier>,
}

impl Tiers {
    /// The table of `rows` on `basis`, refused when a maintenance rate is not below
    /// one or reaches its tier's initial rate, when a notional table's amounts break
    /// continuity, `c_k = c_(k-1) + (m_k - m_(k-1)) * up_to_(k-1)` from a first amount
    /// of zero, and when the bounds do not rise, the leverage rises or the
    /// maintenance rate falls with them, since both requirements have to grow with
    /// the position, and wherever a number is not finite or a leverage or a rate is
    /// negative. A notional table keeps each amount at the value continuity sets,
    /// once the one given is within rounding of it, so the requirement meets itself
    /// at every bound.
    pub fn new(basis: TierBasis, mut rows: Vec<Tier>) -> Result<Tiers, TierError> {
        let cap = |leverage: f64| {
            if leverage > 0.0 {
                leverage
            } else {
                f64::INFINITY
            }
        };
        let mut previous: Option<Tier> = None;
        for (row, tier) in rows.iter_mut().enumerate() {
            let named = [
                ("up_to", tier.up_to),
                ("leverage", tier.leverage),
                ("rate", tier.rate),
                ("amount", tier.amount),
            ];
            for (field, value) in named {
                if !value.is_finite() {
                    return Err(TierError::NotFinite { row, field, value });
                }
            }
            for (field, value) in [("leverage", tier.leverage), ("rate", tier.rate)] {
                if value < 0.0 {
                    return Err(TierError::Negative { row, field, value });
                }
            }
            let below = previous.map_or(0.0, |previous| previous.up_to);
            if tier.up_to <= below {
                return Err(TierError::Bound {
                    row,
                    up_to: tier.up_to,
                    below,
                });
            }
            if let Some(previous) = previous {
                if cap(tier.leverage) > cap(previous.leverage) {
                    return Err(TierError::Leverage {
                        row,
                        leverage: tier.leverage,
                        below: previous.leverage,
                    });
                }
                if tier.rate < previous.rate {
                    return Err(TierError::Falling {
                        row,
                        rate: tier.rate,
                        below: previous.rate,
                    });
                }
            }
            if tier.rate >= 1.0 {
                return Err(TierError::Rate {
                    row,
                    rate: tier.rate,
                });
            }
            if tier.leverage > 0.0 && tier.rate >= 1.0 / tier.leverage {
                return Err(TierError::Initial {
                    row,
                    rate: tier.rate,
                    initial: 1.0 / tier.leverage,
                });
            }
            if basis == TierBasis::Notional {
                let wanted = previous.map_or(0.0, |previous| {
                    previous.amount + (tier.rate - previous.rate) * previous.up_to
                });
                if (tier.amount - wanted).abs() > 1e-9 * wanted.abs().max(1.0) {
                    return Err(TierError::Amount {
                        row,
                        amount: tier.amount,
                        wanted,
                    });
                }
                tier.amount = wanted;
            }
            previous = Some(*tier);
        }
        Ok(Tiers { basis, rows })
    }

    /// What the table's bounds measure.
    pub fn basis(&self) -> TierBasis {
        self.basis
    }

    /// The table's rows, empty when there is no table.
    pub fn rows(&self) -> &[Tier] {
        &self.rows
    }

    /// The tier a position of `size` and `notional` falls in, a bound belonging to
    /// the tier it closes and anything past the last bound to the last tier, or
    /// `None` when there is no table.
    pub fn tier(&self, size: f64, notional: f64) -> Option<&Tier> {
        let measure = match self.basis {
            TierBasis::Notional => notional,
            TierBasis::Quantity => size,
        };
        self.rows
            .iter()
            .find(|tier| measure <= tier.up_to)
            .or(self.rows.last())
    }
}

#[cfg(test)]
mod tests {
    use super::{Tier, TierBasis, TierError, Tiers};

    fn tier(up_to: f64, leverage: f64, rate: f64, amount: f64) -> Tier {
        Tier {
            up_to,
            leverage,
            rate,
            amount,
        }
    }

    #[test]
    fn binances_first_two_btc_tiers_are_continuous_at_300_000() {
        // the Specification's example: up to 300,000 at 0.4% with no amount, then
        // up to 800,000 at 0.5% with 300, both asking 1,200 at the bound
        let rows = vec![
            tier(300_000.0, 125.0, 0.004, 0.0),
            tier(800_000.0, 100.0, 0.005, 300.0),
        ];
        let tiers = Tiers::new(TierBasis::Notional, rows).expect("continuous");
        assert_eq!(tiers.tier(5.0, 300_000.0).map(|t| t.amount), Some(0.0));
        assert_eq!(tiers.tier(5.0, 300_001.0).map(|t| t.amount), Some(300.0));
        assert_eq!(tiers.tier(5.0, 9e9).map(|t| t.up_to), Some(800_000.0));
    }

    #[test]
    fn a_table_the_loader_cannot_trust_is_refused_at_its_row() {
        let continuous = |amount| {
            Tiers::new(
                TierBasis::Notional,
                vec![
                    tier(300_000.0, 125.0, 0.004, 0.0),
                    tier(800_000.0, 100.0, 0.005, amount),
                ],
            )
        };
        assert_eq!(
            continuous(250.0),
            Err(TierError::Amount {
                row: 1,
                amount: 250.0,
                wanted: 300.0
            })
        );
        let first = Tiers::new(
            TierBasis::Notional,
            vec![tier(300_000.0, 125.0, 0.004, 30.0)],
        );
        assert_eq!(
            first,
            Err(TierError::Amount {
                row: 0,
                amount: 30.0,
                wanted: 0.0
            })
        );
        let nan = Tiers::new(TierBasis::Quantity, vec![tier(10.0, 20.0, f64::NAN, 0.0)]);
        assert!(matches!(
            nan,
            Err(TierError::NotFinite {
                row: 0,
                field: "rate",
                ..
            })
        ));
        let negative = Tiers::new(TierBasis::Quantity, vec![tier(10.0, 20.0, -0.01, 0.0)]);
        assert_eq!(
            negative,
            Err(TierError::Negative {
                row: 0,
                field: "rate",
                value: -0.01
            })
        );
        let one = Tiers::new(TierBasis::Quantity, vec![tier(10.0, 0.0, 1.0, 0.0)]);
        assert_eq!(one, Err(TierError::Rate { row: 0, rate: 1.0 }));
        let initial = Tiers::new(TierBasis::Quantity, vec![tier(10.0, 20.0, 0.05, 0.0)]);
        assert_eq!(
            initial,
            Err(TierError::Initial {
                row: 0,
                rate: 0.05,
                initial: 0.05
            })
        );
    }

    #[test]
    fn a_notional_table_keeps_each_amount_at_the_value_continuity_sets() {
        let tiers = Tiers::new(
            TierBasis::Notional,
            vec![
                tier(300_000.0, 125.0, 0.004, 0.0),
                tier(800_000.0, 100.0, 0.005, 300.000_000_1),
            ],
        )
        .expect("within rounding of continuity");
        assert_eq!(tiers.rows()[1].amount, (0.005 - 0.004) * 300_000.0);
    }

    #[test]
    fn a_quantity_table_reads_the_size_and_none_reads_nothing() {
        let tiers = Tiers::new(
            TierBasis::Quantity,
            vec![tier(10.0, 50.0, 0.01, 0.0), tier(50.0, 20.0, 0.02, 0.0)],
        )
        .expect("checked");
        assert_eq!(tiers.tier(20.0, 1.0).map(|t| t.leverage), Some(20.0));
        assert_eq!(Tiers::default().tier(20.0, 1.0), None);
    }

    #[test]
    fn a_table_whose_requirement_would_not_grow_with_the_size_is_refused() {
        let quantity = |rows| Tiers::new(TierBasis::Quantity, rows);
        assert_eq!(
            quantity(vec![tier(0.0, 20.0, 0.0, 0.0)]),
            Err(TierError::Bound {
                row: 0,
                up_to: 0.0,
                below: 0.0
            })
        );
        assert_eq!(
            quantity(vec![tier(10.0, 20.0, 0.0, 0.0), tier(5.0, 10.0, 0.0, 0.0)]),
            Err(TierError::Bound {
                row: 1,
                up_to: 5.0,
                below: 10.0
            })
        );
        for higher in [20.0, 0.0] {
            assert_eq!(
                quantity(vec![
                    tier(10.0, 10.0, 0.0, 0.0),
                    tier(50.0, higher, 0.0, 0.0)
                ]),
                Err(TierError::Leverage {
                    row: 1,
                    leverage: higher,
                    below: 10.0
                })
            );
        }
        assert!(quantity(vec![tier(10.0, 0.0, 0.0, 0.0), tier(50.0, 10.0, 0.0, 0.0)]).is_ok());
        assert_eq!(
            quantity(vec![
                tier(10.0, 20.0, 0.02, 0.0),
                tier(50.0, 10.0, 0.01, 0.0)
            ]),
            Err(TierError::Falling {
                row: 1,
                rate: 0.01,
                below: 0.02
            })
        );
    }
}
