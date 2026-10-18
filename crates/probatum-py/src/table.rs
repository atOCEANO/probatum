//! A run's records as tables any Arrow reader takes through the C stream interface, so
//! polars, pyarrow and DuckDB read them without probatum importing one (ADR 0026). The
//! table and its export are probatum-core's; this hands the export to Python in a
//! capsule, reads its columns as numpy arrays, and builds the tables from the engine's
//! records.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use numpy::ndarray::ArrayView1;
use numpy::npyffi::flags::NPY_ARRAY_WRITEABLE;
use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList};

use probatum_core::arrow::{
    self, bit, ArrowArrayStream, ArrowSchema, Column, Given, Table, Values,
};
use probatum_core::Side;
use probatum_engine::reporter::{FillRecord, LedgerRow};
use probatum_engine::{Engine as BarEngine, RoundTrip, Units};

use crate::arrow::text;
use crate::{booleans, reason_text, side_str};

/// The capsule's destructor, which frees the stream it holds, read under whatever name
/// the capsule carries now.
///
/// # Safety
/// `capsule` is a capsule `stream_capsule` made.
unsafe extern "C" fn drop_stream(capsule: *mut pyo3::ffi::PyObject) {
    let name = pyo3::ffi::PyCapsule_GetName(capsule);
    let stream = pyo3::ffi::PyCapsule_GetPointer(capsule, name) as *mut ArrowArrayStream;
    if stream.is_null() {
        pyo3::ffi::PyErr_Clear();
        return;
    }
    arrow::free_stream(stream);
}

fn stream_capsule(py: Python<'_>, table: Arc<Table>) -> PyResult<Bound<'_, PyCapsule>> {
    let stream = Box::into_raw(Box::new(arrow::stream(table)));
    // SAFETY: the stream is a live box, which the capsule's destructor releases and
    // frees, and the name is static
    let made = unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            NonNull::new_unchecked(stream.cast::<c_void>()),
            c"arrow_array_stream",
            Some(drop_stream),
        )
    };
    if made.is_err() {
        // SAFETY: no capsule holds the stream, so it is still this function's
        unsafe { arrow::free_stream(stream) };
    }
    made
}

/// Refuse a requested schema that is no schema, or names other columns than
/// `table`'s, in their order; one that names the same columns gets the table's own
/// schema back, which the PyCapsule interface allows. The caller's capsule is read
/// where it is and never moved or released.
fn check_requested(table: &Table, requested: &Bound<'_, PyAny>) -> PyResult<()> {
    let pointer = requested
        .cast::<PyCapsule>()
        .ok()
        .and_then(|capsule| capsule.pointer_checked(Some(c"arrow_schema")).ok());
    let Some(pointer) = pointer else {
        return Err(PyTypeError::new_err(format!(
            "requested_schema: a {}, not a capsule named arrow_schema; pass one, or None",
            requested.get_type().name()?
        )));
    };
    // SAFETY: a capsule named arrow_schema holds a schema of the C data interface
    let schema = unsafe { &*(pointer.as_ptr() as *const ArrowSchema) };
    if schema.release.is_none() {
        return Err(PyValueError::new_err(
            "requested_schema: a schema already released; pass a live one, or None",
        ));
    }
    // SAFETY: a live schema's format is a NUL-terminated string and its children are
    // n_children live schemas
    let asked: Vec<String> = unsafe {
        if text(schema.format) != "+s" {
            vec![]
        } else {
            (0..usize::try_from(schema.n_children).unwrap_or(0))
                .map(|i| text((**schema.children.add(i)).name))
                .collect()
        }
    };
    let names = names_of(table);
    if asked != names {
        return Err(PyValueError::new_err(format!(
            "requested_schema: the columns {}, and this table holds {}; request its own \
             columns, or None",
            if asked.is_empty() {
                "of no table".to_string()
            } else {
                asked.join(", ")
            },
            names.join(", ")
        )));
    }
    Ok(())
}

fn names_of(table: &Table) -> Vec<String> {
    table
        .columns()
        .iter()
        .map(|column| column.name().to_string_lossy().into_owned())
        .collect()
}

/// `array`, made read-only, since a reader may hold the same buffer through a stream
/// and every array the package hands out is read-only.
fn frozen<T: numpy::Element>(array: Bound<'_, PyArray1<T>>) -> Bound<'_, PyArray1<T>> {
    // SAFETY: the array was just made and nothing else holds it yet, so clearing its
    // WRITEABLE flag races with nothing
    unsafe {
        (*array.as_array_ptr()).flags &= !(NPY_ARRAY_WRITEABLE as std::os::raw::c_int);
    }
    array
}

/// A read-only numpy view of `values`, which `owner` keeps alive.
fn view<'py, T: numpy::Element>(values: &[T], owner: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
    let view = ArrayView1::from(values);
    // SAFETY: `values` lives in a table that never changes, held by `owner`, which
    // becomes the array's base
    frozen(unsafe { PyArray1::borrow_from_array(&view, owner) }).into_any()
}

/// The bits of `bits` for `rows` rows as a read-only boolean array.
fn unpacked<'py>(py: Python<'py>, bits: &[u8], rows: usize) -> Bound<'py, PyAny> {
    frozen(PyArray1::from_vec(
        py,
        (0..rows).map(|row| bit(bits, row)).collect(),
    ))
    .into_any()
}

/// The value of `row` of `column` as a Python object, None for a null.
fn value<'py>(py: Python<'py>, column: &Column, row: usize) -> PyResult<Bound<'py, PyAny>> {
    if !column.here(row) {
        return Ok(py.None().into_bound(py));
    }
    Ok(match column.values() {
        Values::Float(values) => values[row].into_pyobject(py)?.into_any(),
        Values::Int(values) | Values::Time(values) => values[row].into_pyobject(py)?.into_any(),
        Values::Bool(bits) => bit(bits, row).into_pyobject(py)?.to_owned().into_any(),
        Values::Text(..) => column.text_at(row).into_pyobject(py)?.into_any(),
    })
}

/// One of a run's tables, read by any Arrow reader through `__arrow_c_stream__`,
/// column by column as numpy arrays, or row by row.
#[pyclass(frozen, eq, name = "Table", module = "probatum._probatum")]
pub(crate) struct PyTable {
    table: Arc<Table>,
}

impl PartialEq for PyTable {
    fn eq(&self, other: &PyTable) -> bool {
        self.table == other.table
    }
}

impl PyTable {
    fn named(&self, name: &str) -> PyResult<&Column> {
        self.table.column(name).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "name: no column {name:?}; use one of {}",
                names_of(&self.table).join(", ")
            ))
        })
    }
}

impl From<Table> for PyTable {
    fn from(table: Table) -> PyTable {
        PyTable {
            table: Arc::new(table),
        }
    }
}

#[pymethods]
impl PyTable {
    /// A table of `parts`, each column as `parts()` gives it: its name, its kind
    /// ("float", "int", "time", "bool" or "text"), whether it can hold a null, its
    /// values (for text the offset of each row into its bytes and one past the last),
    /// a boolean mask of the rows that hold a value or None where every row does, and
    /// for text its bytes, else None. Every value is copied in, a float's NaN is a
    /// null, and each null holds the value that reads as missing.
    #[new]
    fn new(parts: &Bound<'_, PyAny>) -> PyResult<PyTable> {
        let mut columns = Vec::new();
        for part in parts.try_iter()? {
            #[allow(clippy::type_complexity)]
            let (name, kind, nullable, values, valid, data): (
                String,
                String,
                bool,
                Bound<'_, PyAny>,
                Option<PyReadonlyArray1<'_, bool>>,
                Option<PyReadonlyArray1<'_, u8>>,
            ) = part?.extract()?;
            let floats = |values: &Bound<'_, PyAny>| -> PyResult<Vec<f64>> {
                Ok(values
                    .extract::<PyReadonlyArray1<'_, f64>>()?
                    .as_slice()?
                    .to_vec())
            };
            let integers = |values: &Bound<'_, PyAny>| -> PyResult<Vec<i64>> {
                Ok(values
                    .extract::<PyReadonlyArray1<'_, i64>>()?
                    .as_slice()?
                    .to_vec())
            };
            let given = match kind.as_str() {
                "float" => Given::Float(floats(&values)?),
                "int" => Given::Int(integers(&values)?),
                "time" => Given::Time(integers(&values)?),
                "bool" => Given::Bool(booleans(&values.extract()?)?),
                "text" => {
                    let Some(data) = &data else {
                        return Err(PyValueError::new_err(format!(
                            "parts: text column {name:?} has no bytes; give them last"
                        )));
                    };
                    Given::Text(integers(&values)?, data.as_slice()?.to_vec())
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "parts: column {name:?} of kind {kind:?}; use float, int, time, bool \
                         or text"
                    )))
                }
            };
            let present = valid.map(|valid| booleans(&valid)).transpose()?;
            columns.push(
                Column::checked(&name, given, nullable, present)
                    .map_err(|said| PyValueError::new_err(format!("parts: {said}")))?,
            );
        }
        Table::new(columns)
            .map(PyTable::from)
            .map_err(|said| PyValueError::new_err(format!("parts: {said}")))
    }

    fn __len__(&self) -> usize {
        self.table.rows()
    }

    /// The columns' names, in order.
    #[getter]
    fn names(&self) -> Vec<String> {
        names_of(&self.table)
    }

    /// Each column's name, kind and whether it can hold a null, in order.
    #[getter]
    fn kinds(&self) -> Vec<(String, &'static str, bool)> {
        self.table
            .columns()
            .iter()
            .map(|column| {
                (
                    column.name().to_string_lossy().into_owned(),
                    column.values().kind(),
                    column.nullable(),
                )
            })
            .collect()
    }

    /// Column `name` as a read-only numpy array: floats, NaN where one is null;
    /// integers and times as int64 views with no copy, the least int64 where one is
    /// null; booleans; and text as objects, None for a null.
    fn column<'py>(slf: &Bound<'py, Self>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let held = slf.get();
        let column = held.named(name)?;
        let owner = slf.clone().into_any();
        Ok(match column.values() {
            Values::Float(values) => view(values, owner),
            Values::Int(values) | Values::Time(values) => view(values, owner),
            Values::Bool(bits) => unpacked(py, bits, held.table.rows()),
            Values::Text(..) => {
                // one str for each distinct text, which every row holding it shares
                let mut made: HashMap<&str, Py<PyAny>> = HashMap::new();
                let mut objects = Vec::with_capacity(held.table.rows());
                for row in 0..held.table.rows() {
                    objects.push(if column.here(row) {
                        let said = column.text_at(row);
                        match made.get(said) {
                            Some(object) => object.clone_ref(py),
                            None => {
                                let object = said.into_pyobject(py)?.into_any().unbind();
                                made.insert(said, object.clone_ref(py));
                                object
                            }
                        }
                    } else {
                        py.None()
                    });
                }
                frozen(PyArray1::from_vec(py, objects)).into_any()
            }
        })
    }

    /// The rows where text column `name` holds `value`, as a read-only boolean mask,
    /// compared on its bytes with no str made a row.
    fn equal<'py>(&self, py: Python<'py>, name: &str, value: &str) -> PyResult<Bound<'py, PyAny>> {
        let column = self.named(name)?;
        if !matches!(column.values(), Values::Text(..)) {
            return Err(PyTypeError::new_err(format!(
                "name: {name:?} holds {}, not text; compare a text column",
                column.values().kind()
            )));
        }
        let hits =
            (0..self.table.rows()).map(|row| column.here(row) && column.text_at(row) == value);
        Ok(frozen(PyArray1::from_vec(py, hits.collect())).into_any())
    }

    /// The rows of column `name` that hold a value, as a read-only boolean mask, or
    /// None where every row does.
    fn valid<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        let column = self.named(name)?;
        Ok(column
            .valid()
            .map(|bits| unpacked(py, bits, self.table.rows())))
    }

    /// Row `row` as a dict of its values, None for a null.
    fn row<'py>(&self, py: Python<'py>, row: usize) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.table.rows();
        if row >= rows {
            return Err(PyIndexError::new_err(if rows == 0 {
                format!("row: {row} of a table with no rows; read one that has some")
            } else {
                format!("row: {row} of a table of {rows} rows; give one below {rows}")
            }));
        }
        let out = PyDict::new(py);
        for column in self.table.columns() {
            out.set_item(
                column.name().to_str().unwrap_or(""),
                value(py, column, row)?,
            )?;
        }
        Ok(out)
    }

    /// Each column as the parts its Arrow array is made of, in the form the
    /// constructor takes back: its name, its kind, whether it can hold a null, its
    /// values as a read-only view (booleans unpacked, and text's offsets), the mask of
    /// the rows that hold a value or None where every row does, and text's bytes.
    fn parts<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
        let py = slf.py();
        let held = slf.get();
        let rows = held.table.rows();
        let out = PyList::empty(py);
        for column in held.table.columns() {
            let owner = slf.clone().into_any();
            let none = || py.None().into_bound(py);
            let (values, data) = match column.values() {
                Values::Float(values) => (view(values, owner), none()),
                Values::Int(values) | Values::Time(values) => (view(values, owner), none()),
                Values::Bool(bits) => (unpacked(py, bits, rows), none()),
                Values::Text(ends, bytes) => (view(ends, owner.clone()), view(bytes, owner)),
            };
            let valid = match column.valid() {
                Some(bits) => unpacked(py, bits, rows),
                None => none(),
            };
            out.append((
                column.name().to_str().unwrap_or(""),
                column.values().kind(),
                column.nullable(),
                values,
                valid,
                data,
            ))?;
        }
        Ok(out)
    }

    /// The rows `indices` of this table, in that order.
    fn take(&self, indices: PyReadonlyArray1<'_, i64>) -> PyResult<PyTable> {
        let rows = self.table.rows();
        let picked: Vec<usize> = indices
            .as_slice()?
            .iter()
            .map(|&i| {
                usize::try_from(i)
                    .ok()
                    .filter(|&i| i < rows)
                    .ok_or_else(|| {
                        PyIndexError::new_err(if rows == 0 {
                            format!("indices: {i} of a table with no rows; give none")
                        } else {
                            format!(
                                "indices: {i} of a table of {rows} rows; give positions from \
                                 0 to {}",
                                rows - 1
                            )
                        })
                    })
            })
            .collect::<PyResult<_>>()?;
        Ok(self.table.take(&picked).into())
    }

    /// The table as an Arrow C stream in a capsule, its schema and then one batch of
    /// every column lent with no copy, each call a stream of its own (ADR 0026).
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        if let Some(requested) = requested_schema.filter(|requested| !requested.is_none()) {
            check_requested(&self.table, &requested)?;
        }
        stream_capsule(py, Arc::clone(&self.table))
    }
}

/// The table of `columns`, which the builders below make of one length.
fn built(columns: Vec<Column>) -> PyTable {
    Table::new(columns)
        .expect("a run's records give every column one row each")
        .into()
}

/// The run's ledger, holdings, fills, order log and round trips as tables, in that
/// order, every bar counted from `start`, the account's first row, and each instrument
/// by its name in `names`, or None when reporting is off (ADR 0026).
pub(crate) fn tables(
    engine: &BarEngine,
    units: Units,
    start: usize,
    names: &[String],
) -> PyResult<Option<[(&'static str, PyTable); 5]>> {
    let Some(reporter) = engine.reporter() else {
        return Ok(None);
    };
    if names.len() != engine.width() {
        return Err(PyValueError::new_err(format!(
            "names: {} names for {} instruments; give one each",
            names.len(),
            engine.width()
        )));
    }
    if engine.num_bars() > 0 && engine.open_time(0).is_none() {
        return Err(PyValueError::new_err(
            "times: none, so no row of a table has a time; build the engine with times=",
        ));
    }
    let bar = |tick: usize| tick as i64 - start as i64;
    let time = |tick: usize| engine.open_time(tick).unwrap_or(i64::MIN);
    let width = engine.width();
    let printed = engine.candles_all();
    let close = |tick: usize, i: usize| printed[tick * width + i].close;

    let rows: Vec<LedgerRow> = reporter
        .ledger()
        .iter()
        .map(|&row| units.row(row))
        .collect();
    let float = |name: &str, get: fn(&LedgerRow) -> f64| {
        Column::float(name, rows.iter().map(get).collect())
    };
    let ledger = built(vec![
        Column::int("bar", rows.iter().map(|row| bar(row.tick)).collect()),
        Column::time("time", rows.iter().map(|row| time(row.tick)).collect()),
        float("equity", |row| row.equity),
        float("spot_cash", |row| row.spot_cash),
        float("margin_cash", |row| row.margin_cash),
        float("gross_exposure", |row| row.gross_exposure),
        float("net_exposure", |row| row.net_exposure),
        float("leverage", |row| row.leverage),
        float("margin_used", |row| row.margin_used),
        float("maintenance", |row| row.maintenance),
        float("fees", |row| row.costs.fees),
        float("funding", |row| row.costs.funding),
        float("spread", |row| row.costs.spread),
        float("slippage", |row| row.costs.slippage),
        float("impact", |row| row.costs.impact),
        float("liquidation", |row| row.costs.liquidation),
        float("dust", |row| row.costs.dust),
        Column::float(
            "equity_in_quote",
            rows.iter()
                .map(|row| units.in_quote(row.equity, close(row.tick, 0)))
                .collect(),
        ),
    ]);

    // one pass over the legs, the largest of the records, row by row
    let held = rows.len() * width;
    let mut bars = Vec::with_capacity(held);
    let mut times = Vec::with_capacity(held);
    let mut fields: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::with_capacity(held));
    for (row, legs) in rows.iter().zip(reporter.legs().chunks_exact(width)) {
        let (at, when) = (bar(row.tick), time(row.tick));
        for (i, &leg) in legs.iter().enumerate() {
            let leg = units.leg(leg, close(row.tick, i));
            bars.push(at);
            times.push(when);
            for (field, value) in fields.iter_mut().zip([
                leg.position,
                leg.entry,
                leg.mark,
                leg.notional,
                leg.unrealized,
                leg.weight,
                leg.drift,
            ]) {
                field.push(value);
            }
        }
    }
    let [position, entry, mark, notional, unrealized, weight, drift] = fields;
    let holdings = built(vec![
        Column::int("bar", bars),
        Column::time("time", times),
        Column::cycled("instrument", names, rows.len()),
        Column::float("position", position),
        Column::float("entry", entry),
        Column::float("mark", mark),
        Column::float("notional", notional),
        Column::float("unrealized", unrealized),
        Column::float("weight", weight),
        Column::float("drift", drift),
    ]);

    let fills: Vec<_> = reporter
        .fills()
        .iter()
        .map(|&fill| units.fill(fill))
        .collect();
    let fill_floats = |name: &str, get: fn(&FillRecord) -> f64| {
        Column::float(name, fills.iter().map(get).collect())
    };
    let fills = built(vec![
        Column::int("bar", fills.iter().map(|fill| bar(fill.tick)).collect()),
        Column::time("time", fills.iter().map(|fill| time(fill.tick)).collect()),
        Column::text(
            "instrument",
            fills.iter().map(|fill| names[fill.instrument].as_str()),
        ),
        Column::int_or_null(
            "order",
            fills.iter().map(|fill| fill.order.map(|id| id.0 as i64)),
        ),
        Column::text("side", fills.iter().map(|fill| side_str(fill.side))),
        fill_floats("size", |fill| fill.size),
        fill_floats("price", |fill| fill.price),
        fill_floats("reference", |fill| fill.costs.reference),
        fill_floats("spread", |fill| fill.costs.spread),
        fill_floats("slippage", |fill| fill.costs.slippage),
        fill_floats("impact", |fill| fill.costs.impact),
        fill_floats("fee", |fill| fill.fee),
        fill_floats("adjustment", |fill| fill.costs.adjustment),
        Column::boolean("taker", fills.iter().map(|fill| fill.taker)),
        Column::text("reason", fills.iter().map(|fill| fill.reason.name())),
    ]);

    let events: Vec<_> = reporter
        .events()
        .iter()
        .map(|&event| units.event(event))
        .collect();
    let reasons: Vec<String> = events.iter().map(reason_text).collect();
    let orders = built(vec![
        Column::int("bar", events.iter().map(|event| bar(event.tick)).collect()),
        Column::time(
            "time",
            events.iter().map(|event| time(event.tick)).collect(),
        ),
        Column::text("kind", events.iter().map(|event| event.kind.name())),
        Column::int_or_null(
            "order",
            events.iter().map(|event| event.order.map(|id| id.0 as i64)),
        ),
        Column::text_or_null(
            "instrument",
            events
                .iter()
                .map(|event| event.instrument.map(|k| names[k].as_str())),
        ),
        Column::text_or_null("side", events.iter().map(|event| event.side.map(side_str))),
        Column::float_or_null("size", events.iter().map(|event| event.size)),
        Column::float_or_null("price", events.iter().map(|event| event.price)),
        Column::text("reason", reasons.iter().map(String::as_str)),
    ]);

    let trips: Vec<_> = engine
        .round_trips()
        .into_iter()
        .map(|trip| units.trip(trip))
        .collect();
    let trip_floats = |name: &str, get: fn(&RoundTrip) -> f64| {
        Column::float(name, trips.iter().map(get).collect())
    };
    let trades = built(vec![
        Column::text(
            "instrument",
            trips.iter().map(|trip| names[trip.instrument].as_str()),
        ),
        Column::text(
            "side",
            trips.iter().map(|trip| {
                if trip.side == Side::Buy {
                    "long"
                } else {
                    "short"
                }
            }),
        ),
        trip_floats("size", |trip| trip.size),
        trip_floats("entry", |trip| trip.entry),
        trip_floats("exit", |trip| trip.exit),
        Column::int(
            "open_bar",
            trips.iter().map(|trip| bar(trip.open_tick)).collect(),
        ),
        Column::int_or_null(
            "close_bar",
            trips.iter().map(|trip| trip.close_tick.map(bar)),
        ),
        Column::time(
            "open_time",
            trips.iter().map(|trip| time(trip.open_tick)).collect(),
        ),
        Column::time_or_null(
            "close_time",
            trips.iter().map(|trip| trip.close_tick.map(time)),
        ),
        trip_floats("realized", |trip| trip.realized),
        trip_floats("fees", |trip| trip.fees),
        trip_floats("funding", |trip| trip.funding),
        trip_floats("pnl", |trip| trip.pnl),
        trip_floats("ret", |trip| trip.ret),
        trip_floats("best", |trip| trip.best),
        trip_floats("worst", |trip| trip.worst),
        Column::text_or_null(
            "reason",
            trips.iter().map(|trip| trip.reason.map(|why| why.name())),
        ),
        Column::boolean("open", trips.iter().map(|trip| trip.close_tick.is_none())),
    ]);

    Ok(Some([
        ("ledger", ledger),
        ("holdings", holdings),
        ("fills", fills),
        ("orders", orders),
        ("trades", trades),
    ]))
}
