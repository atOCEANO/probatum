//! A frame's columns through the Arrow C data interface, which polars, pyarrow, DuckDB
//! and pandas with pyarrow all export, so a Data reads any of them without probatum
//! importing one (ADR 0044). Only what bars are made of is read: numbers, and times
//! and dates as nanoseconds since the epoch with the zone the column names. Any other
//! column is named, with its format, and left unread, and nothing in a column the
//! caller does not read can refuse the frame, but a time past what nanoseconds can
//! count.

use std::ffi::{c_char, c_void, CStr};
use std::ptr::null;

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};

use probatum_core::arrow::{empty_array, empty_schema, ArrowArray, ArrowArrayStream, ArrowSchema};

use crate::{DataError, ProbatumTypeError};

/// A schema a stream handed over, released when it goes out of scope.
struct OwnedSchema(ArrowSchema);

impl Drop for OwnedSchema {
    fn drop(&mut self) {
        if let Some(release) = self.0.release {
            // SAFETY: a schema the stream filled is released once, by its own callback
            unsafe { release(&mut self.0) }
        }
    }
}

/// An array a stream handed over, released when it goes out of scope.
struct OwnedArray(ArrowArray);

impl Drop for OwnedArray {
    fn drop(&mut self) {
        if let Some(release) = self.0.release {
            // SAFETY: an array the stream filled is released once, by its own callback
            unsafe { release(&mut self.0) }
        }
    }
}

/// What a column's format says it holds.
enum Kind {
    /// a float of so many bytes
    Float(usize),
    /// an integer of so many bytes, signed or not
    Integer(usize, bool),
    /// a time, each tick so many nanoseconds, in the zone named, empty where none is
    Time(i64, String),
    /// a date, each tick so many nanoseconds, of so many bytes
    Date(i64, usize),
    /// anything else, by its format
    Other(String),
}

fn kind_of(format: &str) -> Kind {
    match format {
        "g" => Kind::Float(8),
        "f" => Kind::Float(4),
        "l" => Kind::Integer(8, true),
        "i" => Kind::Integer(4, true),
        "s" => Kind::Integer(2, true),
        "c" => Kind::Integer(1, true),
        "L" => Kind::Integer(8, false),
        "I" => Kind::Integer(4, false),
        "S" => Kind::Integer(2, false),
        "C" => Kind::Integer(1, false),
        "tdD" => Kind::Date(86_400_000_000_000, 4),
        "tdm" => Kind::Date(1_000_000, 8),
        _ => match format.split_once(':') {
            Some(("tss", zone)) => Kind::Time(1_000_000_000, zone.to_string()),
            Some(("tsm", zone)) => Kind::Time(1_000_000, zone.to_string()),
            Some(("tsu", zone)) => Kind::Time(1_000, zone.to_string()),
            Some(("tsn", zone)) => Kind::Time(1, zone.to_string()),
            _ => Kind::Other(format.to_string()),
        },
    }
}

/// One column as the batches hand it over: its values as floats, an integer, time or
/// date column's as integers too, whether each was there, and whether an unsigned
/// integer passed what a signed 64-bit one holds.
struct Column {
    name: String,
    kind: Kind,
    floats: Vec<f64>,
    integers: Vec<i64>,
    present: Vec<bool>,
    wide: bool,
}

/// The columns `schema` names: a table's, or the one column an array of anything else
/// is, which the second value says.
///
/// # Safety
/// `schema` points at a schema of the C data interface.
unsafe fn columns_of(schema: &ArrowSchema) -> Result<(Vec<Column>, bool), String> {
    if schema.release.is_none() {
        return Err("an Arrow schema already released; pass the frame again".to_string());
    }
    let column = |field: &ArrowSchema| {
        let kind = if field.dictionary.is_null() {
            kind_of(&text(field.format))
        } else {
            Kind::Other(format!(
                "dictionary of {}",
                text((*field.dictionary).format)
            ))
        };
        Column {
            name: text(field.name),
            kind,
            floats: Vec::new(),
            integers: Vec::new(),
            present: Vec::new(),
            wide: false,
        }
    };
    if text(schema.format) != "+s" {
        return Ok((vec![column(schema)], true));
    }
    let width = usize::try_from(schema.n_children).map_err(|_| {
        format!(
            "an Arrow table of {} columns; give a table",
            schema.n_children
        )
    })?;
    Ok((
        (0..width)
            .map(|i| column(&**schema.children.add(i)))
            .collect(),
        false,
    ))
}

/// The text a C string holds, empty where there is none.
///
/// # Safety
/// `text` is null or a NUL-terminated string.
pub(crate) unsafe fn text(text: *const c_char) -> String {
    if text.is_null() {
        String::new()
    } else {
        CStr::from_ptr(text).to_string_lossy().into_owned()
    }
}

/// Whether bit `i` of the bitmap at `bits` is set, every bit set where there is none.
///
/// # Safety
/// `bits` is null or holds at least `i + 1` bits.
unsafe fn set(bits: *const c_void, i: usize) -> bool {
    bits.is_null() || (*(bits as *const u8).add(i / 8) >> (i % 8)) & 1 == 1
}

/// Value `i` of the buffer at `values`, an integer of `bytes` bytes, signed or not,
/// or none where it is an unsigned one past what a signed 64-bit integer holds.
///
/// # Safety
/// `values` holds at least `i + 1` integers of that width.
unsafe fn integer(values: *const c_void, i: usize, bytes: usize, signed: bool) -> Option<i64> {
    match (bytes, signed) {
        (8, true) => Some((values as *const i64).add(i).read_unaligned()),
        (4, true) => Some((values as *const i32).add(i).read_unaligned().into()),
        (2, true) => Some((values as *const i16).add(i).read_unaligned().into()),
        (1, true) => Some((values as *const i8).add(i).read_unaligned().into()),
        (8, false) => i64::try_from((values as *const u64).add(i).read_unaligned()).ok(),
        (4, false) => Some((values as *const u32).add(i).read_unaligned().into()),
        (2, false) => Some((values as *const u16).add(i).read_unaligned().into()),
        _ => Some((values as *const u8).add(i).read_unaligned().into()),
    }
}

/// Time `i` of the buffer at `values`, ticks of `bytes` bytes each `unit`
/// nanoseconds, in nanoseconds since 1970, or what is wrong with it where
/// nanoseconds since 1970 cannot count it.
///
/// # Safety
/// `values` holds at least `i + 1` ticks of that width.
unsafe fn time(values: *const c_void, i: usize, bytes: usize, unit: i64) -> Result<i64, String> {
    let ticks = integer(values, i, bytes, true).unwrap_or(0);
    ticks.checked_mul(unit).ok_or_else(|| {
        format!(
            "{ticks} ticks of {unit} nanoseconds, past what nanoseconds since 1970 can \
             count"
        )
    })
}

/// Append `array`, a batch of the table whose columns are `columns`, or with `alone`
/// a batch of the one column they are. A missing time is read as the int64 numpy
/// calls NaT, so a column no one reads as the clock cannot refuse the frame.
///
/// # Safety
/// `array` points at an array of the C data interface whose children hold the
/// columns' formats, or with `alone` whose format is the column's.
unsafe fn append(array: &ArrowArray, columns: &mut [Column], alone: bool) -> Result<(), String> {
    if array.release.is_none() {
        return Err("an Arrow array already released; pass the frame again".to_string());
    }
    if alone {
        let Ok(length) = usize::try_from(array.length) else {
            return Err(format!(
                "an Arrow array of {} values; give an array",
                array.length
            ));
        };
        return read(array, &mut columns[0], 0, length, null());
    }
    let width = usize::try_from(array.n_children).unwrap_or(0);
    if width != columns.len() {
        return Err(format!(
            "an Arrow batch of {width} columns in a table of {}; give batches of the \
             table's own columns",
            columns.len()
        ));
    }
    let (Ok(length), Ok(offset)) = (usize::try_from(array.length), usize::try_from(array.offset))
    else {
        return Err(format!(
            "an Arrow batch of {} rows from row {}; give a table",
            array.length, array.offset
        ));
    };
    // a table's row can itself be null, which leaves every column without a value
    let rows_present = if array.n_buffers > 0 {
        *array.buffers
    } else {
        null()
    };
    for (i, column) in columns.iter_mut().enumerate() {
        read(
            &**array.children.add(i),
            column,
            offset,
            length,
            rows_present,
        )?;
    }
    Ok(())
}

/// Append rows `offset` to `offset + length` of `child` to `column`, each present
/// where `child` and `rows_present`, its batch's own bitmap, both set it.
///
/// # Safety
/// `child` points at an array of the C data interface of the column's format, and
/// `rows_present` is null or holds a bit for each of those rows.
unsafe fn read(
    child: &ArrowArray,
    column: &mut Column,
    offset: usize,
    length: usize,
    rows_present: *const c_void,
) -> Result<(), String> {
    if matches!(column.kind, Kind::Other(_)) {
        return Ok(());
    }
    let Ok(start) = usize::try_from(child.offset) else {
        return Err(format!(
            "column {:?} starts at row {}; give a table",
            column.name, child.offset
        ));
    };
    if child.n_buffers < 2 {
        return Err(format!(
            "column {:?} has {} buffers where its values need 2; give a table",
            column.name, child.n_buffers
        ));
    }
    if usize::try_from(child.length).unwrap_or(0) < offset + length {
        return Err(format!(
            "column {:?} holds {} values for rows {offset} to {}; give whole columns",
            column.name,
            child.length,
            offset + length
        ));
    }
    let present = *child.buffers;
    let values = *child.buffers.add(1);
    let before = column.present.len();
    for row in 0..length {
        let at = start + offset + row;
        let here = set(present, at) && set(rows_present, offset + row);
        column.present.push(here);
        match &column.kind {
            Kind::Float(8) => column.floats.push(if here {
                (values as *const f64).add(at).read_unaligned()
            } else {
                f64::NAN
            }),
            Kind::Float(_) => column.floats.push(if here {
                (values as *const f32).add(at).read_unaligned().into()
            } else {
                f64::NAN
            }),
            Kind::Integer(bytes, signed) => {
                match here.then(|| integer(values, at, *bytes, *signed)) {
                    None => {
                        column.integers.push(0);
                        column.floats.push(f64::NAN);
                    }
                    Some(Some(value)) => {
                        column.integers.push(value);
                        column.floats.push(value as f64);
                    }
                    Some(None) => {
                        // an unsigned value past a signed 64-bit integer, which
                        // leaves the column to come back as floats
                        column.wide = true;
                        column.integers.push(0);
                        column
                            .floats
                            .push((values as *const u64).add(at).read_unaligned() as f64);
                    }
                }
            }
            Kind::Time(unit, _) => column.integers.push(if here {
                time(values, at, 8, *unit).map_err(|said| {
                    format!(
                        "column {:?} holds {said} at row {}",
                        column.name,
                        before + row
                    )
                })?
            } else {
                i64::MIN
            }),
            Kind::Date(unit, bytes) => column.integers.push(if here {
                time(values, at, *bytes, *unit).map_err(|said| {
                    format!(
                        "column {:?} holds {said} at row {}",
                        column.name,
                        before + row
                    )
                })?
            } else {
                i64::MIN
            }),
            Kind::Other(_) => unreachable!("skipped above"),
        }
    }
    Ok(())
}

/// The message a stream gives for its last error.
///
/// # Safety
/// `stream` is a live stream.
unsafe fn last_error(stream: *mut ArrowArrayStream) -> String {
    let said = match (*stream).get_last_error {
        Some(get_last_error) => text(get_last_error(stream)),
        None => String::new(),
    };
    if said.is_empty() {
        "the Arrow stream failed and gave no reason; pass the frame again".to_string()
    } else {
        format!("the Arrow stream failed: {said}; pass the frame again")
    }
}

/// Every batch of `stream`, read into its table's columns.
///
/// # Safety
/// `stream` points at a stream of the C stream interface.
unsafe fn read_stream(stream: *mut ArrowArrayStream) -> Result<Vec<Column>, String> {
    // a stream moved out of its capsule keeps its callbacks and loses its release,
    // which alone marks it released
    let (Some(_), Some(get_schema), Some(get_next)) =
        ((*stream).release, (*stream).get_schema, (*stream).get_next)
    else {
        return Err("an Arrow stream already released; pass the frame again".to_string());
    };
    let mut schema = OwnedSchema(empty_schema());
    if get_schema(stream, &mut schema.0) != 0 {
        return Err(last_error(stream));
    }
    let (mut columns, alone) = columns_of(&schema.0)?;
    loop {
        let mut batch = OwnedArray(empty_array());
        if get_next(stream, &mut batch.0) != 0 {
            return Err(last_error(stream));
        }
        if batch.0.release.is_none() {
            return Ok(columns);
        }
        append(&batch.0, &mut columns, alone)?;
    }
}

/// The pointer `capsule`, which `method` gave, holds under `name`.
fn pointer_in(capsule: &Bound<'_, PyAny>, name: &CStr, method: &str) -> PyResult<*mut c_void> {
    let held = capsule
        .cast::<PyCapsule>()
        .ok()
        .and_then(|capsule| capsule.pointer_checked(Some(name)).ok());
    match held {
        Some(pointer) => Ok(pointer.as_ptr()),
        None => Err(ProbatumTypeError::new_err(format!(
            "data: {method} gave a {}, not a capsule named {}; pass a frame that exports \
             one",
            capsule.get_type().name()?,
            name.to_string_lossy()
        ))),
    }
}

/// The columns of `data`, an object that exports the Arrow C stream or array
/// interface, a table's or the one an array of anything else is, each as a tuple of
/// its name, its kind and its values: "float" with float64 values, NaN where one is
/// missing; "int" with int64 values where an integer column misses none and fits in
/// them, and "float" where it does not; "time" and "date" with int64 nanoseconds
/// since 1970, NaT where one is missing, and, last, the zone a time names, empty
/// where it names none; and "other" with no values and, last, the column's format.
#[pyfunction]
pub fn arrow_columns<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let read = if data.hasattr("__arrow_c_stream__")? {
        let capsule = data.call_method0("__arrow_c_stream__")?;
        let stream = pointer_in(&capsule, c"arrow_array_stream", "__arrow_c_stream__")?;
        // SAFETY: the capsule holds the stream until it is dropped, after this read,
        // and its destructor releases the stream then
        unsafe { read_stream(stream as *mut ArrowArrayStream) }
    } else if data.hasattr("__arrow_c_array__")? {
        let pair = data.call_method0("__arrow_c_array__")?;
        let Ok((schema, array)) = pair.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>() else {
            return Err(ProbatumTypeError::new_err(format!(
                "data: __arrow_c_array__ gave a {}, not a pair of capsules; pass a frame \
                 that exports one",
                pair.get_type().name()?
            )));
        };
        let schema = pointer_in(&schema, c"arrow_schema", "__arrow_c_array__")?;
        let array = pointer_in(&array, c"arrow_array", "__arrow_c_array__")?;
        // SAFETY: each capsule holds its schema or array until it is dropped, after
        // this read, and its destructor releases it then
        unsafe {
            columns_of(&*(schema as *const ArrowSchema)).and_then(|(mut columns, alone)| {
                append(&*(array as *const ArrowArray), &mut columns, alone).map(|()| columns)
            })
        }
    } else {
        return Err(ProbatumTypeError::new_err(format!(
            "data: a {} has neither the Arrow C stream nor the array interface; pass a \
             frame that exports one",
            data.get_type().name()?
        )));
    };
    let columns = read.map_err(|said| DataError::new_err(format!("data: {said}")))?;
    let out = PyList::empty(py);
    for column in columns {
        let missing = column.present.iter().any(|present| !present);
        let (kind, values, detail) = match column.kind {
            Kind::Float(_) => (
                "float",
                PyArray1::from_vec(py, column.floats).into_any(),
                String::new(),
            ),
            Kind::Integer(..) if missing || column.wide => (
                "float",
                PyArray1::from_vec(py, column.floats).into_any(),
                String::new(),
            ),
            Kind::Integer(..) => (
                "int",
                PyArray1::from_vec(py, column.integers).into_any(),
                String::new(),
            ),
            Kind::Time(_, zone) => (
                "time",
                PyArray1::from_vec(py, column.integers).into_any(),
                zone,
            ),
            Kind::Date(..) => (
                "date",
                PyArray1::from_vec(py, column.integers).into_any(),
                String::new(),
            ),
            Kind::Other(format) => ("other", py.None().into_bound(py), format),
        };
        out.append((column.name, kind, values, detail))?;
    }
    Ok(out)
}
