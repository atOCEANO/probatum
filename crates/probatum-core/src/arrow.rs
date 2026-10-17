//! A table of named columns in the Arrow columnar layout and its export through the
//! Arrow C data and C stream interfaces, whose structs are laid out here as those
//! interfaces define them (ADR 0026). A table never changes once built, and every
//! schema, batch and column it lends owns a share of it, so a reader may move any of
//! them, release them in any order and keep one after the table's own owner is gone.
//! Nothing here touches Python, so the release paths run under Miri.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr::{null, null_mut};
use std::sync::Arc;

/// A schema as the C data interface lays it out.
#[repr(C)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

/// An array as the C data interface lays it out.
#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    pub private_data: *mut c_void,
}

/// A stream of arrays as the C stream interface lays it out.
#[repr(C)]
pub struct ArrowArrayStream {
    pub get_schema: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowSchema) -> c_int>,
    pub get_next: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowArray) -> c_int>,
    pub get_last_error: Option<unsafe extern "C" fn(*mut ArrowArrayStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArrayStream)>,
    pub private_data: *mut c_void,
}

/// A schema with nothing in it, released.
pub fn empty_schema() -> ArrowSchema {
    ArrowSchema {
        format: null(),
        name: null(),
        metadata: null(),
        flags: 0,
        n_children: 0,
        children: null_mut(),
        dictionary: null_mut(),
        release: None,
        private_data: null_mut(),
    }
}

/// An array with nothing in it, released, which is how a stream says it has ended.
pub fn empty_array() -> ArrowArray {
    ArrowArray {
        length: 0,
        null_count: 0,
        offset: 0,
        n_buffers: 0,
        n_children: 0,
        buffers: null_mut(),
        children: null_mut(),
        dictionary: null_mut(),
        release: None,
        private_data: null_mut(),
    }
}

/// The field flag that says a column can hold a null.
const NULLABLE: i64 = 2;

/// What a column holds after its validity bitmap, as its Arrow array lays it out,
/// with a value under every null that reads as missing: NaN for a float, and the
/// least int64, numpy's NaT, for an integer or a time.
#[derive(Debug)]
pub enum Values {
    /// float64
    Float(Vec<f64>),
    /// int64
    Int(Vec<i64>),
    /// nanoseconds since 1970, in UTC
    Time(Vec<i64>),
    /// a bit a row, row i at bit i % 8 of byte i / 8
    Bool(Vec<u8>),
    /// each row's offset into the bytes and one past the last, a lone 0 for no rows,
    /// and the bytes
    Text(Vec<i64>, Vec<u8>),
}

impl Values {
    /// What the column holds, by the name the Python layer gives it.
    pub fn kind(&self) -> &'static str {
        match self {
            Values::Float(_) => "float",
            Values::Int(_) => "int",
            Values::Time(_) => "time",
            Values::Bool(_) => "bool",
            Values::Text(..) => "text",
        }
    }

    fn format(&self) -> &'static CStr {
        match self {
            Values::Float(_) => c"g",
            Values::Int(_) => c"l",
            Values::Time(_) => c"tsn:UTC",
            Values::Bool(_) => c"b",
            Values::Text(..) => c"U",
        }
    }
}

/// A column's values as a caller gives them, booleans one to a row.
pub enum Given {
    Float(Vec<f64>),
    Int(Vec<i64>),
    Time(Vec<i64>),
    Bool(Vec<bool>),
    /// each row's offset into the bytes and one past the last, and the bytes
    Text(Vec<i64>, Vec<u8>),
}

impl Given {
    fn rows(&self) -> usize {
        match self {
            Given::Float(values) => values.len(),
            Given::Int(values) | Given::Time(values) => values.len(),
            Given::Bool(values) => values.len(),
            Given::Text(ends, _) => ends.len().saturating_sub(1),
        }
    }
}

/// Bits a row, row i at bit i % 8 of byte i / 8.
pub fn packed(bits: impl IntoIterator<Item = bool>) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, bit) in bits.into_iter().enumerate() {
        if i % 8 == 0 {
            out.push(0);
        }
        if bit {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Bit `i` of `bits`.
pub fn bit(bits: &[u8], i: usize) -> bool {
    bits[i / 8] >> (i % 8) & 1 == 1
}

/// Text of each row as offsets into its bytes, one past the last, and the bytes, with
/// which rows hold a value where any does not, an empty string under a null.
pub fn texts<'a>(values: impl IntoIterator<Item = Option<&'a str>>) -> (Given, Option<Vec<bool>>) {
    let (mut ends, mut bytes, mut present) = (vec![0], Vec::new(), Vec::new());
    for value in values {
        bytes.extend_from_slice(value.unwrap_or("").as_bytes());
        ends.push(bytes.len() as i64);
        present.push(value.is_some());
    }
    let missing = present.iter().any(|here| !here);
    (Given::Text(ends, bytes), missing.then_some(present))
}

/// One named column.
#[derive(Debug)]
pub struct Column {
    name: CString,
    rows: usize,
    values: Values,
    /// whether the column can hold a null, which its schema says
    nullable: bool,
    /// a bit a row, set where the row holds a value, or none where every row does
    valid: Option<Vec<u8>>,
    nulls: usize,
}

impl Column {
    /// A column of `given`, the rows `present` says hold a value, or every row where it
    /// says nothing; a float's NaN is a null, and every null holds the value that reads
    /// as missing, a NaN already there kept as it is.
    fn with(name: &str, given: Given, nullable: bool, mut present: Option<Vec<bool>>) -> Column {
        let rows = given.rows();
        let values = match given {
            Given::Float(mut values) => {
                if present.is_some() || values.iter().any(|value| value.is_nan()) {
                    let here = present.get_or_insert_with(|| vec![true; rows]);
                    for (here, value) in here.iter_mut().zip(values.iter_mut()) {
                        *here &= !value.is_nan();
                        if !*here && !value.is_nan() {
                            *value = f64::NAN;
                        }
                    }
                }
                Values::Float(values)
            }
            Given::Int(mut values) => {
                Column::missing(&mut values, present.as_deref());
                Values::Int(values)
            }
            Given::Time(mut values) => {
                Column::missing(&mut values, present.as_deref());
                Values::Time(values)
            }
            Given::Bool(values) => {
                Values::Bool(packed(values.iter().enumerate().map(|(row, &value)| {
                    value && present.as_ref().is_none_or(|here| here[row])
                })))
            }
            Given::Text(ends, bytes) => Values::Text(ends, bytes),
        };
        let nulls = present
            .as_ref()
            .map_or(0, |here| here.iter().filter(|&&here| !here).count());
        Column {
            name: CString::new(name).expect("a column's name holds no NUL"),
            rows,
            values,
            nullable,
            valid: present.filter(|_| nulls > 0).map(packed),
            nulls,
        }
    }

    fn missing(values: &mut [i64], present: Option<&[bool]>) {
        for (value, &here) in values.iter_mut().zip(present.unwrap_or_default()) {
            if !here {
                *value = i64::MIN;
            }
        }
    }

    /// Floats, every NaN a null.
    pub fn float(name: &str, values: Vec<f64>) -> Column {
        Column::with(name, Given::Float(values), true, None)
    }

    /// Floats, None a null.
    pub fn float_or_null(name: &str, values: impl IntoIterator<Item = Option<f64>>) -> Column {
        Column::float(
            name,
            values
                .into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect(),
        )
    }

    /// Integers, none of them missing.
    pub fn int(name: &str, values: Vec<i64>) -> Column {
        Column::with(name, Given::Int(values), false, None)
    }

    /// Integers, None a null.
    pub fn int_or_null(name: &str, values: impl IntoIterator<Item = Option<i64>>) -> Column {
        let (values, present) = values
            .into_iter()
            .map(|value| (value.unwrap_or(0), value.is_some()))
            .unzip();
        Column::with(name, Given::Int(values), true, Some(present))
    }

    /// Times in nanoseconds since 1970, none of them missing.
    pub fn time(name: &str, values: Vec<i64>) -> Column {
        Column::with(name, Given::Time(values), false, None)
    }

    /// Times in nanoseconds since 1970, None a null.
    pub fn time_or_null(name: &str, values: impl IntoIterator<Item = Option<i64>>) -> Column {
        let (values, present) = values
            .into_iter()
            .map(|value| (value.unwrap_or(0), value.is_some()))
            .unzip();
        Column::with(name, Given::Time(values), true, Some(present))
    }

    /// Booleans, none of them missing.
    pub fn boolean(name: &str, values: impl IntoIterator<Item = bool>) -> Column {
        Column::with(name, Given::Bool(values.into_iter().collect()), false, None)
    }

    /// Text, none of it missing.
    pub fn text<'a>(name: &str, values: impl IntoIterator<Item = &'a str>) -> Column {
        let (given, _) = texts(values.into_iter().map(Some));
        Column::with(name, given, false, None)
    }

    /// Text, None a null.
    pub fn text_or_null<'a>(
        name: &str,
        values: impl IntoIterator<Item = Option<&'a str>>,
    ) -> Column {
        let (given, present) = texts(values);
        Column::with(name, given, true, present)
    }

    /// Text that runs through `names` in order, `times` times over, with no row
    /// missing, as the rows of a universe name its instruments bar by bar.
    pub fn cycled(name: &str, names: &[String], times: usize) -> Column {
        let block: Vec<u8> = names.iter().flat_map(|name| name.bytes()).collect();
        let mut ends = vec![0];
        let mut at = 0;
        for _ in 0..times {
            for name in names {
                at += name.len() as i64;
                ends.push(at);
            }
        }
        Column::with(name, Given::Text(ends, block.repeat(times)), false, None)
    }

    /// A column of `given`, its rows those `present` says hold a value, or every row
    /// where it says nothing, refused where its name holds a NUL, its mask is not one
    /// a row, a text column's offsets do not cut its bytes into whole UTF-8 rows from
    /// 0 to its end, or a column that cannot hold a null misses a value.
    pub fn checked(
        name: &str,
        given: Given,
        nullable: bool,
        present: Option<Vec<bool>>,
    ) -> Result<Column, String> {
        if name.contains('\0') {
            return Err(format!(
                "column {name:?} holds a NUL in its name; name it without one"
            ));
        }
        let rows = given.rows();
        if let Given::Text(ends, bytes) = &given {
            let whole = ends.first() == Some(&0)
                && ends.windows(2).all(|pair| pair[0] <= pair[1])
                && ends.last().is_some_and(|&end| end as usize == bytes.len())
                && ends.windows(2).all(|pair| {
                    std::str::from_utf8(&bytes[pair[0] as usize..pair[1] as usize]).is_ok()
                });
            if !whole {
                return Err(format!(
                    "text column {name:?} has offsets that do not cut its bytes into UTF-8 \
                     rows from 0 to its end; give one a row and one past the last"
                ));
            }
        }
        if let Some(present) = present.as_ref().filter(|present| present.len() != rows) {
            return Err(format!(
                "column {name:?} has a mask of {} rows for {rows}; give one a row",
                present.len()
            ));
        }
        let column = Column::with(name, given, nullable, present);
        if !nullable && column.nulls > 0 {
            return Err(format!(
                "column {name:?} cannot hold a null and misses a value; give it every value, \
                 or make it nullable"
            ));
        }
        Ok(column)
    }

    pub fn name(&self) -> &CStr {
        &self.name
    }

    pub fn values(&self) -> &Values {
        &self.values
    }

    pub fn nullable(&self) -> bool {
        self.nullable
    }

    /// The validity bitmap, none where every row holds a value.
    pub fn valid(&self) -> Option<&[u8]> {
        self.valid.as_deref()
    }

    /// Whether row `row` holds a value.
    pub fn here(&self, row: usize) -> bool {
        self.valid.as_ref().is_none_or(|bits| bit(bits, row))
    }

    /// The text of row `row` of a text column.
    pub fn text_at(&self, row: usize) -> &str {
        let Values::Text(ends, bytes) = &self.values else {
            unreachable!("read only on a text column");
        };
        let span = &bytes[ends[row] as usize..ends[row + 1] as usize];
        // SAFETY: every row's bytes are a whole str: the typed constructors build them
        // from strs, and `checked` refuses any row that is not one
        unsafe { std::str::from_utf8_unchecked(span) }
    }

    /// This column's rows `rows`, in that order.
    fn take(&self, rows: &[usize]) -> Column {
        let present = self
            .valid
            .is_some()
            .then(|| rows.iter().map(|&row| self.here(row)).collect());
        let given = match &self.values {
            Values::Float(values) => Given::Float(rows.iter().map(|&row| values[row]).collect()),
            Values::Int(values) => Given::Int(rows.iter().map(|&row| values[row]).collect()),
            Values::Time(values) => Given::Time(rows.iter().map(|&row| values[row]).collect()),
            Values::Bool(bits) => Given::Bool(rows.iter().map(|&row| bit(bits, row)).collect()),
            Values::Text(..) => texts(rows.iter().map(|&row| Some(self.text_at(row)))).0,
        };
        Column::with(&self.name.to_string_lossy(), given, self.nullable, present)
    }

    /// Whether row `row` holds the same value here as in `other`, a column of the same
    /// kind.
    fn same(&self, other: &Column, row: usize) -> bool {
        let here = self.here(row);
        here == other.here(row)
            && (!here
                || match (&self.values, &other.values) {
                    (Values::Float(a), Values::Float(b)) => a[row] == b[row],
                    (Values::Int(a), Values::Int(b)) | (Values::Time(a), Values::Time(b)) => {
                        a[row] == b[row]
                    }
                    (Values::Bool(a), Values::Bool(b)) => bit(a, row) == bit(b, row),
                    _ => self.text_at(row) == other.text_at(row),
                })
    }
}

/// A table's columns, each as long as the rest.
#[derive(Debug)]
pub struct Table {
    rows: usize,
    columns: Vec<Column>,
}

impl PartialEq for Table {
    /// Two tables are equal where their columns have the same names, kinds and
    /// nullability, the same rows hold values, and those values are equal.
    fn eq(&self, other: &Table) -> bool {
        self.rows == other.rows
            && self.columns.len() == other.columns.len()
            && self.columns.iter().zip(&other.columns).all(|(a, b)| {
                a.name == b.name
                    && a.values.kind() == b.values.kind()
                    && a.nullable == b.nullable
                    && (0..self.rows).all(|row| a.same(b, row))
            })
    }
}

impl Table {
    /// A table of `columns`, refused where two share a name or one is not as long as
    /// the first.
    pub fn new(columns: Vec<Column>) -> Result<Table, String> {
        let rows = columns.first().map_or(0, |column| column.rows);
        for (k, column) in columns.iter().enumerate() {
            if column.rows != rows {
                return Err(format!(
                    "column {:?} holds {} rows where the first holds {rows}; give columns of \
                     one length",
                    column.name, column.rows
                ));
            }
            if columns[..k].iter().any(|other| other.name == column.name) {
                return Err(format!(
                    "two columns named {:?}; give each column its own name",
                    column.name
                ));
            }
        }
        Ok(Table { rows, columns })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// The column named `name`.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns
            .iter()
            .find(|column| column.name.as_bytes() == name.as_bytes())
    }

    /// The rows `rows` of this table, in that order, each one of its rows.
    pub fn take(&self, rows: &[usize]) -> Table {
        Table {
            rows: rows.len(),
            columns: self
                .columns
                .iter()
                .map(|column| column.take(rows))
                .collect(),
        }
    }
}

/// What a schema handed out owns: a share of the table its names live in, and for a
/// table's own schema the children it points at.
struct SchemaHeld {
    _table: Arc<Table>,
    children: Vec<*mut ArrowSchema>,
}

/// What an array handed out owns: a share of the table its buffers live in, the
/// pointers its `buffers` field points at, and for a batch the columns it holds.
struct ArrayHeld {
    _table: Arc<Table>,
    buffers: Vec<*const c_void>,
    children: Vec<*mut ArrowArray>,
}

/// What a stream handed out owns: a share of its table, and whether its one batch has
/// gone.
struct StreamHeld {
    table: Arc<Table>,
    sent: bool,
}

/// Release a schema `export_schema` filled, or a child of one: each child still
/// unreleased, then the children's places, then what it owns. A child a reader moved
/// out reads as released here, and only its place is freed.
///
/// # Safety
/// `schema` points at a schema this module filled.
unsafe extern "C" fn release_schema(schema: *mut ArrowSchema) {
    let schema = &mut *schema;
    if schema.release.is_none() {
        return;
    }
    let held = Box::from_raw(schema.private_data as *mut SchemaHeld);
    for &child in &held.children {
        if let Some(release) = (*child).release {
            release(child);
        }
        drop(Box::from_raw(child));
    }
    schema.release = None;
}

/// Release an array `export_array` filled, or a column of one, as `release_schema`
/// releases a schema.
///
/// # Safety
/// `array` points at an array this module filled.
unsafe extern "C" fn release_array(array: *mut ArrowArray) {
    let array = &mut *array;
    if array.release.is_none() {
        return;
    }
    let held = Box::from_raw(array.private_data as *mut ArrayHeld);
    for &child in &held.children {
        if let Some(release) = (*child).release {
            release(child);
        }
        drop(Box::from_raw(child));
    }
    array.release = None;
}

/// Fill `out` with the schema of `table`, a struct of its columns.
///
/// # Safety
/// `out` points at a place a schema can be written to.
unsafe fn export_schema(table: &Arc<Table>, out: *mut ArrowSchema) {
    let children: Vec<*mut ArrowSchema> = table
        .columns
        .iter()
        .map(|column| {
            let held = Box::new(SchemaHeld {
                _table: Arc::clone(table),
                children: Vec::new(),
            });
            Box::into_raw(Box::new(ArrowSchema {
                format: column.values.format().as_ptr(),
                name: column.name.as_ptr(),
                metadata: null(),
                flags: if column.nullable { NULLABLE } else { 0 },
                n_children: 0,
                children: null_mut(),
                dictionary: null_mut(),
                release: Some(release_schema),
                private_data: Box::into_raw(held).cast(),
            }))
        })
        .collect();
    let mut held = Box::new(SchemaHeld {
        _table: Arc::clone(table),
        children,
    });
    out.write(ArrowSchema {
        format: c"+s".as_ptr(),
        name: null(),
        metadata: null(),
        flags: 0,
        n_children: held.children.len() as i64,
        children: held.children.as_mut_ptr(),
        dictionary: null_mut(),
        release: Some(release_schema),
        private_data: Box::into_raw(held).cast(),
    });
}

/// Fill `out` with `table` as one batch, a struct array of its columns, each lent with
/// no copy.
///
/// # Safety
/// `out` points at a place an array can be written to.
unsafe fn export_array(table: &Arc<Table>, out: *mut ArrowArray) {
    let children: Vec<*mut ArrowArray> = table
        .columns
        .iter()
        .map(|column| {
            let valid = column
                .valid
                .as_ref()
                .map_or(null(), |bits| bits.as_ptr().cast());
            let buffers = match &column.values {
                Values::Float(values) => vec![valid, values.as_ptr().cast()],
                Values::Int(values) | Values::Time(values) => vec![valid, values.as_ptr().cast()],
                Values::Bool(bits) => vec![valid, bits.as_ptr().cast()],
                Values::Text(ends, bytes) => {
                    vec![valid, ends.as_ptr().cast(), bytes.as_ptr().cast()]
                }
            };
            let mut held = Box::new(ArrayHeld {
                _table: Arc::clone(table),
                buffers,
                children: Vec::new(),
            });
            Box::into_raw(Box::new(ArrowArray {
                length: table.rows as i64,
                null_count: column.nulls as i64,
                offset: 0,
                n_buffers: held.buffers.len() as i64,
                n_children: 0,
                buffers: held.buffers.as_mut_ptr(),
                children: null_mut(),
                dictionary: null_mut(),
                release: Some(release_array),
                private_data: Box::into_raw(held).cast(),
            }))
        })
        .collect();
    let mut held = Box::new(ArrayHeld {
        _table: Arc::clone(table),
        buffers: vec![null()],
        children,
    });
    out.write(ArrowArray {
        length: table.rows as i64,
        null_count: 0,
        offset: 0,
        n_buffers: 1,
        n_children: held.children.len() as i64,
        buffers: held.buffers.as_mut_ptr(),
        children: held.children.as_mut_ptr(),
        dictionary: null_mut(),
        release: Some(release_array),
        private_data: Box::into_raw(held).cast(),
    });
}

/// The table's schema, a struct of its columns, filled into `out`.
///
/// # Safety
/// `stream` is a live stream `stream` made, and `out` a place a schema can go.
unsafe extern "C" fn stream_schema(stream: *mut ArrowArrayStream, out: *mut ArrowSchema) -> c_int {
    let held = &*((*stream).private_data as *const StreamHeld);
    export_schema(&held.table, out);
    0
}

/// The table's one batch, then a released array for every call after, which ends the
/// stream.
///
/// # Safety
/// `stream` is a live stream `stream` made, and `out` a place an array can go.
unsafe extern "C" fn stream_next(stream: *mut ArrowArrayStream, out: *mut ArrowArray) -> c_int {
    let held = &mut *((*stream).private_data as *mut StreamHeld);
    if held.sent {
        out.write(empty_array());
    } else {
        held.sent = true;
        export_array(&held.table, out);
    }
    0
}

/// Nothing a table's stream does can fail once it exists, so it has no error to give.
unsafe extern "C" fn stream_error(_: *mut ArrowArrayStream) -> *const c_char {
    null()
}

/// Release the stream and its share of the table.
///
/// # Safety
/// `stream` points at a stream `stream` made.
unsafe extern "C" fn release_stream(stream: *mut ArrowArrayStream) {
    let stream = &mut *stream;
    if stream.release.is_none() {
        return;
    }
    drop(Box::from_raw(stream.private_data as *mut StreamHeld));
    stream.release = None;
}

/// Release the boxed stream `stream` unless a reader moved it out, then free its box,
/// as the capsule holding it does when it goes.
///
/// # Safety
/// `stream` came from `Box::into_raw` of a stream `stream` made, and is freed once.
pub unsafe fn free_stream(stream: *mut ArrowArrayStream) {
    if let Some(release) = (*stream).release {
        release(stream);
    }
    drop(Box::from_raw(stream));
}

/// `table` as a stream of the C stream interface, whoever holds the struct releasing
/// it: the table's schema, then the whole table as one batch, then the end.
pub fn stream(table: Arc<Table>) -> ArrowArrayStream {
    let held = Box::into_raw(Box::new(StreamHeld { table, sent: false }));
    ArrowArrayStream {
        get_schema: Some(stream_schema),
        get_next: Some(stream_next),
        get_last_error: Some(stream_error),
        release: Some(release_stream),
        private_data: held.cast(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three rows of every kind a table holds, with a null in every column that can
    /// hold one.
    fn table() -> Arc<Table> {
        Arc::new(
            Table::new(vec![
                Column::int("bar", vec![0, 1, 2]),
                Column::time("time", vec![10, 20, 30]),
                Column::float("price", vec![1.5, f64::NAN, 3.5]),
                Column::int_or_null("order", [Some(7), None, Some(9)]),
                Column::time_or_null("closed", [None, Some(25), None]),
                Column::boolean("taker", [true, false, true]),
                Column::cycled(
                    "instrument",
                    &["BTC".to_string(), "ETH\u{00e9}".to_string()],
                    1,
                )
                .take(&[0, 1, 1]),
                Column::text_or_null("reason", [Some("stop"), None, Some("limit")]),
            ])
            .expect("columns of one length"),
        )
    }

    /// What a reader does to take a stream: its schema, then its batches until one
    /// comes back released.
    unsafe fn read(stream: &mut ArrowArrayStream) -> (ArrowSchema, Vec<ArrowArray>) {
        let mut schema = empty_schema();
        assert_eq!((stream.get_schema.unwrap())(stream, &mut schema), 0);
        let mut batches = Vec::new();
        loop {
            let mut batch = empty_array();
            assert_eq!((stream.get_next.unwrap())(stream, &mut batch), 0);
            if batch.release.is_none() {
                return (schema, batches);
            }
            batches.push(batch);
        }
    }

    unsafe fn floats(array: &ArrowArray) -> &[f64] {
        std::slice::from_raw_parts(*array.buffers.add(1) as *const f64, array.length as usize)
    }

    #[test]
    fn a_stream_hands_its_schema_and_one_batch_and_then_ends() {
        let mut stream = stream(table());
        // SAFETY: the stream was just made and is read as a reader reads one
        unsafe {
            let (mut schema, mut batches) = read(&mut stream);
            assert_eq!(CStr::from_ptr(schema.format), c"+s");
            assert_eq!(schema.n_children, 8);
            let order = &**schema.children.add(3);
            assert_eq!(
                (CStr::from_ptr(order.format), CStr::from_ptr(order.name)),
                (c"l", c"order")
            );
            assert_eq!(order.flags, 2);
            assert_eq!((**schema.children).flags, 0);
            assert_eq!(batches.len(), 1);
            let batch = &batches[0];
            assert_eq!((batch.length, batch.n_children, batch.n_buffers), (3, 8, 1));
            let price = &**batch.children.add(2);
            assert_eq!(price.null_count, 1);
            assert!(floats(price)[1].is_nan());
            assert_eq!(*(*price.buffers as *const u8), 0b101);
            let closed = &**batch.children.add(4);
            let times = std::slice::from_raw_parts(*closed.buffers.add(1) as *const i64, 3);
            assert_eq!(times, [i64::MIN, 25, i64::MIN]);
            let reason = &**batch.children.add(7);
            let ends = std::slice::from_raw_parts(*reason.buffers.add(1) as *const i64, 4);
            assert_eq!(ends, [0, 4, 4, 9]);
            (batches[0].release.unwrap())(&mut batches[0]);
            (schema.release.unwrap())(&mut schema);
            (stream.release.unwrap())(&mut stream);
        }
    }

    #[test]
    fn a_column_moved_out_of_its_batch_outlives_the_batch_the_stream_and_the_table() {
        let mut stream = stream(table());
        // SAFETY: the child is moved out by a bitwise copy and its place marked
        // released, as the C data interface lets a reader do, before the batch goes
        unsafe {
            let (mut schema, mut batches) = read(&mut stream);
            (stream.release.unwrap())(&mut stream);
            (schema.release.unwrap())(&mut schema);
            let place = *batches[0].children.add(2);
            let mut price = std::ptr::read(place);
            (*place).release = None;
            (batches[0].release.unwrap())(&mut batches[0]);
            assert_eq!(floats(&price)[2], 3.5);
            (price.release.unwrap())(&mut price);
            assert!(price.release.is_none());
        }
    }

    #[test]
    fn a_column_moved_out_and_released_first_leaves_its_batch_to_release() {
        let held = table();
        let mut stream = stream(Arc::clone(&held));
        // SAFETY: as above, the column released before the batch it came from
        unsafe {
            let (mut schema, mut batches) = read(&mut stream);
            let place = *batches[0].children.add(6);
            let mut instrument = std::ptr::read(place);
            (*place).release = None;
            (instrument.release.unwrap())(&mut instrument);
            (batches[0].release.unwrap())(&mut batches[0]);
            (schema.release.unwrap())(&mut schema);
            (stream.release.unwrap())(&mut stream);
        }
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn a_field_moved_out_of_its_schema_outlives_the_schema() {
        let mut stream = stream(table());
        // SAFETY: as above, for a schema's child
        unsafe {
            let (mut schema, mut batches) = read(&mut stream);
            (stream.release.unwrap())(&mut stream);
            (batches[0].release.unwrap())(&mut batches[0]);
            let place = *schema.children.add(6);
            let mut field = std::ptr::read(place);
            (*place).release = None;
            (schema.release.unwrap())(&mut schema);
            assert_eq!(CStr::from_ptr(field.name), c"instrument");
            (field.release.unwrap())(&mut field);
        }
    }

    #[test]
    fn a_stream_freed_unread_frees_its_table() {
        let held = table();
        let boxed = Box::into_raw(Box::new(stream(Arc::clone(&held))));
        assert_eq!(Arc::strong_count(&held), 2);
        // SAFETY: the box is freed once, as the capsule holding it frees it
        unsafe { free_stream(boxed) };
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn a_stream_moved_out_of_its_box_outlives_the_box() {
        let held = table();
        let boxed = Box::into_raw(Box::new(stream(Arc::clone(&held))));
        // SAFETY: the stream is moved out by a bitwise copy and the box's marked
        // released, as a reader takes a stream out of its capsule, before the box goes
        unsafe {
            let mut moved = std::ptr::read(boxed);
            (*boxed).release = None;
            free_stream(boxed);
            assert_eq!(Arc::strong_count(&held), 2);
            let (mut schema, mut batches) = read(&mut moved);
            assert_eq!(floats(&**batches[0].children.add(2))[2], 3.5);
            (batches[0].release.unwrap())(&mut batches[0]);
            (schema.release.unwrap())(&mut schema);
            (moved.release.unwrap())(&mut moved);
        }
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn an_empty_table_streams_one_offset_for_its_text_and_no_rows() {
        let empty = Arc::new(
            Table::new(vec![
                Column::float("price", Vec::new()),
                Column::text("instrument", Vec::<&str>::new()),
                Column::boolean("taker", Vec::new()),
            ])
            .expect("columns of one length"),
        );
        let mut stream = stream(empty);
        // SAFETY: as in the tests above
        unsafe {
            let (mut schema, mut batches) = read(&mut stream);
            assert_eq!(batches[0].length, 0);
            let text = &**batches[0].children.add(1);
            assert_eq!(*(*text.buffers.add(1) as *const i64), 0);
            (batches[0].release.unwrap())(&mut batches[0]);
            (schema.release.unwrap())(&mut schema);
            (stream.release.unwrap())(&mut stream);
        }
    }

    #[test]
    fn rows_taken_keep_their_values_and_their_nulls() {
        let held = table();
        let taken = held.take(&[2, 1]);
        assert_eq!(taken.rows(), 2);
        let order = taken.column("order").expect("a column named order");
        assert!(order.here(0) && !order.here(1));
        assert_eq!(taken.column("reason").expect("reason").text_at(0), "limit");
        assert_eq!(
            taken.column("instrument").expect("instrument").text_at(0),
            "ETH\u{00e9}"
        );
        assert_eq!(held.take(&[0, 1, 2]), *held);
        assert_ne!(held.take(&[1, 0, 2]), *held);
        let one = |order: Option<i64>| {
            Table::new(vec![Column::int_or_null("order", [order])]).expect("one column")
        };
        // a null differs from a value on whichever side of the comparison it stands
        assert_ne!(one(None), one(Some(7)));
        assert_ne!(one(Some(7)), one(None));
    }

    #[test]
    fn each_column_gives_its_name_its_kind_its_nullability_and_its_bitmap() {
        let held = table();
        let read: Vec<_> = held
            .columns()
            .iter()
            .map(|column| {
                (
                    column.name().to_str().expect("an ASCII name"),
                    column.values().kind(),
                    column.nullable(),
                    column.valid(),
                )
            })
            .collect();
        assert_eq!(
            read,
            [
                ("bar", "int", false, None),
                ("time", "time", false, None),
                ("price", "float", true, Some(&[0b101][..])),
                ("order", "int", true, Some(&[0b101][..])),
                ("closed", "time", true, Some(&[0b010][..])),
                ("taker", "bool", false, None),
                ("instrument", "text", false, None),
                ("reason", "text", true, Some(&[0b101][..])),
            ]
        );
        let Values::Bool(bits) = held.column("taker").expect("taker").values() else {
            unreachable!("a bool column holds bits")
        };
        assert_eq!(bits, &[0b101]);
        let mut nine = vec![1.0; 9];
        nine[8] = f64::NAN;
        assert_eq!(Column::float("x", nine).valid(), Some(&[0xff, 0][..]));
        let none = Column::text_or_null("reason", [None, None]);
        assert!(!none.here(0) && !none.here(1));
        let whole = held.take(&[0, 2]);
        assert_eq!(whole.column("order").expect("order").valid(), None);
    }

    #[test]
    fn columns_of_two_lengths_or_of_one_name_make_no_table() {
        let short = Table::new(vec![
            Column::int("bar", vec![0, 1]),
            Column::int("x", vec![0]),
        ]);
        assert!(short.is_err());
        let twice = Table::new(vec![
            Column::int("bar", vec![0]),
            Column::int("bar", vec![1]),
        ]);
        assert!(twice.is_err());
    }

    #[test]
    fn a_cycled_column_names_each_row_s_instrument_in_turn() {
        let names = ["BTCUSDT".to_string(), "ETH".to_string()];
        let column = Column::cycled("instrument", &names, 3);
        let read: Vec<&str> = (0..6).map(|row| column.text_at(row)).collect();
        assert_eq!(read, ["BTCUSDT", "ETH", "BTCUSDT", "ETH", "BTCUSDT", "ETH"]);
        assert_eq!(Column::cycled("instrument", &names, 0).rows, 0);
    }

    #[test]
    fn parts_that_do_not_cut_into_whole_rows_are_refused() {
        let cut = |ends: Vec<i64>, bytes: &[u8]| {
            Column::checked("reason", Given::Text(ends, bytes.to_vec()), true, None)
        };
        assert!(cut(vec![0, 7, 5], b"abcde").is_err());
        assert!(cut(vec![0, 2, 6], b"abcde").is_err());
        assert!(cut(vec![1, 5], b"abcde").is_err());
        assert!(cut(Vec::new(), b"").is_err());
        assert!(cut(vec![0, 1, 2], "\u{00e9}".as_bytes()).is_err());
        assert!(cut(vec![0, 2, 5], b"abcde").is_ok());
        let missing = Column::checked(
            "bar",
            Given::Int(vec![1, 2]),
            false,
            Some(vec![true, false]),
        );
        assert!(missing.is_err());
        assert!(Column::checked("bar", Given::Int(vec![1, 2]), false, None).is_ok());
        let short = Column::checked("bar", Given::Int(vec![1, 2]), true, Some(vec![true]));
        assert!(short.is_err());
        assert!(Column::checked("b\0r", Given::Int(vec![1]), false, None).is_err());
        let floated = Column::checked(
            "x",
            Given::Float(vec![0.0, 2.0]),
            true,
            Some(vec![false, true]),
        )
        .expect("a float column");
        let Values::Float(values) = floated.values() else {
            unreachable!("a float column holds floats")
        };
        assert!(values[0].is_nan() && !floated.here(0));
    }
}
