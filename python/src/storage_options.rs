// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use lance_io::object_store::{
    LanceNamespaceStorageOptionsProvider, StorageOptionsAccessor, StorageOptionsProvider,
};
use pyo3::prelude::*;

use crate::rt;

/// Python wrapper for StorageOptionsAccessor
///
/// This wraps a Rust StorageOptionsAccessor and exposes it to Python.
#[pyclass(name = "StorageOptionsAccessor", skip_from_py_object)]
#[derive(Clone)]
pub struct PyStorageOptionsAccessor {
    inner: Arc<StorageOptionsAccessor>,
}

impl PyStorageOptionsAccessor {
    pub fn new(accessor: Arc<StorageOptionsAccessor>) -> Self {
        Self { inner: accessor }
    }

    pub fn inner(&self) -> Arc<StorageOptionsAccessor> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyStorageOptionsAccessor {
    /// Create an accessor with only static options (no refresh capability)
    #[staticmethod]
    fn with_static_options(options: HashMap<String, String>) -> Self {
        Self {
            inner: Arc::new(StorageOptionsAccessor::with_static_options(options)),
        }
    }

    /// Get current valid storage options
    fn get_storage_options(&self, py: Python<'_>) -> PyResult<HashMap<String, String>> {
        let accessor = self.inner.clone();
        let options = rt()
            .block_on(Some(py), accessor.get_storage_options())?
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(options.0)
    }

    /// Get the initial storage options without refresh
    fn initial_storage_options(&self) -> Option<HashMap<String, String>> {
        self.inner.initial_storage_options().cloned()
    }

    /// Get the accessor ID for equality/hashing
    fn accessor_id(&self) -> String {
        self.inner.accessor_id()
    }

    /// Check if this accessor has a dynamic provider
    fn has_provider(&self) -> bool {
        self.inner.has_provider()
    }

    /// Get the refresh offset in seconds
    fn refresh_offset_secs(&self) -> u64 {
        self.inner.refresh_offset().as_secs()
    }

    fn __repr__(&self) -> String {
        format!(
            "StorageOptionsAccessor(id={}, has_provider={})",
            self.inner.accessor_id(),
            self.inner.has_provider()
        )
    }
}

/// Create a StorageOptionsAccessor from storage options
///
/// When a namespace client and table id are given, the accessor is backed by a
/// [`LanceNamespaceStorageOptionsProvider`] so that vended credentials are refreshed
/// before they expire. Without them the accessor holds static options that never refresh.
///
/// This mirrors the guard used when opening a dataset: a provider is attached whenever
/// storage options are present. Options without `expires_at_millis` are treated as never
/// expiring, so they never trigger a refresh.
///
/// A namespace with no storage options at all yields no accessor, matching the rest of
/// the Python surface. The Java binding and [`object_store_from_uri_or_path_with_provider`]
/// instead fall back to `StorageOptionsAccessor::with_provider`, which fetches credentials
/// on first use; aligning the two is a behavior change worth making on its own.
///
/// [`object_store_from_uri_or_path_with_provider`]: crate::file::object_store_from_uri_or_path_with_provider
pub fn create_accessor_from_storage_options(
    storage_options: Option<HashMap<String, String>>,
    namespace_client: Option<&Bound<'_, PyAny>>,
    table_id: Option<&[String]>,
) -> PyResult<Option<Arc<StorageOptionsAccessor>>> {
    let Some(opts) = storage_options else {
        return Ok(None);
    };

    if let (Some(ns_client), Some(table_id)) = (namespace_client, table_id) {
        let ns_client = crate::namespace::extract_namespace_arc(ns_client.py(), ns_client)?;
        let provider: Arc<dyn StorageOptionsProvider> = Arc::new(
            LanceNamespaceStorageOptionsProvider::new(ns_client, table_id.to_vec()),
        );
        return Ok(Some(Arc::new(
            StorageOptionsAccessor::with_initial_and_provider(opts, provider),
        )));
    }

    Ok(Some(Arc::new(StorageOptionsAccessor::with_static_options(
        opts,
    ))))
}
