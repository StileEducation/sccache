// Copyright 2016 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::cache::{Cache, CacheMode, CacheRead, CacheWrite, GetPathResult, Storage};
use crate::compiler::PreprocessorCacheEntry;
use crate::lru_disk_cache::{Error as LruError, ReadSeek};
use async_trait::async_trait;
use bytes::Bytes;
use std::ffi::OsStr;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::errors::*;

use super::lazy_disk_cache::LazyDiskCache;
use super::utils::normalize_key;
use crate::config::PreprocessorCacheModeConfig;

/// A cache that stores entries at local disk paths.
pub struct DiskCache {
    /// `LruDiskCache` does all the real work here.
    lru: Arc<Mutex<LazyDiskCache>>,
    /// Thread pool to execute disk I/O
    pool: tokio::runtime::Handle,
    preprocessor_cache_mode_config: PreprocessorCacheModeConfig,
    preprocessor_cache: Arc<Mutex<LazyDiskCache>>,
    rw_mode: CacheMode,
    basedirs: Vec<Vec<u8>>,
    root: PathBuf,
    /// Where outputs are staged as clones, if the cache dir supports them;
    /// probed on first use.
    clone_staging: std::sync::OnceLock<Option<PathBuf>>,
}

impl DiskCache {
    /// Create a new `DiskCache` rooted at `root`, with `max_size` as the maximum cache size on-disk, in bytes.
    pub fn new<T: AsRef<OsStr>>(
        root: T,
        max_size: u64,
        pool: &tokio::runtime::Handle,
        preprocessor_cache_mode_config: PreprocessorCacheModeConfig,
        rw_mode: CacheMode,
        basedirs: Vec<Vec<u8>>,
    ) -> DiskCache {
        DiskCache {
            lru: Arc::new(Mutex::new(LazyDiskCache::Uninit {
                root: root.as_ref().to_os_string(),
                max_size,
            })),
            pool: pool.clone(),
            preprocessor_cache_mode_config,
            preprocessor_cache: Arc::new(Mutex::new(LazyDiskCache::Uninit {
                root: Path::new(root.as_ref())
                    .join("preprocessor")
                    .into_os_string(),
                max_size,
            })),
            rw_mode,
            basedirs,
            root: PathBuf::from(root.as_ref()),
            clone_staging: std::sync::OnceLock::new(),
        }
    }
}

/// Where a cloned object with content hash `hash` lives in the cache.
fn make_clone_key_path(hash: &str) -> PathBuf {
    Path::new("clones")
        .join(&hash[0..1])
        .join(&hash[1..2])
        .join(hash)
}

/// Make a path to the cache entry with key `key`.
fn make_key_path(key: &str) -> PathBuf {
    Path::new(&key[0..1]).join(&key[1..2]).join(key)
}

#[async_trait]
impl Storage for DiskCache {
    async fn get(&self, key: &str) -> Result<Cache> {
        trace!("DiskCache::get({})", key);
        let path = make_key_path(key);
        let lru = self.lru.clone();
        let key = key.to_owned();

        self.pool
            .spawn_blocking(move || {
                let io = match lru.lock().unwrap().get_or_init()?.get(&path) {
                    Ok(f) => f,
                    Err(LruError::FileNotInCache) => {
                        trace!("DiskCache::get({}): FileNotInCache", key);
                        return Ok(Cache::Miss);
                    }
                    Err(LruError::Io(e)) => {
                        trace!("DiskCache::get({}): IoError: {:?}", key, e);
                        return Err(e.into());
                    }
                    Err(_) => unreachable!(),
                };
                let mut hit = CacheRead::from(io)?;
                // Objects stored as clones must all still be here, or it's a miss.
                for clone in hit.clone_refs()? {
                    let clone_key = make_clone_key_path(&clone.hash);
                    let mut lru = lru.lock().unwrap();
                    let lru = lru.get_or_init()?;
                    // get_file also refreshes the mtime the LRU order is rebuilt from.
                    match lru
                        .get_file(&clone_key)
                        .ok()
                        .and_then(|_| lru.get_abs_path(&clone_key))
                    {
                        Some(path) => hit.set_clone_source(clone.key, path, clone.mode),
                        None => {
                            trace!("DiskCache::get({}): clone {} evicted", key, clone.hash);
                            return Ok(Cache::Miss);
                        }
                    }
                }
                Ok(Cache::Hit(hit))
            })
            .await?
    }

    async fn get_with_raw(&self, key: &str) -> Result<(Cache, Option<Bytes>)> {
        match self.get_raw(key).await? {
            Some(data) => {
                let hit = CacheRead::from(std::io::Cursor::new(data.clone()))?;
                Ok((Cache::Hit(hit), Some(data)))
            }
            None => Ok((Cache::Miss, None)),
        }
    }

    async fn get_raw(&self, key: &str) -> Result<Option<Bytes>> {
        trace!("DiskCache::get_raw({})", key);
        let path = make_key_path(key);
        let lru = self.lru.clone();
        let key = key.to_owned();

        self.pool
            .spawn_blocking(
                move || match lru.lock().unwrap().get_or_init()?.get(&path) {
                    Ok(mut io) => {
                        let mut data = Vec::new();
                        io.read_to_end(&mut data)?;
                        trace!("DiskCache::get_raw({}): Found {} bytes", key, data.len());
                        Ok(Some(Bytes::from(data)))
                    }
                    Err(LruError::FileNotInCache) => {
                        trace!("DiskCache::get_raw({}): FileNotInCache", key);
                        Ok(None)
                    }
                    Err(LruError::Io(e)) => {
                        trace!("DiskCache::get_raw({}): IoError: {:?}", key, e);
                        Err(e.into())
                    }
                    Err(_) => unreachable!(),
                },
            )
            .await?
    }

    async fn get_path(&self, key: &str) -> GetPathResult {
        let rel_path = make_key_path(key);
        let lru = self.lru.clone();
        self.pool
            .spawn_blocking(move || {
                match lru
                    .lock()
                    .unwrap()
                    .get_or_init()
                    .ok()
                    .and_then(|c| c.get_abs_path(&rel_path))
                {
                    Some(p) => GetPathResult::Found(p),
                    None => GetPathResult::Miss,
                }
            })
            .await
            .unwrap_or(GetPathResult::Miss)
    }

    async fn put(&self, key: &str, mut entry: CacheWrite) -> Result<Duration> {
        trace!("DiskCache::put({})", key);
        // File staged clones under their content hashes first, so the entry
        // never refers to a clone that isn't there yet. Identical outputs from
        // other compiles share one stored clone.
        let clones = entry.take_clones();
        if !clones.is_empty() {
            let lru = self.lru.clone();
            self.pool
                .spawn_blocking(move || -> Result<()> {
                    for clone in clones {
                        let clone_key = make_clone_key_path(&clone.hash);
                        let mut lru = lru.lock().unwrap();
                        let lru = lru.get_or_init()?;
                        if lru.contains_key(&clone_key) {
                            continue; // `clone.path` is deleted on drop
                        }
                        lru.insert_file(&clone_key, &*clone.path)?;
                        // insert_file moved it into place.
                        clone.path.keep()?;
                    }
                    Ok(())
                })
                .await??;
        }
        // Delegate to put_raw after serializing the entry
        let data = entry.finish()?;
        self.put_raw(key, data.into()).await
    }

    fn clone_staging_dir(&self) -> Option<PathBuf> {
        self.clone_staging
            .get_or_init(|| {
                let disabled = std::env::var("SCCACHE_DISK_CLONES")
                    .is_ok_and(|v| matches!(v.as_str(), "0" | "false" | "off"));
                if disabled || self.rw_mode == CacheMode::ReadOnly {
                    return None;
                }
                let dir = self.root.join("clone-staging");
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    debug!("Couldn't create {}: {e}", dir.display());
                    return None;
                }
                crate::util::clones_supported(&dir).then_some(dir)
            })
            .clone()
    }

    async fn put_raw(&self, key: &str, data: Bytes) -> Result<Duration> {
        trace!("DiskCache::put_raw({}, {} bytes)", key, data.len());

        if self.rw_mode == CacheMode::ReadOnly {
            return Err(anyhow!("Cannot write to a read-only cache"));
        }

        let lru = self.lru.clone();
        let key = make_key_path(key);

        self.pool
            .spawn_blocking(move || {
                let start = Instant::now();
                let mut f = lru
                    .lock()
                    .unwrap()
                    .get_or_init()?
                    .prepare_add(key, data.len() as u64)?;
                f.as_file_mut().write_all(&data)?;
                lru.lock().unwrap().get().unwrap().commit(f)?;
                Ok(start.elapsed())
            })
            .await?
    }

    async fn check(&self) -> Result<CacheMode> {
        Ok(self.rw_mode)
    }

    fn location(&self) -> String {
        format!("Local disk: {:?}", self.lru.lock().unwrap().path())
    }

    fn cache_type_name(&self) -> &'static str {
        "disk"
    }

    async fn current_size(&self) -> Result<Option<u64>> {
        Ok(self.lru.lock().unwrap().get().map(|l| l.size()))
    }
    async fn max_size(&self) -> Result<Option<u64>> {
        Ok(Some(self.lru.lock().unwrap().capacity()))
    }
    fn preprocessor_cache_mode_config(&self) -> PreprocessorCacheModeConfig {
        self.preprocessor_cache_mode_config
    }
    fn basedirs(&self) -> &[Vec<u8>] {
        &self.basedirs
    }
    async fn get_preprocessor_cache_entry(&self, key: &str) -> Result<Option<Box<dyn ReadSeek>>> {
        let key = normalize_key(key);
        Ok(self
            .preprocessor_cache
            .lock()
            .unwrap()
            .get_or_init()?
            .get(key)
            .ok())
    }
    async fn put_preprocessor_cache_entry(
        &self,
        key: &str,
        preprocessor_cache_entry: PreprocessorCacheEntry,
    ) -> Result<()> {
        if self.rw_mode == CacheMode::ReadOnly {
            return Err(anyhow!("Cannot write to a read-only cache"));
        }

        let key = normalize_key(key);
        let mut f = self
            .preprocessor_cache
            .lock()
            .unwrap()
            .get_or_init()?
            .prepare_add(key, 0)?;
        preprocessor_cache_entry.serialize_to(BufWriter::new(f.as_file_mut()))?;
        Ok(self
            .preprocessor_cache
            .lock()
            .unwrap()
            .get()
            .unwrap()
            .commit(f)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disk_cache_type_name() {
        let tempdir = tempfile::tempdir().unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        let disk = DiskCache::new(
            tempdir.path(),
            1024 * 1024,
            runtime.handle(),
            PreprocessorCacheModeConfig::default(),
            CacheMode::ReadWrite,
            vec![],
        );

        assert_eq!(disk.cache_type_name(), "disk");
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn test_disk_cache_clones_round_trip() {
        use crate::cache::FileObjectSource;
        let cache_dir = tempfile::tempdir().unwrap();
        let work = tempfile::tempdir().unwrap();
        if !crate::util::clones_supported(work.path()) {
            // tmpfs/ext4 CI runners can't clone; nothing to test there.
            return;
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let disk = DiskCache::new(
            cache_dir.path(),
            1024 * 1024 * 1024,
            runtime.handle(),
            PreprocessorCacheModeConfig::default(),
            CacheMode::ReadWrite,
            vec![],
        );
        let staging = disk.clone_staging_dir().expect("clones supported");

        let output = work.path().join("libfoo.rlib");
        std::fs::write(&output, vec![42u8; 256 * 1024]).unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&output, std::fs::Permissions::from_mode(0o640)).unwrap();
        }
        let objects = vec![FileObjectSource {
            key: "libfoo.rlib".into(),
            path: output.clone(),
            optional: false,
        }];
        let mut entry = runtime
            .block_on(CacheWrite::from_objects_cloned(
                objects,
                staging.clone(),
                runtime.handle(),
            ))
            .unwrap();
        entry.put_stdout(b"compiled").unwrap();
        runtime
            .block_on(disk.put("0123456789abcdef", entry))
            .unwrap();
        assert_eq!(
            std::fs::read_dir(&staging).unwrap().count(),
            0,
            "staged clones filed away"
        );

        // The output being rebuilt afterwards doesn't touch the cached copy.
        std::fs::write(&output, b"rebuilt").unwrap();

        let restored = work.path().join("restored/libfoo.rlib");
        std::fs::create_dir_all(restored.parent().unwrap()).unwrap();
        let Cache::Hit(mut hit) = runtime.block_on(disk.get("0123456789abcdef")).unwrap() else {
            panic!("expected a hit");
        };
        assert_eq!(hit.get_stdout(), b"compiled");
        let objects = vec![FileObjectSource {
            key: "libfoo.rlib".into(),
            path: restored.clone(),
            optional: false,
        }];
        runtime
            .block_on(hit.extract_objects(objects, runtime.handle()))
            .unwrap();
        assert_eq!(std::fs::read(&restored).unwrap(), vec![42u8; 256 * 1024]);
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&restored).unwrap().permissions().mode() & 0o777,
            0o640
        );
        let age = std::fs::metadata(&restored)
            .unwrap()
            .modified()
            .unwrap()
            .elapsed()
            .unwrap();
        assert!(age.as_secs() < 60, "restored outputs look freshly written");

        // An evicted clone makes the entry a miss rather than a broken hit.
        let mut lru = disk.lru.lock().unwrap();
        let clone_keys: Vec<_> = walkdir::WalkDir::new(cache_dir.path().join("clones"))
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().strip_prefix(cache_dir.path()).unwrap().to_owned())
            .collect();
        assert_eq!(clone_keys.len(), 1);
        lru.get_or_init().unwrap().remove(&clone_keys[0]).unwrap();
        drop(lru);
        assert!(matches!(
            runtime.block_on(disk.get("0123456789abcdef")).unwrap(),
            Cache::Miss
        ));
    }

    #[test]
    fn test_disk_cache_clones_can_be_disabled() {
        let tempdir = tempfile::tempdir().unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let disk = DiskCache::new(
            tempdir.path(),
            1024 * 1024,
            runtime.handle(),
            PreprocessorCacheModeConfig::default(),
            CacheMode::ReadOnly,
            vec![],
        );
        assert_eq!(
            disk.clone_staging_dir(),
            None,
            "a read-only cache never stores clones"
        );
    }
}
