use std::ops::Deref;

pub struct Sealed<T> {
    inner: T,
}

// Safety: T is now Sync because it is immutable
unsafe impl<T> Sync for Sealed<T> where T: Send {}

impl<T> Sealed<T> {
    pub fn new(inner: T) -> Self {
        Sealed { inner }
    }
}

impl<T> Deref for Sealed<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
