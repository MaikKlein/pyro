use crate::{Slice, SliceMut};
use std::{any::TypeId, collections::HashMap, marker::PhantomData, mem::size_of};
pub const CHUNK_SIZE: usize = 64;
#[repr(transparent)]
pub struct Chunk {
    data: [u8; CHUNK_SIZE],
}
impl Chunk {
    pub fn zeroed() -> Self {
        Chunk {
            data: [0; CHUNK_SIZE],
        }
    }
    pub unsafe fn as_slice<'a, T>(&'a self) -> Slice<'a, T> {
        Slice::from_raw(self.data.as_ptr() as *const T, CHUNK_SIZE / size_of::<T>())
    }
    pub unsafe fn as_slice_mut<'a, T>(&'a mut self) -> SliceMut<'a, T> {
        SliceMut::from_raw(
            self.data.as_mut_ptr() as *mut T,
            CHUNK_SIZE / size_of::<T>(),
        )
    }
}
pub struct Allocator {
    pub data: Vec<u8>,
    pub free_list: Vec<usize>,
}

impl Allocator {
    pub fn alloc(&mut self) -> *mut Chunk {
        unimplemented!()
    }
}

pub struct Storage {
    alloc: Allocator,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_as_slice() {
        let mut chunk = Chunk::zeroed();
        unsafe {
            {
                chunk.as_slice_mut::<bool>().get_mut(0).write(true);
            }
            assert_eq!(chunk.data[0], 1);
        }
    }
}

trait MaybeDrop {
    unsafe fn maybe_drop(&self, ptr: *mut ());
}

struct Deleter<T> {
    _marker: PhantomData<T>,
}

impl<T> Default for Deleter<T> {
    fn default() -> Self {
        Deleter {
            _marker: PhantomData,
        }
    }
}

impl<T> MaybeDrop for Deleter<T> {
    unsafe fn maybe_drop(&self, drop: *mut ()) {
        let drop = drop as *mut T;
        if std::mem::needs_drop::<T>() {
            std::ptr::drop_in_place(drop);
        }
    }
}
