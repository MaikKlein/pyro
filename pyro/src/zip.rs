use crate::{expand, Index};
#[cfg(feature = "threading")]
use rayon::iter::{
    plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};
use std::marker::PhantomData;

pub struct ZipSlice<'a, Tuple> {
    tuple: Tuple,
    idx: usize,
    _m: std::marker::PhantomData<&'a ()>,
}
impl<Tuple> ZipSlice<'_, Tuple> {
    pub fn new(tuple: Tuple) -> Self {
        ZipSlice {
            tuple,
            idx: 0,
            _m: PhantomData,
        }
    }
}

#[cfg(feature = "threading")]
struct ZipProducer<'a, Tuple>(ZipSlice<'a, Tuple>);

macro_rules! impl_zip_iterator {
    ($($ty: ident),*) => {
        impl<'a, $($ty,)*> Iterator for ZipSlice<'a, ($($ty,)*)>
        where
            $(
                $ty: Index<'a>,
            )*
        {
            type Item = ($($ty::Item,)*);
            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let idx = self.idx;
                let len = self.tuple.0.len();
                if idx >= len {
                    return None;
                }
                #[allow(non_snake_case)]
                let ($($ty,)*) = &self.tuple;
                let r = unsafe {
                    Some(
                            ($($ty.get_unchecked(idx),)*)
                    )
                };
                self.idx += 1;
                r
            }
        }
        impl<'a, $($ty,)*> ExactSizeIterator for ZipSlice<'a, ($($ty,)*)>
        where
            $(
                $ty: Index<'a>,
            )*
        {
            fn len(&self) -> usize {
                self.tuple.0.len()
            }
        }
        impl<'a, $($ty,)*> DoubleEndedIterator for ZipSlice<'a, ($($ty,)*)>
        where
            $(
                $ty: Index<'a>,
            )*
        {
            fn next_back(&mut self) -> Option<Self::Item> {
                unimplemented!()
            }
        }

        #[cfg(feature = "threading")]
        impl<'a, $($ty,)*> Producer for ZipProducer<'a, ($($ty,)*)>
        where
            $(
                $ty: Index<'a> + Send + Sync,
            )*
        {
            type Item = <Self::IntoIter as Iterator>::Item;
            type IntoIter = ZipSlice<'a, ($($ty,)*)>;
            fn into_iter(self) -> Self::IntoIter {
                self.0
            }
            fn split_at(self, index: usize) -> (Self, Self) {
                #[allow(non_snake_case)]
                let ($($ty,)*) = self.0.tuple;
                $(
                    #[allow(non_snake_case)]
                    let $ty = $ty.split_at(index);
                )*
                let left = ZipProducer(ZipSlice::new(($($ty.0,)*)));
                let right = ZipProducer(ZipSlice::new(($($ty.1,)*)));
                (left, right)
            }
        }
        #[cfg(feature = "threading")]
        impl<'a, $($ty,)*> ParallelIterator for ZipSlice<'a, ($($ty,)*)>
        where
            $(
                $ty: Index<'a> + Sync + Send,
                $ty::Item: Sync + Send,
            )*
            <Self as Iterator>::Item: Sync + Send,
        {
            type Item = <Self as Iterator>::Item;

            fn drive_unindexed<C1>(self, consumer: C1) -> C1::Result
            where
                C1: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }

            fn opt_len(&self) -> Option<usize> {
                Some(ExactSizeIterator::len(self))
            }
        }
        #[cfg(feature = "threading")]
        impl<'a, $($ty,)*> IndexedParallelIterator for ZipSlice<'a, ($($ty,)*)>
        where
            $(
                $ty: Index<'a> + Sync + Send,
                $ty::Item: Sync + Send,
            )*
        {
            fn drive<C1>(self, consumer: C1) -> C1::Result
            where
                C1: Consumer<Self::Item>,
            {
                bridge(self, consumer)
            }

            fn len(&self) -> usize {
                ExactSizeIterator::len(self)
            }

            fn with_producer<CB>(self, callback: CB) -> CB::Output
            where
                CB: ProducerCallback<Self::Item>,
            {
                callback.callback(ZipProducer(self))
            }
        }
    };
}

expand!(impl_zip_iterator, A, B, C, D, E, F, G, H, I);
