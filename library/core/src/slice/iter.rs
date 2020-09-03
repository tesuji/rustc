//! Definitions of a bunch of iterators for `[T]`.

use crate::cmp;
use crate::cmp::Ordering;
use crate::fmt;
use crate::intrinsics::{assume, exact_div, unchecked_sub};
use crate::iter::{FusedIterator, TrustedLen, TrustedRandomAccess};
use crate::marker::{self, Send, Sized, Sync};
use crate::mem;
use crate::ptr::NonNull;

use super::{from_raw_parts, from_raw_parts_mut};

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a [T] {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut [T] {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

// Macro helper functions
#[inline(always)]
fn size_from_ptr<T>(_: *const T) -> usize {
    mem::size_of::<T>()
}

// Inlining is_empty and len makes a huge performance difference
macro_rules! is_empty {
    // The way we encode the length of a ZST iterator, this works both for ZST
    // and non-ZST.
    ($self: ident) => {
        $self.ptr.as_ptr() as *const T == $self.end
    };
}

// To get rid of some bounds checks (see `position`), we compute the length in a somewhat
// unexpected way. (Tested by `codegen/slice-position-bounds-check`.)
macro_rules! len {
    ($self: ident) => {{
        #![allow(unused_unsafe)] // we're sometimes used within an unsafe block

        let start = $self.ptr;
        let size = size_from_ptr(start.as_ptr());
        if size == 0 {
            // This _cannot_ use `unchecked_sub` because we depend on wrapping
            // to represent the length of long ZST slice iterators.
            ($self.end as usize).wrapping_sub(start.as_ptr() as usize)
        } else {
            // We know that `start <= end`, so can do better than `offset_from`,
            // which needs to deal in signed.  By setting appropriate flags here
            // we can tell LLVM this, which helps it remove bounds checks.
            // SAFETY: By the type invariant, `start <= end`
            let diff = unsafe { unchecked_sub($self.end as usize, start.as_ptr() as usize) };
            // By also telling LLVM that the pointers are apart by an exact
            // multiple of the type size, it can optimize `len() == 0` down to
            // `start == end` instead of `(end - start) < size`.
            // SAFETY: By the type invariant, the pointers are aligned so the
            //         distance between them must be a multiple of pointee size
            unsafe { exact_div(diff, size) }
        }
    }};
}

// The shared definition of the `Iter` and `IterMut` iterators
macro_rules! iterator {
    (
        struct $name:ident -> $ptr:ty,
        $elem:ty,
        $raw_mut:tt,
        {$( $mut_:tt )?},
        {$($extra:tt)*}
    ) => {
        // Returns the first element and moves the start of the iterator forwards by 1.
        // Greatly improves performance compared to an inlined function. The iterator
        // must not be empty.
        macro_rules! next_unchecked {
            ($self: ident) => {& $( $mut_ )? *$self.post_inc_start(1)}
        }

        // Returns the last element and moves the end of the iterator backwards by 1.
        // Greatly improves performance compared to an inlined function. The iterator
        // must not be empty.
        macro_rules! next_back_unchecked {
            ($self: ident) => {& $( $mut_ )? *$self.pre_dec_end(1)}
        }

        // Shrinks the iterator when T is a ZST, by moving the end of the iterator
        // backwards by `n`. `n` must not exceed `self.len()`.
        macro_rules! zst_shrink {
            ($self: ident, $n: ident) => {
                $self.end = ($self.end as * $raw_mut u8).wrapping_offset(-$n) as * $raw_mut T;
            }
        }

        impl<'a, T> $name<'a, T> {
            // Helper function for creating a slice from the iterator.
            #[inline(always)]
            fn make_slice(&self) -> &'a [T] {
                // SAFETY: the iterator was created from a slice with pointer
                // `self.ptr` and length `len!(self)`. This guarantees that all
                // the prerequisites for `from_raw_parts` are fulfilled.
                unsafe { from_raw_parts(self.ptr.as_ptr(), len!(self)) }
            }

            // Helper function for moving the start of the iterator forwards by `offset` elements,
            // returning the old start.
            // Unsafe because the offset must not exceed `self.len()`.
            #[inline(always)]
            unsafe fn post_inc_start(&mut self, offset: isize) -> * $raw_mut T {
                if mem::size_of::<T>() == 0 {
                    zst_shrink!(self, offset);
                    self.ptr.as_ptr()
                } else {
                    let old = self.ptr.as_ptr();
                    // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                    // so this new pointer is inside `self` and thus guaranteed to be non-null.
                    self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().offset(offset)) };
                    old
                }
            }

            // Helper function for moving the end of the iterator backwards by `offset` elements,
            // returning the new end.
            // Unsafe because the offset must not exceed `self.len()`.
            #[inline(always)]
            unsafe fn pre_dec_end(&mut self, offset: isize) -> * $raw_mut T {
                if mem::size_of::<T>() == 0 {
                    zst_shrink!(self, offset);
                    self.ptr.as_ptr()
                } else {
                    // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                    // which is guaranteed to not overflow an `isize`. Also, the resulting pointer
                    // is in bounds of `slice`, which fulfills the other requirements for `offset`.
                    self.end = unsafe { self.end.offset(-offset) };
                    self.end
                }
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<T> ExactSizeIterator for $name<'_, T> {
            #[inline(always)]
            fn len(&self) -> usize {
                len!(self)
            }

            #[inline(always)]
            fn is_empty(&self) -> bool {
                is_empty!(self)
            }
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> Iterator for $name<'a, T> {
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks

                // SAFETY: `assume` calls are safe since a slice's start pointer
                // must be non-null, and slices over non-ZSTs must also have a
                // non-null end pointer. The call to `next_unchecked!` is safe
                // since we check if the iterator is empty first.
                unsafe {
                    assume(!self.ptr.as_ptr().is_null());
                    if mem::size_of::<T>() != 0 {
                        assume(!self.end.is_null());
                    }
                    if is_empty!(self) {
                        None
                    } else {
                        Some(next_unchecked!(self))
                    }
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let exact = len!(self);
                (exact, Some(exact))
            }

            #[inline]
            fn count(self) -> usize {
                len!(self)
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<$elem> {
                if n >= len!(self) {
                    // This iterator is now empty.
                    if mem::size_of::<T>() == 0 {
                        // We have to do it this way as `ptr` may never be 0, but `end`
                        // could be (due to wrapping).
                        self.end = self.ptr.as_ptr();
                    } else {
                        // SAFETY: end can't be 0 if T isn't ZST because ptr isn't 0 and end >= ptr
                        unsafe {
                            self.ptr = NonNull::new_unchecked(self.end as *mut T);
                        }
                    }
                    return None;
                }
                // SAFETY: We are in bounds. `post_inc_start` does the right thing even for ZSTs.
                unsafe {
                    self.post_inc_start(n as isize);
                    Some(next_unchecked!(self))
                }
            }

            #[inline]
            fn last(mut self) -> Option<$elem> {
                self.next_back()
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile.
            #[inline]
            fn for_each<F>(mut self, mut f: F)
            where
                Self: Sized,
                F: FnMut(Self::Item),
            {
                while let Some(x) = self.next() {
                    f(x);
                }
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile.
            #[inline]
            fn all<F>(&mut self, mut f: F) -> bool
            where
                Self: Sized,
                F: FnMut(Self::Item) -> bool,
            {
                while let Some(x) = self.next() {
                    if !f(x) {
                        return false;
                    }
                }
                true
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile.
            #[inline]
            fn any<F>(&mut self, mut f: F) -> bool
            where
                Self: Sized,
                F: FnMut(Self::Item) -> bool,
            {
                while let Some(x) = self.next() {
                    if f(x) {
                        return true;
                    }
                }
                false
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile.
            #[inline]
            fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item>
            where
                Self: Sized,
                P: FnMut(&Self::Item) -> bool,
            {
                while let Some(x) = self.next() {
                    if predicate(&x) {
                        return Some(x);
                    }
                }
                None
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile.
            #[inline]
            fn find_map<B, F>(&mut self, mut f: F) -> Option<B>
            where
                Self: Sized,
                F: FnMut(Self::Item) -> Option<B>,
            {
                while let Some(x) = self.next() {
                    if let Some(y) = f(x) {
                        return Some(y);
                    }
                }
                None
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile. Also, the `assume` avoids a bounds check.
            #[inline]
            #[rustc_inherit_overflow_checks]
            fn position<P>(&mut self, mut predicate: P) -> Option<usize> where
                Self: Sized,
                P: FnMut(Self::Item) -> bool,
            {
                let n = len!(self);
                let mut i = 0;
                while let Some(x) = self.next() {
                    if predicate(x) {
                        // SAFETY: we are guaranteed to be in bounds by the loop invariant:
                        // when `i >= n`, `self.next()` returns `None` and the loop breaks.
                        unsafe { assume(i < n) };
                        return Some(i);
                    }
                    i += 1;
                }
                None
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile. Also, the `assume` avoids a bounds check.
            #[inline]
            fn rposition<P>(&mut self, mut predicate: P) -> Option<usize> where
                P: FnMut(Self::Item) -> bool,
                Self: Sized + ExactSizeIterator + DoubleEndedIterator
            {
                let n = len!(self);
                let mut i = n;
                while let Some(x) = self.next_back() {
                    i -= 1;
                    if predicate(x) {
                        // SAFETY: `i` must be lower than `n` since it starts at `n`
                        // and is only decreasing.
                        unsafe { assume(i < n) };
                        return Some(i);
                    }
                }
                None
            }

            #[doc(hidden)]
            unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
                // SAFETY: the caller must guarantee that `i` is in bounds of
                // the underlying slice, so `i` cannot overflow an `isize`, and
                // the returned references is guaranteed to refer to an element
                // of the slice and thus guaranteed to be valid.
                //
                // Also note that the caller also guarantees that we're never
                // called with the same index again, and that no other methods
                // that will access this subslice are called, so it is valid
                // for the returned reference to be mutable in the case of
                // `IterMut`
                unsafe { & $( $mut_ )? * self.ptr.as_ptr().add(idx) }
            }

            $($extra)*
        }

        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<$elem> {
                // could be implemented with slices, but this avoids bounds checks

                // SAFETY: `assume` calls are safe since a slice's start pointer must be non-null,
                // and slices over non-ZSTs must also have a non-null end pointer.
                // The call to `next_back_unchecked!` is safe since we check if the iterator is
                // empty first.
                unsafe {
                    assume(!self.ptr.as_ptr().is_null());
                    if mem::size_of::<T>() != 0 {
                        assume(!self.end.is_null());
                    }
                    if is_empty!(self) {
                        None
                    } else {
                        Some(next_back_unchecked!(self))
                    }
                }
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<$elem> {
                if n >= len!(self) {
                    // This iterator is now empty.
                    self.end = self.ptr.as_ptr();
                    return None;
                }
                // SAFETY: We are in bounds. `pre_dec_end` does the right thing even for ZSTs.
                unsafe {
                    self.pre_dec_end(n as isize);
                    Some(next_back_unchecked!(self))
                }
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<T> FusedIterator for $name<'_, T> {}

        #[unstable(feature = "trusted_len", issue = "37572")]
        unsafe impl<T> TrustedLen for $name<'_, T> {}
    }
}

/// Immutable slice iterator
///
/// This struct is created by the [`iter`] method on [slices].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter` method to get the `Iter` struct (&[usize here]):
/// let slice = &[1, 2, 3];
///
/// // Then, we iterate over it:
/// for element in slice.iter() {
///     println!("{}", element);
/// }
/// ```
///
/// [`iter`]: ../../std/primitive.slice.html#method.iter
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    pub(super) ptr: NonNull<T>,
    pub(super) end: *const T, // If T is a ZST, this is actually ptr+len.  This encoding is picked so that
    // ptr == end is a quick test for the Iterator being empty, that works
    // for both ZST and non-ZST.
    pub(super) _marker: marker::PhantomData<&'a T>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter").field(&self.as_slice()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for Iter<'_, T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Send for Iter<'_, T> {}

impl<'a, T> Iter<'a, T> {
    /// Views the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // First, we declare a type which has the `iter` method to get the `Iter`
    /// // struct (&[usize here]):
    /// let slice = &[1, 2, 3];
    ///
    /// // Then, we get the iterator:
    /// let mut iter = slice.iter();
    /// // So if we print what `as_slice` method returns here, we have "[1, 2, 3]":
    /// println!("{:?}", iter.as_slice());
    ///
    /// // Next, we move to the second element of the slice:
    /// iter.next();
    /// // Now `as_slice` returns "[2, 3]":
    /// println!("{:?}", iter.as_slice());
    /// ```
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    pub fn as_slice(&self) -> &'a [T] {
        self.make_slice()
    }
}

iterator! {struct Iter -> *const T, &'a T, const, {/* no mut */}, {
    fn is_sorted_by<F>(self, mut compare: F) -> bool
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Option<Ordering>,
    {
        self.as_slice().windows(2).all(|w| {
            compare(&&w[0], &&w[1]).map(|o| o != Ordering::Greater).unwrap_or(false)
        })
    }
}}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter { ptr: self.ptr, end: self.end, _marker: self._marker }
    }
}

#[stable(feature = "slice_iter_as_ref", since = "1.13.0")]
impl<T> AsRef<[T]> for Iter<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

/// Mutable slice iterator.
///
/// This struct is created by the [`iter_mut`] method on [slices].
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// // First, we declare a type which has `iter_mut` method to get the `IterMut`
/// // struct (&[usize here]):
/// let mut slice = &mut [1, 2, 3];
///
/// // Then, we iterate over it and increment each element value:
/// for element in slice.iter_mut() {
///     *element += 1;
/// }
///
/// // We now have "[2, 3, 4]":
/// println!("{:?}", slice);
/// ```
///
/// [`iter_mut`]: ../../std/primitive.slice.html#method.iter_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    pub(super) ptr: NonNull<T>,
    pub(super) end: *mut T, // If T is a ZST, this is actually ptr+len.  This encoding is picked so that
    // ptr == end is a quick test for the Iterator being empty, that works
    // for both ZST and non-ZST.
    pub(super) _marker: marker::PhantomData<&'a mut T>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IterMut").field(&self.make_slice()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for IterMut<'_, T> {}
#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for IterMut<'_, T> {}

impl<'a, T> IterMut<'a, T> {
    /// Views the underlying data as a subslice of the original data.
    ///
    /// To avoid creating `&mut` references that alias, this is forced
    /// to consume the iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// // First, we declare a type which has `iter_mut` method to get the `IterMut`
    /// // struct (&[usize here]):
    /// let mut slice = &mut [1, 2, 3];
    ///
    /// {
    ///     // Then, we get the iterator:
    ///     let mut iter = slice.iter_mut();
    ///     // We move to next element:
    ///     iter.next();
    ///     // So if we print what `into_slice` method returns here, we have "[2, 3]":
    ///     println!("{:?}", iter.into_slice());
    /// }
    ///
    /// // Now let's modify a value of the slice:
    /// {
    ///     // First we get back the iterator:
    ///     let mut iter = slice.iter_mut();
    ///     // We change the value of the first element of the slice returned by the `next` method:
    ///     *iter.next().unwrap() += 1;
    /// }
    /// // Now slice is "[2, 2, 3]":
    /// println!("{:?}", slice);
    /// ```
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    pub fn into_slice(self) -> &'a mut [T] {
        // SAFETY: the iterator was created from a mutable slice with pointer
        // `self.ptr` and length `len!(self)`. This guarantees that all the prerequisites
        // for `from_raw_parts_mut` are fulfilled.
        unsafe { from_raw_parts_mut(self.ptr.as_ptr(), len!(self)) }
    }

    /// Views the underlying data as a subslice of the original data.
    ///
    /// To avoid creating `&mut [T]` references that alias, the returned slice
    /// borrows its lifetime from the iterator the method is applied on.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # #![feature(slice_iter_mut_as_slice)]
    /// let mut slice: &mut [usize] = &mut [1, 2, 3];
    ///
    /// // First, we get the iterator:
    /// let mut iter = slice.iter_mut();
    /// // So if we check what the `as_slice` method returns here, we have "[1, 2, 3]":
    /// assert_eq!(iter.as_slice(), &[1, 2, 3]);
    ///
    /// // Next, we move to the second element of the slice:
    /// iter.next();
    /// // Now `as_slice` returns "[2, 3]":
    /// assert_eq!(iter.as_slice(), &[2, 3]);
    /// ```
    #[unstable(feature = "slice_iter_mut_as_slice", reason = "recently added", issue = "58957")]
    pub fn as_slice(&self) -> &[T] {
        self.make_slice()
    }
}

iterator! {struct IterMut -> *mut T, &'a mut T, mut, {mut}, {}}

/// An internal abstraction over the splitting iterators, so that
/// splitn, splitn_mut etc can be implemented once.
#[doc(hidden)]
pub(super) trait SplitIter: DoubleEndedIterator {
    /// Marks the underlying iterator as complete, extracting the remaining
    /// portion of the slice.
    fn finish(&mut self) -> Option<Self::Item>;
}

/// An iterator over subslices separated by elements that match a predicate
/// function.
///
/// This struct is created by the [`split`] method on [slices].
///
/// [`split`]: ../../std/primitive.slice.html#method.split
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Split<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) v: &'a [T],
    pub(super) pred: P,
    pub(super) finished: bool,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for Split<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Split").field("v", &self.v).field("finished", &self.finished).finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T, P> Clone for Split<'_, T, P>
where
    P: Clone + FnMut(&T) -> bool,
{
    fn clone(&self) -> Self {
        Split { v: self.v, pred: self.pred.clone(), finished: self.finished }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Iterator for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        match self.v.iter().position(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let ret = Some(&self.v[..idx]);
                self.v = &self.v[idx + 1..];
                ret
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished { (0, Some(0)) } else { (1, Some(self.v.len() + 1)) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> DoubleEndedIterator for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        match self.v.iter().rposition(|x| (self.pred)(x)) {
            None => self.finish(),
            Some(idx) => {
                let ret = Some(&self.v[idx + 1..]);
                self.v = &self.v[..idx];
                ret
            }
        }
    }
}

impl<'a, T, P> SplitIter for Split<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<&'a [T]> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(self.v)
        }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, P> FusedIterator for Split<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function. Unlike `Split`, it contains the matched part as a terminator
/// of the subslice.
///
/// This struct is created by the [`split_inclusive`] method on [slices].
///
/// [`split_inclusive`]: ../../std/primitive.slice.html#method.split_inclusive
/// [slices]: ../../std/primitive.slice.html
#[unstable(feature = "split_inclusive", issue = "72360")]
pub struct SplitInclusive<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) v: &'a [T],
    pub(super) pred: P,
    pub(super) finished: bool,
}

#[unstable(feature = "split_inclusive", issue = "72360")]
impl<T: fmt::Debug, P> fmt::Debug for SplitInclusive<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitInclusive")
            .field("v", &self.v)
            .field("finished", &self.finished)
            .finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[unstable(feature = "split_inclusive", issue = "72360")]
impl<T, P> Clone for SplitInclusive<'_, T, P>
where
    P: Clone + FnMut(&T) -> bool,
{
    fn clone(&self) -> Self {
        SplitInclusive { v: self.v, pred: self.pred.clone(), finished: self.finished }
    }
}

#[unstable(feature = "split_inclusive", issue = "72360")]
impl<'a, T, P> Iterator for SplitInclusive<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        let idx =
            self.v.iter().position(|x| (self.pred)(x)).map(|idx| idx + 1).unwrap_or(self.v.len());
        if idx == self.v.len() {
            self.finished = true;
        }
        let ret = Some(&self.v[..idx]);
        self.v = &self.v[idx..];
        ret
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished { (0, Some(0)) } else { (1, Some(self.v.len() + 1)) }
    }
}

#[unstable(feature = "split_inclusive", issue = "72360")]
impl<'a, T, P> DoubleEndedIterator for SplitInclusive<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.finished {
            return None;
        }

        // The last index of self.v is already checked and found to match
        // by the last iteration, so we start searching a new match
        // one index to the left.
        let remainder = if self.v.is_empty() { &[] } else { &self.v[..(self.v.len() - 1)] };
        let idx = remainder.iter().rposition(|x| (self.pred)(x)).map(|idx| idx + 1).unwrap_or(0);
        if idx == 0 {
            self.finished = true;
        }
        let ret = Some(&self.v[idx..]);
        self.v = &self.v[..idx];
        ret
    }
}

#[unstable(feature = "split_inclusive", issue = "72360")]
impl<T, P> FusedIterator for SplitInclusive<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the mutable subslices of the vector which are separated
/// by elements that match `pred`.
///
/// This struct is created by the [`split_mut`] method on [slices].
///
/// [`split_mut`]: ../../std/primitive.slice.html#method.split_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) v: &'a mut [T],
    pub(super) pred: P,
    pub(super) finished: bool,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitMut").field("v", &self.v).field("finished", &self.finished).finish()
    }
}

impl<'a, T, P> SplitIter for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            None
        } else {
            self.finished = true;
            Some(mem::replace(&mut self.v, &mut []))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> Iterator for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            return None;
        }

        let idx_opt = {
            // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().position(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.split_at_mut(idx);
                self.v = &mut tail[1..];
                Some(head)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // if the predicate doesn't match anything, we yield one slice
            // if it matches every element, we yield len+1 empty slices.
            (1, Some(self.v.len() + 1))
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T, P> DoubleEndedIterator for SplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            return None;
        }

        let idx_opt = {
            // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().rposition(|x| (*pred)(x))
        };
        match idx_opt {
            None => self.finish(),
            Some(idx) => {
                let tmp = mem::replace(&mut self.v, &mut []);
                let (head, tail) = tmp.split_at_mut(idx);
                self.v = head;
                Some(&mut tail[1..])
            }
        }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<T, P> FusedIterator for SplitMut<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the mutable subslices of the vector which are separated
/// by elements that match `pred`. Unlike `SplitMut`, it contains the matched
/// parts in the ends of the subslices.
///
/// This struct is created by the [`split_inclusive_mut`] method on [slices].
///
/// [`split_inclusive_mut`]: ../../std/primitive.slice.html#method.split_inclusive_mut
/// [slices]: ../../std/primitive.slice.html
#[unstable(feature = "split_inclusive", issue = "72360")]
pub struct SplitInclusiveMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) v: &'a mut [T],
    pub(super) pred: P,
    pub(super) finished: bool,
}

#[unstable(feature = "split_inclusive", issue = "72360")]
impl<T: fmt::Debug, P> fmt::Debug for SplitInclusiveMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitInclusiveMut")
            .field("v", &self.v)
            .field("finished", &self.finished)
            .finish()
    }
}

#[unstable(feature = "split_inclusive", issue = "72360")]
impl<'a, T, P> Iterator for SplitInclusiveMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            return None;
        }

        let idx_opt = {
            // work around borrowck limitations
            let pred = &mut self.pred;
            self.v.iter().position(|x| (*pred)(x))
        };
        let idx = idx_opt.map(|idx| idx + 1).unwrap_or(self.v.len());
        if idx == self.v.len() {
            self.finished = true;
        }
        let tmp = mem::replace(&mut self.v, &mut []);
        let (head, tail) = tmp.split_at_mut(idx);
        self.v = tail;
        Some(head)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            // if the predicate doesn't match anything, we yield one slice
            // if it matches every element, we yield len+1 empty slices.
            (1, Some(self.v.len() + 1))
        }
    }
}

#[unstable(feature = "split_inclusive", issue = "72360")]
impl<'a, T, P> DoubleEndedIterator for SplitInclusiveMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.finished {
            return None;
        }

        let idx_opt = if self.v.is_empty() {
            None
        } else {
            // work around borrowck limitations
            let pred = &mut self.pred;

            // The last index of self.v is already checked and found to match
            // by the last iteration, so we start searching a new match
            // one index to the left.
            let remainder = &self.v[..(self.v.len() - 1)];
            remainder.iter().rposition(|x| (*pred)(x))
        };
        let idx = idx_opt.map(|idx| idx + 1).unwrap_or(0);
        if idx == 0 {
            self.finished = true;
        }
        let tmp = mem::replace(&mut self.v, &mut []);
        let (head, tail) = tmp.split_at_mut(idx);
        self.v = head;
        Some(tail)
    }
}

#[unstable(feature = "split_inclusive", issue = "72360")]
impl<T, P> FusedIterator for SplitInclusiveMut<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over subslices separated by elements that match a predicate
/// function, starting from the end of the slice.
///
/// This struct is created by the [`rsplit`] method on [slices].
///
/// [`rsplit`]: ../../std/primitive.slice.html#method.rsplit
/// [slices]: ../../std/primitive.slice.html
#[derive(Clone)] // Is this correct, or does it incorrectly require `T: Clone`?
#[stable(feature = "slice_rsplit", since = "1.27.0")]
pub struct RSplit<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) inner: Split<'a, T, P>,
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplit<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RSplit")
            .field("v", &self.inner.v)
            .field("finished", &self.inner.finished)
            .finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> Iterator for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> DoubleEndedIterator for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        self.inner.next()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> SplitIter for RSplit<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<&'a [T]> {
        self.inner.finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T, P> FusedIterator for RSplit<'_, T, P> where P: FnMut(&T) -> bool {}

/// An iterator over the subslices of the vector which are separated
/// by elements that match `pred`, starting from the end of the slice.
///
/// This struct is created by the [`rsplit_mut`] method on [slices].
///
/// [`rsplit_mut`]: ../../std/primitive.slice.html#method.rsplit_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "slice_rsplit", since = "1.27.0")]
pub struct RSplitMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) inner: SplitMut<'a, T, P>,
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RSplitMut")
            .field("v", &self.inner.v)
            .field("finished", &self.inner.finished)
            .finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> SplitIter for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn finish(&mut self) -> Option<&'a mut [T]> {
        self.inner.finish()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> Iterator for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        self.inner.next_back()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<'a, T, P> DoubleEndedIterator for RSplitMut<'a, T, P>
where
    P: FnMut(&T) -> bool,
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        self.inner.next()
    }
}

#[stable(feature = "slice_rsplit", since = "1.27.0")]
impl<T, P> FusedIterator for RSplitMut<'_, T, P> where P: FnMut(&T) -> bool {}

/// An private iterator over subslices separated by elements that
/// match a predicate function, splitting at most a fixed number of
/// times.
#[derive(Debug)]
pub(super) struct GenericSplitN<I> {
    pub(super) iter: I,
    pub(super) count: usize,
}

impl<T, I: SplitIter<Item = T>> Iterator for GenericSplitN<I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.count {
            0 => None,
            1 => {
                self.count -= 1;
                self.iter.finish()
            }
            _ => {
                self.count -= 1;
                self.iter.next()
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper_opt) = self.iter.size_hint();
        (lower, upper_opt.map(|upper| cmp::min(self.count, upper)))
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn`] method on [slices].
///
/// [`splitn`]: ../../std/primitive.slice.html#method.splitn
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitN<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) inner: GenericSplitN<Split<'a, T, P>>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitN<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitN").field("inner", &self.inner).finish()
    }
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
///
/// This struct is created by the [`rsplitn`] method on [slices].
///
/// [`rsplitn`]: ../../std/primitive.slice.html#method.rsplitn
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitN<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) inner: GenericSplitN<RSplit<'a, T, P>>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitN<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RSplitN").field("inner", &self.inner).finish()
    }
}

/// An iterator over subslices separated by elements that match a predicate
/// function, limited to a given number of splits.
///
/// This struct is created by the [`splitn_mut`] method on [slices].
///
/// [`splitn_mut`]: ../../std/primitive.slice.html#method.splitn_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct SplitNMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) inner: GenericSplitN<SplitMut<'a, T, P>>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for SplitNMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitNMut").field("inner", &self.inner).finish()
    }
}

/// An iterator over subslices separated by elements that match a
/// predicate function, limited to a given number of splits, starting
/// from the end of the slice.
///
/// This struct is created by the [`rsplitn_mut`] method on [slices].
///
/// [`rsplitn_mut`]: ../../std/primitive.slice.html#method.rsplitn_mut
/// [slices]: ../../std/primitive.slice.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct RSplitNMut<'a, T: 'a, P>
where
    P: FnMut(&T) -> bool,
{
    pub(super) inner: GenericSplitN<RSplitMut<'a, T, P>>,
}

#[stable(feature = "core_impl_debug", since = "1.9.0")]
impl<T: fmt::Debug, P> fmt::Debug for RSplitNMut<'_, T, P>
where
    P: FnMut(&T) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RSplitNMut").field("inner", &self.inner).finish()
    }
}

macro_rules! forward_iterator {
    ($name:ident: $elem:ident, $iter_of:ty) => {
        #[stable(feature = "rust1", since = "1.0.0")]
        impl<'a, $elem, P> Iterator for $name<'a, $elem, P>
        where
            P: FnMut(&T) -> bool,
        {
            type Item = $iter_of;

            #[inline]
            fn next(&mut self) -> Option<$iter_of> {
                self.inner.next()
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                self.inner.size_hint()
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<'a, $elem, P> FusedIterator for $name<'a, $elem, P> where P: FnMut(&T) -> bool {}
    };
}

forward_iterator! { SplitN: T, &'a [T] }
forward_iterator! { RSplitN: T, &'a [T] }
forward_iterator! { SplitNMut: T, &'a mut [T] }
forward_iterator! { RSplitNMut: T, &'a mut [T] }

/// An iterator over overlapping subslices of length `size`.
///
/// This struct is created by the [`windows`] method on [slices].
///
/// [`windows`]: ../../std/primitive.slice.html#method.windows
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Windows<'a, T: 'a> {
    pub(super) v: &'a [T],
    pub(super) size: usize,
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Windows<'_, T> {
    fn clone(&self) -> Self {
        Windows { v: self.v, size: self.size }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Windows<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.size > self.v.len() {
            None
        } else {
            let ret = Some(&self.v[..self.size]);
            self.v = &self.v[1..];
            ret
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.size > self.v.len() {
            (0, Some(0))
        } else {
            let size = self.v.len() - self.size + 1;
            (size, Some(size))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = self.size.overflowing_add(n);
        if end > self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let nth = &self.v[n..end];
            self.v = &self.v[n + 1..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.size > self.v.len() {
            None
        } else {
            let start = self.v.len() - self.size;
            Some(&self.v[start..])
        }
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        // SAFETY: since the caller guarantees that `i` is in bounds,
        // which means that `i` cannot overflow an `isize`, and the
        // slice created by `from_raw_parts` is a subslice of `self.v`
        // thus is guaranteed to be valid for the lifetime `'a` of `self.v`.
        unsafe { from_raw_parts(self.v.as_ptr().add(idx), self.size) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Windows<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.size > self.v.len() {
            None
        } else {
            let ret = Some(&self.v[self.v.len() - self.size..]);
            self.v = &self.v[..self.v.len() - 1];
            ret
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = self.v.len().overflowing_sub(n);
        if end < self.size || overflow {
            self.v = &[];
            None
        } else {
            let ret = &self.v[end - self.size..end];
            self.v = &self.v[..end - 1];
            Some(ret)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Windows<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Windows<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Windows<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for Windows<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`chunks`] method on [slices].
///
/// [`chunks`]: ../../std/primitive.slice.html#method.chunks
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chunks<'a, T: 'a> {
    pub(super) v: &'a [T],
    pub(super) chunk_size: usize,
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Chunks<'_, T> {
    fn clone(&self) -> Self {
        Chunks { v: self.v, chunk_size: self.chunk_size }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Chunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let chunksz = cmp::min(self.v.len(), self.chunk_size);
            let (fst, snd) = self.v.split_at(chunksz);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let end = match start.checked_add(self.chunk_size) {
                Some(sum) => cmp::min(self.v.len(), sum),
                None => self.v.len(),
            };
            let nth = &self.v[start..end];
            self.v = &self.v[end..];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let start = (self.v.len() - 1) / self.chunk_size * self.chunk_size;
            Some(&self.v[start..])
        }
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        let start = idx * self.chunk_size;
        let end = match start.checked_add(self.chunk_size) {
            None => self.v.len(),
            Some(end) => cmp::min(end, self.v.len()),
        };
        // SAFETY: the caller guarantees that `i` is in bounds,
        // which means that `start` must be in bounds of the
        // underlying `self.v` slice, and we made sure that `end`
        // is also in bounds of `self.v`. Thus, `start` cannot overflow
        // an `isize`, and the slice constructed by `from_raw_parts`
        // is a subslice of `self.v` which is guaranteed to be valid
        // for the lifetime `'a` of `self.v`.
        unsafe { from_raw_parts(self.v.as_ptr().add(start), end - start) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Chunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let chunksz = if remainder != 0 { remainder } else { self.chunk_size };
            let (fst, snd) = self.v.split_at(self.v.len() - chunksz);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &[];
            None
        } else {
            let start = (len - 1 - n) * self.chunk_size;
            let end = match start.checked_add(self.chunk_size) {
                Some(res) => cmp::min(res, self.v.len()),
                None => self.v.len(),
            };
            let nth_back = &self.v[start..end];
            self.v = &self.v[..start];
            Some(nth_back)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Chunks<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Chunks<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Chunks<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for Chunks<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`chunks_mut`] method on [slices].
///
/// [`chunks_mut`]: ../../std/primitive.slice.html#method.chunks_mut
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ChunksMut<'a, T: 'a> {
    pub(super) v: &'a mut [T],
    pub(super) chunk_size: usize,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for ChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let sz = cmp::min(self.v.len(), self.chunk_size);
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(sz);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let end = match start.checked_add(self.chunk_size) {
                Some(sum) => cmp::min(self.v.len(), sum),
                None => self.v.len(),
            };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(end);
            let (_, nth) = head.split_at_mut(start);
            self.v = tail;
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let start = (self.v.len() - 1) / self.chunk_size * self.chunk_size;
            Some(&mut self.v[start..])
        }
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        let start = idx * self.chunk_size;
        let end = match start.checked_add(self.chunk_size) {
            None => self.v.len(),
            Some(end) => cmp::min(end, self.v.len()),
        };
        // SAFETY: see comments for `Chunks::get_unchecked`.
        //
        // Also note that the caller also guarantees that we're never called
        // with the same index again, and that no other methods that will
        // access this subslice are called, so it is valid for the returned
        // slice to be mutable.
        unsafe { from_raw_parts_mut(self.v.as_mut_ptr().add(start), end - start) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for ChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let sz = if remainder != 0 { remainder } else { self.chunk_size };
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - sz);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &mut [];
            None
        } else {
            let start = (len - 1 - n) * self.chunk_size;
            let end = match start.checked_add(self.chunk_size) {
                Some(res) => cmp::min(res, self.v.len()),
                None => self.v.len(),
            };
            let (temp, _tail) = mem::replace(&mut self.v, &mut []).split_at_mut(end);
            let (head, nth_back) = temp.split_at_mut(start);
            self.v = head;
            Some(nth_back)
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for ChunksMut<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksMut<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for ChunksMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for ChunksMut<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`chunks_exact`] method on [slices].
///
/// [`chunks_exact`]: ../../std/primitive.slice.html#method.chunks_exact
/// [`remainder`]: ChunksExact::remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub struct ChunksExact<'a, T: 'a> {
    pub(super) v: &'a [T],
    pub(super) rem: &'a [T],
    pub(super) chunk_size: usize,
}

impl<'a, T> ChunksExact<'a, T> {
    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    pub fn remainder(&self) -> &'a [T] {
        self.rem
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> Clone for ChunksExact<'_, T> {
    fn clone(&self) -> Self {
        ChunksExact { v: self.v, rem: self.rem, chunk_size: self.chunk_size }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> Iterator for ChunksExact<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.chunk_size);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let (_, snd) = self.v.split_at(start);
            self.v = snd;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        let start = idx * self.chunk_size;
        // SAFETY: mostly identical to `Chunks::get_unchecked`.
        unsafe { from_raw_parts(self.v.as_ptr().add(start), self.chunk_size) }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for ChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.v.len() - self.chunk_size);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &[];
            None
        } else {
            let start = (len - 1 - n) * self.chunk_size;
            let end = start + self.chunk_size;
            let nth_back = &self.v[start..end];
            self.v = &self.v[..start];
            Some(nth_back)
        }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> ExactSizeIterator for ChunksExact<'_, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksExact<'_, T> {}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> FusedIterator for ChunksExact<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for ChunksExact<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last up to
/// `chunk_size-1` elements will be omitted but can be retrieved from the
/// [`into_remainder`] function from the iterator.
///
/// This struct is created by the [`chunks_exact_mut`] method on [slices].
///
/// [`chunks_exact_mut`]: ../../std/primitive.slice.html#method.chunks_exact_mut
/// [`into_remainder`]: ChunksExactMut::into_remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "chunks_exact", since = "1.31.0")]
pub struct ChunksExactMut<'a, T: 'a> {
    pub(super) v: &'a mut [T],
    pub(super) rem: &'a mut [T],
    pub(super) chunk_size: usize,
}

impl<'a, T> ChunksExactMut<'a, T> {
    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[stable(feature = "chunks_exact", since = "1.31.0")]
    pub fn into_remainder(self) -> &'a mut [T] {
        self.rem
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> Iterator for ChunksExactMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(self.chunk_size);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (start, overflow) = n.overflowing_mul(self.chunk_size);
        if start >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (_, snd) = tmp.split_at_mut(start);
            self.v = snd;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        let start = idx * self.chunk_size;
        // SAFETY: see comments for `ChunksMut::get_unchecked`.
        unsafe { from_raw_parts_mut(self.v.as_mut_ptr().add(start), self.chunk_size) }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for ChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - self.chunk_size);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &mut [];
            None
        } else {
            let start = (len - 1 - n) * self.chunk_size;
            let end = start + self.chunk_size;
            let (temp, _tail) = mem::replace(&mut self.v, &mut []).split_at_mut(end);
            let (head, nth_back) = temp.split_at_mut(start);
            self.v = head;
            Some(nth_back)
        }
    }
}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> ExactSizeIterator for ChunksExactMut<'_, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for ChunksExactMut<'_, T> {}

#[stable(feature = "chunks_exact", since = "1.31.0")]
impl<T> FusedIterator for ChunksExactMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for ChunksExactMut<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) chunks (`N` elements at a
/// time), starting at the beginning of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`array_chunks`] method on [slices].
///
/// [`array_chunks`]: ../../std/primitive.slice.html#method.array_chunks
/// [`remainder`]: ArrayChunks::remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[unstable(feature = "array_chunks", issue = "74985")]
pub struct ArrayChunks<'a, T: 'a, const N: usize> {
    pub(super) iter: Iter<'a, [T; N]>,
    pub(super) rem: &'a [T],
}

impl<'a, T, const N: usize> ArrayChunks<'a, T, N> {
    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[unstable(feature = "array_chunks", issue = "74985")]
    pub fn remainder(&self) -> &'a [T] {
        self.rem
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[unstable(feature = "array_chunks", issue = "74985")]
impl<T, const N: usize> Clone for ArrayChunks<'_, T, N> {
    fn clone(&self) -> Self {
        ArrayChunks { iter: self.iter.clone(), rem: self.rem }
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<'a, T, const N: usize> Iterator for ArrayChunks<'a, T, N> {
    type Item = &'a [T; N];

    #[inline]
    fn next(&mut self) -> Option<&'a [T; N]> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n)
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.iter.last()
    }

    unsafe fn get_unchecked(&mut self, i: usize) -> &'a [T; N] {
        // SAFETY: The safety guarantees of `get_unchecked` are transferred to
        // the caller.
        unsafe { self.iter.get_unchecked(i) }
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<'a, T, const N: usize> DoubleEndedIterator for ArrayChunks<'a, T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T; N]> {
        self.iter.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth_back(n)
    }
}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<T, const N: usize> ExactSizeIterator for ArrayChunks<'_, T, N> {
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T, const N: usize> TrustedLen for ArrayChunks<'_, T, N> {}

#[unstable(feature = "array_chunks", issue = "74985")]
impl<T, const N: usize> FusedIterator for ArrayChunks<'_, T, N> {}

#[doc(hidden)]
#[unstable(feature = "array_chunks", issue = "74985")]
unsafe impl<'a, T, const N: usize> TrustedRandomAccess for ArrayChunks<'a, T, N> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`rchunks`] method on [slices].
///
/// [`rchunks`]: ../../std/primitive.slice.html#method.rchunks
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunks<'a, T: 'a> {
    pub(super) v: &'a [T],
    pub(super) chunk_size: usize,
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> Clone for RChunks<'_, T> {
    fn clone(&self) -> Self {
        RChunks { v: self.v, chunk_size: self.chunk_size }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunks<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let chunksz = cmp::min(self.v.len(), self.chunk_size);
            let (fst, snd) = self.v.split_at(self.v.len() - chunksz);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            // Can't underflow because of the check above
            let end = self.v.len() - end;
            let start = match end.checked_sub(self.chunk_size) {
                Some(sum) => sum,
                None => 0,
            };
            let nth = &self.v[start..end];
            self.v = &self.v[0..start];
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let rem = self.v.len() % self.chunk_size;
            let end = if rem == 0 { self.chunk_size } else { rem };
            Some(&self.v[0..end])
        }
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        let end = self.v.len() - idx * self.chunk_size;
        let start = match end.checked_sub(self.chunk_size) {
            None => 0,
            Some(start) => start,
        };
        // SAFETY: mostly identical to `Chunks::get_unchecked`.
        unsafe { from_raw_parts(self.v.as_ptr().add(start), end - start) }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunks<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let chunksz = if remainder != 0 { remainder } else { self.chunk_size };
            let (fst, snd) = self.v.split_at(chunksz);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &[];
            None
        } else {
            // can't underflow because `n < len`
            let offset_from_end = (len - 1 - n) * self.chunk_size;
            let end = self.v.len() - offset_from_end;
            let start = end.saturating_sub(self.chunk_size);
            let nth_back = &self.v[start..end];
            self.v = &self.v[end..];
            Some(nth_back)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> ExactSizeIterator for RChunks<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for RChunks<'_, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> FusedIterator for RChunks<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for RChunks<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last slice
/// of the iteration will be the remainder.
///
/// This struct is created by the [`rchunks_mut`] method on [slices].
///
/// [`rchunks_mut`]: ../../std/primitive.slice.html#method.rchunks_mut
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksMut<'a, T: 'a> {
    pub(super) v: &'a mut [T],
    pub(super) chunk_size: usize,
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let sz = cmp::min(self.v.len(), self.chunk_size);
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - sz);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.v.is_empty() {
            (0, Some(0))
        } else {
            let n = self.v.len() / self.chunk_size;
            let rem = self.v.len() % self.chunk_size;
            let n = if rem > 0 { n + 1 } else { n };
            (n, Some(n))
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            // Can't underflow because of the check above
            let end = self.v.len() - end;
            let start = match end.checked_sub(self.chunk_size) {
                Some(sum) => sum,
                None => 0,
            };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(start);
            let (nth, _) = tail.split_at_mut(end - start);
            self.v = head;
            Some(nth)
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.v.is_empty() {
            None
        } else {
            let rem = self.v.len() % self.chunk_size;
            let end = if rem == 0 { self.chunk_size } else { rem };
            Some(&mut self.v[0..end])
        }
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        let end = self.v.len() - idx * self.chunk_size;
        let start = match end.checked_sub(self.chunk_size) {
            None => 0,
            Some(start) => start,
        };
        // SAFETY: see comments for `RChunks::get_unchecked` and `ChunksMut::get_unchecked`
        unsafe { from_raw_parts_mut(self.v.as_mut_ptr().add(start), end - start) }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.is_empty() {
            None
        } else {
            let remainder = self.v.len() % self.chunk_size;
            let sz = if remainder != 0 { remainder } else { self.chunk_size };
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(sz);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &mut [];
            None
        } else {
            // can't underflow because `n < len`
            let offset_from_end = (len - 1 - n) * self.chunk_size;
            let end = self.v.len() - offset_from_end;
            let start = end.saturating_sub(self.chunk_size);
            let (tmp, tail) = mem::replace(&mut self.v, &mut []).split_at_mut(end);
            let (_, nth_back) = tmp.split_at_mut(start);
            self.v = tail;
            Some(nth_back)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> ExactSizeIterator for RChunksMut<'_, T> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for RChunksMut<'_, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> FusedIterator for RChunksMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksMut<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) chunks (`chunk_size` elements at a
/// time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from
/// the [`remainder`] function from the iterator.
///
/// This struct is created by the [`rchunks_exact`] method on [slices].
///
/// [`rchunks_exact`]: ../../std/primitive.slice.html#method.rchunks_exact
/// [`remainder`]: ChunksExact::remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksExact<'a, T: 'a> {
    pub(super) v: &'a [T],
    pub(super) rem: &'a [T],
    pub(super) chunk_size: usize,
}

impl<'a, T> RChunksExact<'a, T> {
    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[stable(feature = "rchunks", since = "1.31.0")]
    pub fn remainder(&self) -> &'a [T] {
        self.rem
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Clone for RChunksExact<'a, T> {
    fn clone(&self) -> RChunksExact<'a, T> {
        RChunksExact { v: self.v, rem: self.rem, chunk_size: self.chunk_size }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksExact<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.v.len() - self.chunk_size);
            self.v = fst;
            Some(snd)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &[];
            None
        } else {
            let (fst, _) = self.v.split_at(self.v.len() - end);
            self.v = fst;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        let end = self.v.len() - idx * self.chunk_size;
        let start = end - self.chunk_size;
        // SAFETY:
        // SAFETY: mostmy identical to `Chunks::get_unchecked`.
        unsafe { from_raw_parts(self.v.as_ptr().add(start), self.chunk_size) }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let (fst, snd) = self.v.split_at(self.chunk_size);
            self.v = snd;
            Some(fst)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &[];
            None
        } else {
            // now that we know that `n` corresponds to a chunk,
            // none of these operations can underflow/overflow
            let offset = (len - n) * self.chunk_size;
            let start = self.v.len() - offset;
            let end = start + self.chunk_size;
            let nth_back = &self.v[start..end];
            self.v = &self.v[end..];
            Some(nth_back)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> ExactSizeIterator for RChunksExact<'a, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for RChunksExact<'_, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> FusedIterator for RChunksExact<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksExact<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

/// An iterator over a slice in (non-overlapping) mutable chunks (`chunk_size`
/// elements at a time), starting at the end of the slice.
///
/// When the slice len is not evenly divided by the chunk size, the last up to
/// `chunk_size-1` elements will be omitted but can be retrieved from the
/// [`into_remainder`] function from the iterator.
///
/// This struct is created by the [`rchunks_exact_mut`] method on [slices].
///
/// [`rchunks_exact_mut`]: ../../std/primitive.slice.html#method.rchunks_exact_mut
/// [`into_remainder`]: ChunksExactMut::into_remainder
/// [slices]: ../../std/primitive.slice.html
#[derive(Debug)]
#[stable(feature = "rchunks", since = "1.31.0")]
pub struct RChunksExactMut<'a, T: 'a> {
    pub(super) v: &'a mut [T],
    pub(super) rem: &'a mut [T],
    pub(super) chunk_size: usize,
}

impl<'a, T> RChunksExactMut<'a, T> {
    /// Returns the remainder of the original slice that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[stable(feature = "rchunks", since = "1.31.0")]
    pub fn into_remainder(self) -> &'a mut [T] {
        self.rem
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> Iterator for RChunksExactMut<'a, T> {
    type Item = &'a mut [T];

    #[inline]
    fn next(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (head, tail) = tmp.split_at_mut(tmp_len - self.chunk_size);
            self.v = head;
            Some(tail)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.v.len() / self.chunk_size;
        (n, Some(n))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<&'a mut [T]> {
        let (end, overflow) = n.overflowing_mul(self.chunk_size);
        if end >= self.v.len() || overflow {
            self.v = &mut [];
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let tmp_len = tmp.len();
            let (fst, _) = tmp.split_at_mut(tmp_len - end);
            self.v = fst;
            self.next()
        }
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[doc(hidden)]
    unsafe fn get_unchecked(&mut self, idx: usize) -> Self::Item {
        let end = self.v.len() - idx * self.chunk_size;
        let start = end - self.chunk_size;
        // SAFETY: see comments for `RChunksMut::get_unchecked`.
        unsafe { from_raw_parts_mut(self.v.as_mut_ptr().add(start), self.chunk_size) }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<'a, T> DoubleEndedIterator for RChunksExactMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut [T]> {
        if self.v.len() < self.chunk_size {
            None
        } else {
            let tmp = mem::replace(&mut self.v, &mut []);
            let (head, tail) = tmp.split_at_mut(self.chunk_size);
            self.v = tail;
            Some(head)
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.len();
        if n >= len {
            self.v = &mut [];
            None
        } else {
            // now that we know that `n` corresponds to a chunk,
            // none of these operations can underflow/overflow
            let offset = (len - n) * self.chunk_size;
            let start = self.v.len() - offset;
            let end = start + self.chunk_size;
            let (tmp, tail) = mem::replace(&mut self.v, &mut []).split_at_mut(end);
            let (_, nth_back) = tmp.split_at_mut(start);
            self.v = tail;
            Some(nth_back)
        }
    }
}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> ExactSizeIterator for RChunksExactMut<'_, T> {
    fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for RChunksExactMut<'_, T> {}

#[stable(feature = "rchunks", since = "1.31.0")]
impl<T> FusedIterator for RChunksExactMut<'_, T> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for RChunksExactMut<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for Iter<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl<'a, T> TrustedRandomAccess for IterMut<'a, T> {
    fn may_have_side_effect() -> bool {
        false
    }
}
