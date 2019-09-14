/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#pragma once

#include "Compiler.hpp"
#include "GTest.hpp"

namespace util {

// Utilities for creating parameters for typed tests on GoogleTest
//
// Types is used to define type list, it's just an alias to ::testing::Types:
// using Types = util::Types<int,char,float>;
//
// You can declare your tests using a template fixture:
//
// template <class T>
// class TestFixture : ::testing::Test
// {
// };
//
// Inside your test function, you can access the parameters like this:
//
// TEST(TestFixture, mytest)
// {
//      using Type0 = GetType<TypeParam,0>; // the first type element
// }
//
// You can compose complicated type arguments using Concat and CrossJoin
// and RemoveIf
//
// using Types = CrossJoin<Types<int,float>,Types<char, void*>>;
// creates the parameters <int,char> <int,void*> <float,char> <float,void*>
//
// Concat is useful to concatenate parameter lists created with CrossJoin
//
// using Types = Concat<CrossJoin<Types<int,float>,Types<char>>,
//                      CrossJoin<Types<char,double>,ValueType<void*,short>>>;
// creates the parameters <int,char> <float,char> <char,void*> <char,short>
// <double,void*> <double,short>
//
// RemoveIf can be used to remove some parameters that match a given predicate:
//
// using Types = RemoveIf<AllSame, CrossJoin<Types<int,char>, Types<int,char>>>;
// creates the parameters <int,char>,<char,int>

// Types -----------------------------------------
using ::testing::Types;

template <class T, int D>
struct GetTypeImpl {
  static_assert(D == 0, "Out of bounds");
  using type = T;
};

template <class... T, int D>
struct GetTypeImpl<Types<T...>, D> {
  static_assert(D < sizeof...(T), "Out of bounds");

  using type = typename GetTypeImpl<typename Types<T...>::Tail, D - 1>::type;
};

template <class... ARGS>
struct GetTypeImpl<Types<ARGS...>, 0> {
  static_assert(sizeof...(ARGS) > 0, "Out of bounds");

  using type = typename Types<ARGS...>::Head;
};

/**---------------------------------------------------------------------------*
 * @brief Gives the specified type from a type list
 *
 * Example:
 * ```
 * using T = GetType< Types<int, float, char, void*>, 2>
 * // T == char
 * ```
 *
 * @tparam TUPLE The type list
 * @tparam D Index of the desired type
 *---------------------------------------------------------------------------**/
template <class TUPLE, int D>
using GetType = typename GetTypeImpl<TUPLE, D>::type;

// GetSize -------------------------------
template <class TUPLE>
struct GetSizeImpl;

template <class... TYPES>
struct GetSizeImpl<Types<TYPES...>> {
  static constexpr auto value = sizeof...(TYPES);
};

/**---------------------------------------------------------------------------*
 * @brief Returns the size (number of elements) in a type list
 *
 * Example:
 * ```
 * GetSize< Types<int, float, double, void*> == 4
 * ```
 *---------------------------------------------------------------------------**/
template <class TUPLE>
constexpr auto GetSize = GetSizeImpl<TUPLE>::value;

// Concat -----------------------------------------
namespace detail {
template <class A, class B>
struct Concat2;

template <class... T, class... U>
struct Concat2<Types<T...>, Types<U...>> {
  using type = Types<T..., U...>;
};
}  // namespace detail

template <class... T>
struct ConcatImpl;

template <class HEAD1, class HEAD2, class... TAIL>
struct ConcatImpl<HEAD1, HEAD2, TAIL...> {
  using type = typename ConcatImpl<typename detail::Concat2<HEAD1, HEAD2>::type,
                                   TAIL...>::type;
};

template <class A>
struct ConcatImpl<A> {
  using type = A;
};

template <class... A>
struct ConcatImpl<Types<A...>> {
  using type = Types<A...>;
};

template <>
struct ConcatImpl<> {
  using type = Types<>;
};

/**---------------------------------------------------------------------------*
 * @brief Concantenates compile-time lists of types into a single type list.
 *
 * Example:
 * ```
 * using MyTypes = Concat< Types<int, float>, Types<char, double>>
 * // MyTypes == Types<int, float, char, double>;
 * ```
 *---------------------------------------------------------------------------**/
template <class... T>
using Concat = typename ConcatImpl<T...>::type;

// Flatten -----------------------------------------
template <class T>
struct FlattenImpl;

template <>
struct FlattenImpl<Types<>> {
  using type = Types<>;
};

template <class HEAD, class... TAIL>
struct FlattenImpl<Types<HEAD, TAIL...>> {
  using type = Concat<Types<HEAD>, typename FlattenImpl<Types<TAIL...>>::type>;
};

template <class... HEAD, class... TAIL>
struct FlattenImpl<Types<Types<HEAD...>, TAIL...>> {
  using type = typename FlattenImpl<Types<HEAD..., TAIL...>>::type;
};

/**---------------------------------------------------------------------------*
 * @brief Flattens nested compile-time lists of types into a single list of
 *types.
 *
 * Example:
 * ```
 * // Flatten< Types< int, Types< double, Types<char> > > == Types<int, double,
 *char> static_assert(std::is_same<Flatten<Types<Types<int, Types<double>>,
 *float>>, Types<int, double, float>>::value, "");
 * ```
 *---------------------------------------------------------------------------**/
template <class T>
using Flatten = typename FlattenImpl<T>::type;

// CrossJoin -----------------------------------------
namespace detail {
// prepend T in TUPLE
template <class T, class TUPLE>
struct Prepend1;

template <class T, class... ARGS>
struct Prepend1<T, Types<ARGS...>> {
  using type = Flatten<Types<T, ARGS...>>;
};

template <class T, class TUPLES>
struct Prepend;

// Prepend T in all TUPLES
template <class T, class... TUPLES>
struct Prepend<T, Types<TUPLES...>> {
  using type = Types<typename Prepend1<T, TUPLES>::type...>;
};

// skip empty tuples
template <class T, class... TUPLES>
struct Prepend<T, Types<Types<>, TUPLES...>> : Prepend<T, Types<TUPLES...>> {};
}  // namespace detail

template <class... ARGS>
struct CrossJoinImpl;

template <>
struct CrossJoinImpl<> {
  using type = Types<>;
};

template <class... ARGS>
struct CrossJoinImpl<Types<ARGS...>> {
  using type = Types<Types<ARGS>...>;
};

template <class... AARGS, class... TAIL>
struct CrossJoinImpl<Types<AARGS...>, TAIL...> {
  using type = Concat<typename detail::Prepend<
      AARGS, typename CrossJoinImpl<TAIL...>::type>::type...>;
};

// to make it easy for the user when there's only one element to be joined
template <class T, class... TAIL>
struct CrossJoinImpl<T, TAIL...> : CrossJoinImpl<Types<T>, TAIL...> {};

/**---------------------------------------------------------------------------*
 * @brief Creates a new type list from the cross product (cartesian product) of
 * two type lists.
 *
 * Example:
 * ```
 * using Types = CrossJoin<Types<int,float>, Types<char, double>>;
 * // Types == Types< Types<int, char>, Types<int, double>, Types<float, char>,
 * Types<float, double> >
 * ```
 *---------------------------------------------------------------------------**/
template <class... ARGS>
using CrossJoin = typename CrossJoinImpl<ARGS...>::type;

// AllSame -----------------------------------------
namespace detail {
template <class... ITEMS>
struct AllSame : std::false_type {};

// degenerate case
template <class A>
struct AllSame<A> : std::true_type {};

template <class A>
struct AllSame<A, A> : std::true_type {};

template <class HEAD, class... TAIL>
struct AllSame<HEAD, HEAD, TAIL...> : AllSame<HEAD, TAIL...> {};

template <class... ITEMS>
struct AllSame<Types<ITEMS...>> : AllSame<ITEMS...> {};

}  // namespace detail

/**---------------------------------------------------------------------------*
 * @brief Indicates if all types in a list are identical.
 *
 * This is useful as a predicate for for `RemoveIf`.
 *
 * Example:
 * ```
 * // AllSame::Call<Types<int, int, int>> == true_type
 * // AllSame::Call<Types<float, bool>> == false_type
 *
 * // Used as a predicate
 * RemoveIf<AllSame, Types<Types<int, int, int>>> ==  Types<>
 * RemoveIf<AllSame, Types<Types<int, float, int>>> ==  Types<Types<int, float,
 *int>>
 * ```
 *---------------------------------------------------------------------------**/
struct AllSame {
  template <class... ITEMS>
  using Call = detail::AllSame<ITEMS...>;
};

// Exists ---------------------------------
// Do a linear search to find NEEDLE in HAYSACK
template <class NEEDLE, class HAYSACK>
struct ExistsImpl;

// end case, no more types to check
template <class NEEDLE>
struct ExistsImpl<NEEDLE, Types<>> : std::false_type {};

// next one matches
template <class NEEDLE, class... TAIL>
struct ExistsImpl<NEEDLE, Types<NEEDLE, TAIL...>> : std::true_type {};

// next one doesn't match
template <class NEEDLE, class HEAD, class... TAIL>
struct ExistsImpl<NEEDLE, Types<HEAD, TAIL...>>
    : ExistsImpl<NEEDLE, Types<TAIL...>> {};
/**---------------------------------------------------------------------------*
 * @brief Indicates if a type exists within a type list.
 *
 * Example:
 * ```
 * // Exists<int, Types<float, double, int>> == true_type
 * // Exists<char, Types<int, float, void*>> == false_type
 * ```
 * @tparam NEEDLE The type to search for
 * @tparam HAYSACK The list to search in
 *---------------------------------------------------------------------------**/
template <class NEEDLE, class HAYSACK>
constexpr bool Exists = ExistsImpl<NEEDLE, HAYSACK>::value;

// ContainedIn -----------------------------------------
/**---------------------------------------------------------------------------*
 * @brief Indicates if a type exists within a type list.
 *
 * Used as a predicate for RemoveIf
 *
 * Example:
 * ```
 * ContainedIn<Types<Types<int, char>>>::Call<Types<int, char>>::value == true_type
 * ContainedIn<Types<Types<int, char>>>::Call<Types<int, float>>::value == false_type
 *
 * // Used as a predicate
 * using MyTypes = RemoveIf<ContainedIn<Types<Types<char, char>>,
 *                                      Types<Types<char, char>, Types<float,int>>> 
 * // MyTypes == Types<float, int>
 *
 * ```
 *
 * @tparam HAYSACK The type list to search
 *---------------------------------------------------------------------------**/
template <class HAYSACK>
struct ContainedIn {
  /**---------------------------------------------------------------------------*
   * @brief Invoked as predicate for RemoveIf
   *
   * @tparam NEEDLE The type to search for
   *---------------------------------------------------------------------------**/
  template <class NEEDLE>
  using Call = ExistsImpl<NEEDLE, HAYSACK>;
};

// RemoveIf -----------------------------------------
template<class PRED, class TUPLE>
struct RemoveIfImpl;

template<class PRED>
struct RemoveIfImpl<PRED, Types<>>
{
    using type = Types<>;
};

template<class PRED, class HEAD, class... TAIL>
struct RemoveIfImpl<PRED, Types<HEAD, TAIL...>>
{
    using type = Concat<typename std::conditional<PRED::template
Call<HEAD>::value, Types<>, Types<HEAD>>::type, typename RemoveIfImpl<PRED,
Types<TAIL...>>::type>;
};

/**---------------------------------------------------------------------------*
 * @brief Removes types from a type list that satisfy a predicate
 * 
 * Possible predicates: `AllSame`, `ContainedIn`
 * 
 * Example:
 * ```
 * RemoveIf<AllSame, Types<Types<int, int, int>>> ==  Types<>
 * RemoveIf<AllSame, Types<Types<int, float, int>>> ==  Types<Types<int, float, int>>
 * 
 * using MyTypes = RemoveIf<ContainedIn<Types<Types<char, char>>,
 *                                      Types<Types<char, char>, Types<float,int>>> 
 * // MyTypes == Types<float, int>
 * ```
 *
 * @tparam PRED The predicate
 * @tparam TUPLE The list of types on which to apply the predicate
 *---------------------------------------------------------------------------**/
template<class PRED, class TUPLE>
using RemoveIf = typename RemoveIfImpl<PRED, TUPLE>::type;


// Transform --------------------------------

template<class XFORM, class TYPES>
struct TransformImpl;

template<class XFORM, class... ITEMS>
struct TransformImpl<XFORM, Types<ITEMS...>>
{
    using type = Types<typename XFORM::template Call<ITEMS>...>;
};

/**---------------------------------------------------------------------------*
 * @brief Applies a transformation to every type in a type list
 * 
 * Possible transformations: Repeat
 * 
 * Example:
 * ```
 * // Repeat transformation repeats each type for a specified count
 * using MyTypes = Transform<Repeat<2>, Types<int, float>>;
 * // MyTypes == Types< Types<int, int>, Types<float, float>>);
 * ```
 * 
 * @tparam XFORM The transformation to apply
 * @tparam TYPES The list of types to transform
*---------------------------------------------------------------------------**/
template<class XFORM, class TYPES>
using Transform = typename TransformImpl<XFORM, TYPES>::type;


// Repeat --------------------------------

namespace detail {
template<class T, int N, class RES>
struct Repeat;

template<class T, int N, class... ITEMS>
struct Repeat<T, N, Types<ITEMS...>>
{
    using type = typename Repeat<T, N - 1, Types<T, ITEMS...>>::type;
};

template<class T, class... ITEMS>
struct Repeat<T, 0, Types<ITEMS...>>
{
    using type = Types<ITEMS...>;
};
} // namespace detail

/**---------------------------------------------------------------------------*
 * @brief Transformation that repeats a type for a specified count.
 * 
 * Used in Transform.
 * 
 * Example:
 * ```
 * // Repeat transformation repeats each type for a specified count
 * using MyTypes = Transform<Repeat<2>, Types<int, float>>;
 * // MyTypes == Types< Types<int, int>, Types<float, float>>);
 * ```
 * 
 * @tparam N The number of times to repeat the type
*---------------------------------------------------------------------------**/
template<int N>
struct Repeat
{
    template<class T>
    using Call = typename detail::Repeat<T, N, Types<>>::type;
};


// Append --------------------------------

template<class TYPES, class... ITEMS>
struct AppendImpl;

template<class... HEAD, class... TAIL>
struct AppendImpl<Types<HEAD...>, TAIL...>
{
    using type = Types<HEAD..., TAIL...>;
};

/**---------------------------------------------------------------------------*
 * @brief Appends types to a type list
 *
 * Example:
 * ```
 * using MyTypes = Append<Types<int>, float, char>;
 * MyTypes ==  Types<int, float, char>;
 * ```
 *
 * @tparam TYPES The type list to append to
 * @tparam ITEMS The types to append
 *---------------------------------------------------------------------------**/
template<class TYPES, class... ITEMS>
using Append = typename AppendImpl<TYPES, ITEMS...>::type;

// Remove -------------------------------------------
// remove items from tuple given by their indices

namespace detail {
template<class TUPLE, int CUR, int... IDXs>
struct Remove;

// nothing else to do?
template<class... ITEMS, int CUR>
struct Remove<Types<ITEMS...>, CUR>
{
    using type = Types<ITEMS...>;
};

// index match current item?
template<class HEAD, class... TAIL, int CUR, int... IDXTAIL>
struct Remove<Types<HEAD, TAIL...>, CUR, CUR, IDXTAIL...>
{
    // remove it, and recurse into the remaining items
    using type = typename Remove<Types<TAIL...>, CUR + 1, IDXTAIL...>::type;
};

// index doesn't match current item?
template<class HEAD, class... TAIL, int CUR, int IDXHEAD, int... IDXTAIL>
struct Remove<Types<HEAD, TAIL...>, CUR, IDXHEAD, IDXTAIL...>
{
    static_assert(sizeof...(TAIL) + 1 > IDXHEAD - CUR, "Index out of bounds");

    // add current item to output and recurse into the remaining items
    using type = Concat<Types<HEAD>, typename Remove<Types<TAIL...>, CUR + 1,
IDXHEAD, IDXTAIL...>::type>;
};
} // namespace detail

template<class TUPLE, int... IDXs>
struct RemoveImpl
{
    using type = typename detail::Remove<TUPLE, 0, IDXs...>::type;
};

/**---------------------------------------------------------------------------*
 * @brief Removes types at specified indices from a type list.
 * 
 * @tparam TUPLE Type list to remove types from
 * @tparam IDXs Indices of types to remove
*---------------------------------------------------------------------------**/
template<class TUPLE, int... IDXs>
using Remove = typename RemoveImpl<TUPLE, IDXs...>::type;

/*

// Unique --------------------------------

namespace detail {
template<class... ITEMS>
struct Unique;

template<>
struct Unique<>
{
    using type = Types<>;
};

template<class HEAD, class... TAIL>
struct Unique<HEAD, TAIL...>
{
    using type =
        Concat<std::conditional_t<Exists<HEAD, Types<TAIL...>>, Types<>,
Types<HEAD>>, typename Unique<TAIL...>::type>;
};
} // namespace detail

template<class TYPES>
struct UniqueImpl;

template<class... ITEMS>
struct UniqueImpl<Types<ITEMS...>>
{
    using type = typename detail::Unique<ITEMS...>::type;
};

template<class TYPES>
using Unique = typename UniqueImpl<TYPES>::type;

// Helper to be able to define typed test cases by defining the
// types inline (no worries about commas in macros)

#define VPI_TYPED_TEST_SUITE_F(TEST, ...) \
    using TEST##_Types = __VA_ARGS__;     \
    TYPED_TEST_SUITE(TEST, TEST##_Types)

#define VPI_TYPED_TEST_SUITE(TEST, ...) \
    template<class T>                   \
    class TEST : public ::testing::Test \
    {                                   \
    };                                  \
    VPI_TYPED_TEST_SUITE_F(TEST, __VA_ARGS__)

#define VPI_INSTANTIATE_TYPED_TEST_SUITE_P(INSTNAME, TEST, ...) \
    using TEST##INSTNAME##_Types = __VA_ARGS__;                 \
    INSTANTIATE_TYPED_TEST_SUITE_P(INSTNAME, TEST, TEST##INSTNAME##_Types)

// Contains ------------------------------
// check if value is in util::Values container

template<class T>
constexpr bool Contains(Types<>, T size)
{
    return false;
}

template<class T, auto HEAD, auto... TAIL>
constexpr bool Contains(Types<Value(HEAD), Value(TAIL)...>, T needle)
{
    if (HEAD == needle)
    {
        return true;
    }
    else
    {
        return Contains(Types<Value(TAIL)...>(), needle);
    }
}
*/

}  // namespace util

