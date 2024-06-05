"""Utility functions for internal data (representation) operations.

date: 2024-06-05
"""

__author__ = "Jeroen Van Der Donckt, Jonas Van Der Donckt"

import itertools
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd


def series_dict_to_df(series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Convert the `series_dict` into a pandas DataFrame with an outer merge.

    Parameters
    ----------
    series_dict : Dict[str, pd.Series]
        The dict with the Series.

    Returns
    -------
    pd.DataFrame
        The merged pandas DataFrame

    Note
    ----
    The `series_dict` is an internal representation of the time-series data.
    In this dictionary, the key is always the accompanying series its name.
    This internal representation is constructed in the `process` method of the
    `SeriesPipeline`.

    Examples
    --------
    # 創建一些 pandas Series
    >>> series_a = pd.Series([1, 2, 3], index=pd.date_range("2023-01-01", periods=3), name="A")
    >>> series_b = pd.Series([4, 5, 6], index=pd.date_range("2023-01-01", periods=3, freq="MS"), name="B")
    >>> series_c = pd.Series([7, 8, 9], index=pd.date_range("2023-01-02", periods=3), name="C")

    # 創建符合 Dict[str, pd.Series] 格式的字典
    >>> series_dict = {
        "A": series_a,
        "B": series_b,
        "C": series_c
        }
    
    # 使用 series_dict_to_df 函數將字典轉換為 DataFrame
    >>> df = series_dict_to_df(series_dict)

    >>> print(df)
                A    B    C
    2023-01-01  1.0  4.0  NaN
    2023-01-02  2.0  NaN  7.0
    2023-01-03  3.0  NaN  8.0
    2023-01-04  NaN  NaN  9.0
    2023-02-01  NaN  5.0  NaN
    2023-03-01  NaN  6.0  NaN


    """
    # 0. Check if the series_dict has only 1 series, to create the df efficiently
    if len(series_dict) == 1:
        return pd.DataFrame(series_dict)
    # 1. Check if the time-indexes of the series are equal, to create the df efficiently (the quick way)
    try:
        index_info = set(
            [
                (s.index[0], s.index[-1], len(s), s.index.freq)
                for s in series_dict.values()
            ]
        )
        if len(index_info) == 1:
            # If list(index_info)[0][-1] is None => this code assumes equal index to
            # perform efficient merge, otherwise the join will be still correct, but it
            # would actually be more efficient to perform the code at (2.).
            # But this disadvantage (slower merge) does not outweigh the time-loss when
            # checking the full index.
            # e.g.
            # test_series_a = pd.Series([1, 2, 3], 
            #                           index=[
            #                               pd.Timestamp('2023-01-01'),
            #                               pd.Timestamp('2023-01-03'),
            #                               pd.Timestamp('2023-01-05')
            #                           ],
            #                           name="A")
            # test_series_b = pd.Series([1, 2, 3], 
            #                           index=[
            #                               pd.Timestamp('2023-01-01'),
            #                               pd.Timestamp('2023-01-04'),
            #                               pd.Timestamp('2023-01-05')
            #                           ],
            #                           name="B")
            # index_info = 
            #       {(Timestamp('2023-01-01 00:00:00'), Timestamp('2023-01-05 00:00:00'), 3, None}

            # When the time-indexes are the same we can create df very efficiently
            return pd.DataFrame(series_dict, copy=False)
    except IndexError:
        # We catch an indexError as we make the assumption that there is data within the
        # series -> we do not make that assumption when constructing the DataFrame the
        # slow way.
        pass
    # 2. If check failed, create the df by merging the series (the slow way)
    df = pd.DataFrame()
    for key, s in series_dict.items():
        # Check if we deal with a valid series_dict before merging on series.name
        assert key == s.name
        df = df.merge(s, left_index=True, right_index=True, how="outer", copy=False)
    return df


def to_list(x: Any) -> List:
    """Convert the input to a list if necessary.

    Parameters
    ----------
    x : Any
        The input that needs to be converted into a list.

    Returns
    -------
    List
        A list of `x` if `x` wasn't a list yet, otherwise `x`.

    """
    if not isinstance(x, (list, np.ndarray)):
        return [x]
    return list(x)


def to_tuple(x: Any) -> Tuple[Any, ...]:
    """Convert the input to a tuple if necessary.

    Parameters
    ----------
    x : Any
        The input that needs to be converted into a tuple.

    Returns
    -------
    List
        A tuple of `x` if `x` wasn't a tuple yet, otherwise `x`.

    """
    if not isinstance(x, tuple):
        return (x,)
    return x


def flatten(data: Iterable) -> Iterator:
    """Flatten the given input data to an iterator.

    Parameters
    ----------
    data : Iterable
        The iterable data that needs to be flattened.

    Returns
    -------
    Iterator
        An iterator for the flattened data.

    """
    return itertools.chain.from_iterable(data)
