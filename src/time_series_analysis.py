import xarray as xr
import xarray

def remove_seasonal_cycle(ds):
    # 1) compute the monthly climatology *preserving* each var’s attrs
    clim = ds.groupby("time.month").mean("time", keep_attrs=True)

    # 2) subtract, then restore your dataset‐level attrs
    anom = ds.groupby("time.month") - clim
    anom.attrs = ds.attrs
    return anom

def detrend_da(da: xr.DataArray) -> xr.DataArray:
    """
    Remove the best‐fit linear trend from a time series DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input data, must have a 'time' dimension of dtype datetime64.

    Returns
    -------
    xr.DataArray
        Detrended DataArray, same shape and coords as `da`.
    """
    # 1) Fit a degree‐1 polynomial along 'time'
    fit = da.polyfit(dim="time", deg=1)

    # 2) Extract the fitted coefficients (degree=0 is intercept, degree=1 is slope)
    coeffs = fit["polyfit_coefficients"]

    # 3) Reconstruct the trend at each time point
    trend = xr.polyval(da["time"], coeffs)

    # 4) Subtract the trend
    return da - trend