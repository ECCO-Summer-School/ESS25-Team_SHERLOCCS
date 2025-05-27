import xarray

def remove_seasonal_cycle(ds):
    # 1) compute the monthly climatology *preserving* each var’s attrs
    clim = ds.groupby("time.month").mean("time", keep_attrs=True)

    # 2) subtract, then restore your dataset‐level attrs
    anom = ds.groupby("time.month") - clim
    anom.attrs = ds.attrs
    return anom