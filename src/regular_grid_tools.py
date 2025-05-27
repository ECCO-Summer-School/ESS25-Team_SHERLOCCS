import xarray as xr

def regular_vertical_average(ds, zmin, zmax):
    """
    Compute the vertical average of a DataArray over a given depth range.

    Parameters
    ----------
    ds : xarray.DataArray
        Input DataArray with a depth coordinate named 'Z'.
    zmin : float
        Shallow bound of the depth range (in meters).
    zmax : float
        Deep bound of the depth range (in meters).

    Returns
    -------
    xarray.DataArray
        The vertically averaged DataArray over z ∈ [zmax, zmin].
    """
    # Create a mask of ones where Z is between zmax (deep) and zmin (shallow), zeros elsewhere
    weights = xr.where((ds.Z >= zmax) & (ds.Z <= zmin), 1, 0)
    # Weighted sum in Z divided by sum of weights yields the mean
    return (ds * weights).sum('Z') / weights.sum('Z')


def generate_averaged_regular_velocity_dataset(ds_3D_Vel, ds_3D_Bol, zmin, zmax):
    """
    Build an xarray.Dataset of vertically averaged velocity components.

    Applies vertical_average() to each of the four 3D fields:
      - Eulerian eastward (EVEL)
      - Eulerian northward (NVEL)
      - Bolus  eastward (EVELSTAR)
      - Bolus  northward (NVELSTAR)

    Parameters
    ----------
    zmin : float
        Shallow bound of the depth range (in meters).
    zmax : float
        Deep bound of the depth range (in meters).

    Returns
    -------
    xarray.Dataset
        Dataset containing the four averaged fields, each with appropriate
        `units`, `long_name`, and `depth_range` attributes.
    """
    # Perform the vertical averaging on each field
    EVEL      = regular_vertical_average(ds_3D_Vel["EVEL"],      zmin, zmax)
    NVEL      = regular_vertical_average(ds_3D_Vel["NVEL"],      zmin, zmax)
    EVELSTAR  = regular_vertical_average(ds_3D_Bol["EVELSTAR"],  zmin, zmax)
    NVELSTAR  = regular_vertical_average(ds_3D_Bol["NVELSTAR"],  zmin, zmax)

    # Assemble into a new Dataset
    ds = xr.Dataset({
        "EVEL":     EVEL,
        "NVEL":     NVEL,
        "EVELSTAR": EVELSTAR,
        "NVELSTAR": NVELSTAR,
    })

    # Define metadata for each variable
    attrs_map = {
        "EVEL": {
            "units":       "m/s",
            "long_name":   "Vertically Averaged Eulerian Eastward Velocity",
            "depth_range": f"{zmin} and {zmax} meters"
        },
        "NVEL": {
            "units":       "m/s",
            "long_name":   "Vertically Averaged Eulerian Northward Velocity",
            "depth_range": f"{zmin} and {zmax} meters"
        },
        "EVELSTAR": {
            "units":       "m/s",
            "long_name":   "Vertically Averaged Bolus Eastward Velocity",
            "depth_range": f"{zmin} and {zmax} meters"
        },
        "NVELSTAR": {
            "units":       "m/s",
            "long_name":   "Vertically Averaged Bolus Northward Velocity",
            "depth_range": f"{zmin} and {zmax} meters"
        },
    }

    # Attach attributes to each variable in the dataset
    for var_name, atts in attrs_map.items():
        ds[var_name] = ds[var_name].assign_attrs(**atts)

    return ds


def select_regular_california(ds):
    """
    Subset a Dataset or DataArray to the California region.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input with 'longitude' and 'latitude' coordinates.

    Returns
    -------
    same type as input
        Spatial subset covering longitude ∈ [-140, -100]°E
        and latitude ∈ [15, 40]°N.
    """
    return ds.sel(longitude=slice(-140, -100), latitude=slice(15, 40))
