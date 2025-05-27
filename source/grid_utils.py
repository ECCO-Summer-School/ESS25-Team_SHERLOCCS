import xarray as xr
import numpy as np

def get_tracer_mask(ds_GEOM): 
    return xr.where(ds_GEOM["hFacC"] > 0, 1, np.nan)

def get_cell_depths(ds_GEOM): 
    return get_tracer_mask(ds_GEOM) * ds_GEOM["Z"]

def get_depth_mask(ds_GEOM, zmin, zmax): 
    cell_depths = get_cell_depths(ds_GEOM)
    return xr.where((cell_depths >= zmax) * (cell_depths <= zmin), 1, 0)

def get_cell_volumes(ds_GEOM):
    return ds_GEOM["hFacC"] * xr.where(ds_GEOM["hFacC"] > 0, 1, 0) * ds_GEOM["rA"] * ds_GEOM["drF"]

def enforce_pos_def_mask(mask_3d):
    mask_3d = np.abs(mask_3d).fillna(0.0) #enforce positive weights 
    mask_3d = xr.where(mask_3d > 0, 1, 0.0)
    return mask_3d
    
def volume_average(ds, ds_GEOM, mask_3d): 
    weights = enforce_pos_def_mask(mask_3d) * get_cell_volumes(ds_GEOM)
    weights = weights.where(weights > 0)
    return (ds * weights).sum(["tile", "i", "j", "k"]) / weights.sum(["tile", "i", "j", "k"])



def lateral_volume_average(ds, ds_GEOM, mask_3d): 
    weights = enforce_pos_def_mask(mask_3d) * get_cell_volumes(ds_GEOM)
    weights = weights.where(weights > 0)
    return (ds * weights).sum(["tile", "i", "j"]) / weights.sum(["tile", "i", "j"])

def vertical_volume_average(ds, ds_GEOM, mask_3d): 
    weights = enforce_pos_def_mask(mask_3d) * get_cell_volumes(ds_GEOM)
    weights = weights.where(weights > 0)
    return (ds * weights).sum(["k"]) / weights.sum(["k"])