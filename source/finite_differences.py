import numpy as np
import xarray as xr

def diff_1d_flux_llc90(w_flux, geometry):
    w_flux = w_flux.transpose('time','tile','k_l','j','i')
    w_flux_padded = w_flux.pad(pad_width={'k_l': (0, 1)}, 
                               mode='constant', 
                               constant_values=0) #all fluxes are zero at the bottom
    
    dw = w_flux_padded.diff("k_l")
    dw = dw.rename({'k_l': 'k'}).assign_coords(k=geometry['k'])
    return dw
    
def calc_1d_flux_convergence(w_flux, geometry):
    return diff_1d_flux_llc90(w_flux, geometry)
    
def calc_2d_flux_convergence(u_flux, v_flux, geometry):
    dudv = diff_2d_flux_llc90(u_flux, v_flux, geometry)

    flux_conv = -(dudv["X"] + dudv["Y"])
    
    if ("time" in flux_conv.coords):
        return flux_conv.transpose('time','tile','k','j','i')
    else:
        return flux_conv.transpose('tile','k','j','i')

def diff_2d_flux_llc90(u_flux, v_flux, geometry):
    """
    Manually compute the 2D divergence of fluxes on the LLC90 grid.
    This function mimics xgcm.diff_2d_vector but explicitly handles
    face connections and padding for the LLC topology.

    Parameters
    ----------
    flux_vector : dict
        A dictionary with keys 'X' and 'Y' containing xarray.DataArray
        objects for the U-flux (on i-face) and V-flux (on j-face),
        respectively.

    Returns
    -------
    dict
        A dictionary with keys 'X' and 'Y' for the differenced U and V
        flux components on C-grid points. You can combine these to get
        the divergence: dU/dx + dV/dy.
    """

    # ------------------------------------------------------------------------
    # 1) Pad arrays by one cell in the face dimension to prepare for diff
    # ------------------------------------------------------------------------

    # Pad u_flux along the i_g dimension (add a column of NaNs at the end)
    u_flux_padded = u_flux.pad(
        pad_width={'i_g': (0, 1)},
        mode='constant', constant_values=np.nan
    )
    # Ensure u_padded has a single chunk along i_g for unambiguous padding
    u_flux_padded = u_flux_padded.chunk({'i_g': u_flux_padded.sizes['i_g'] + 1})

    # Pad v_flux along the j_g dimension (add a row of NaNs at the end)
    v_flux_padded = v_flux.pad(
        pad_width={'j_g': (0, 1)},
        mode='constant', constant_values=np.nan
    )
    # Ensure v_padded has a single chunk along j_g
    v_flux_padded = v_flux_padded.chunk({'j_g': v_flux_padded.sizes['j_g'] + 1})

    # ------------------------------------------------------------------------
    # 2) Helper function to replace padded boundary values
    # ------------------------------------------------------------------------
    def da_replace_at_indices(da,indexing_dict,replace_values):
        # replace values in xarray DataArray using locations specified by indexing_dict
        array_data = da.data
        indexing_dict_bynum = {}
        for axis,dim in enumerate(da.dims):
            if dim in indexing_dict.keys():
                indexing_dict_bynum = {**indexing_dict_bynum,**{axis:indexing_dict[dim]}}
        ndims = len(array_data.shape)
        indexing_list = [':']*ndims
        for axis in indexing_dict_bynum.keys():
            indexing_list[axis] = indexing_dict_bynum[axis]
        indexing_str = ",".join(indexing_list)

        # using exec isn't ideal, but this works for both NumPy and Dask arrays
        exec('array_data['+indexing_str+'] = replace_values')

        return da

    # u flux padding
    for tile in range(0,3):
        u_flux_padded = da_replace_at_indices(u_flux_padded,{'tile':str(tile),'i_g':'-1'},\
                                              u_flux.isel(tile=tile+3,i_g=0).data)
    for tile in range(3,6):
        u_flux_padded = da_replace_at_indices(u_flux_padded,{'tile':str(tile),'i_g':'-1'},\
                                              v_flux.isel(tile=12-tile,j_g=0,i=slice(None,None,-1)).data)
    u_flux_padded = da_replace_at_indices(u_flux_padded,{'tile':'6','i_g':'-1'},\
                                          u_flux.isel(tile=7,i_g=0).data)
    for tile in range(7,9):
        u_flux_padded = da_replace_at_indices(u_flux_padded,{'tile':str(tile),'i_g':'-1'},\
                                              u_flux.isel(tile=tile+1,i_g=0).data)
    for tile in range(10,12):
        u_flux_padded = da_replace_at_indices(u_flux_padded,{'tile':str(tile),'i_g':'-1'},\
                                              u_flux.isel(tile=tile+1,i_g=0).data)

    # v flux padding
    for tile in range(0,2):
        v_flux_padded = da_replace_at_indices(v_flux_padded,{'tile':str(tile),'j_g':'-1'},\
                                              v_flux.isel(tile=tile+1,j_g=0).data)
    v_flux_padded = da_replace_at_indices(v_flux_padded,{'tile':'2','j_g':'-1'},\
                                          u_flux.isel(tile=6,j=slice(None,None,-1),i_g=0).data)
    for tile in range(3,6):
        v_flux_padded = da_replace_at_indices(v_flux_padded,{'tile':str(tile),'j_g':'-1'},\
                                              v_flux.isel(tile=tile+1,j_g=0).data)
    v_flux_padded = da_replace_at_indices(v_flux_padded,{'tile':'6','j_g':'-1'},\
                                          u_flux.isel(tile=10,j=slice(None,None,-1),i_g=0).data)
    for tile in range(7,10):
        v_flux_padded = da_replace_at_indices(v_flux_padded,{'tile':str(tile),'j_g':'-1'},\
                                              v_flux.isel(tile=tile+3,j_g=0).data)
    for tile in range(10,13):
        v_flux_padded = da_replace_at_indices(v_flux_padded,{'tile':str(tile),'j_g':'-1'},\
                                              u_flux.isel(tile=12-tile,j=slice(None,None,-1),i_g=0).data)

    # ------------------------------------------------------------------------
    # 5) Compute the finite differences on padded arrays
    # ------------------------------------------------------------------------

    # Difference along i_g then rename to i (C-point)
    du = u_flux_padded.diff('i_g')
    du = du.rename({'i_g': 'i'}).assign_coords(i=geometry['i'])

    # Difference along j_g then rename to j
    dv = v_flux_padded.diff('j_g')
    dv = dv.rename({'j_g': 'j'}).assign_coords(j=geometry['j'])

    # Return a dict matching xgcm's diff_2d_vector output
    return {'X': du, 'Y': dv}
