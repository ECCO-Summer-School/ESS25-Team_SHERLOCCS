# tell Python to use the ecco_v4_py in the 'ECCOv4-py' repository
from os.path import join,expanduser
import sys

# identify user's home directory
user_home_dir = expanduser('~')

# import the ECCOv4 py library 
sys.path.insert(0,join(user_home_dir,'ECCOv4-py'))

import xarray as xr
import numpy as np
from finite_differences import *
from grid_utils import *
import ecco_v4_py as ecco

def read_geothermal_fluxes(ds_GEOM): 
    geoflx = ecco.read_llc_to_tiles("/efs_ecco/ECCO/V4/r4/input/input_forcing/other", 'geothermalFlux.bin')
    # Convert numpy array to an xarray DataArray with matching dimensions as the monthly mean fields
    geoflx_llc = xr.DataArray(geoflx,coords={'tile': ds_GEOM.tile.values,
                                             'j': ds_GEOM.j.values,
                                             'i': ds_GEOM.i.values},dims=['tile','j','i'])
    return geoflx_llc

def calc_3d_geothermal_flux(ds_GEOM):
    # Seawater density (kg/m^3)
    rhoconst = 1029
    ## needed to convert surface mass fluxes to volume fluxes
    
    # Heat capacity (J/kg/K)
    c_p = 3994
    
    
    # Create 3d bathymetry mask
    mskC = 1 * ds_GEOM.hFacC.copy(deep=True).compute()
    
    mskC_shifted = mskC.shift(k=-1)
    
    mskC_shifted.values[-1,:,:,:] = 0
    mskb = mskC - mskC_shifted

    geoflx_llc = read_geothermal_fluxes(ds_GEOM)
    # Create 3d field of geothermal heat flux
    geoflx3d = geoflx_llc * mskb.transpose('k','tile','j','i')
    GEOFLX = geoflx3d.transpose('k','tile','j','i')
    GEOFLX.attrs = {'standard_name': 'GEOFLX','long_name': 'Geothermal heat flux','units': 'W/m^2'}
    
    # Add geothermal heat flux to forcing field and convert from W/m^2 to degC/s
    G_geothermal_forcing = ((GEOFLX)/(rhoconst*c_p))/(ds_GEOM.hFacC*ds_GEOM.drF)
    
    return G_geothermal_forcing

def calc_3d_surf_heat_flux(ds_HFLUX, ds_GEOM):
    # Seawater density (kg/m^3)
    rhoconst = 1029
    ## needed to convert surface mass fluxes to volume fluxes
    
    # Heat capacity (J/kg/K)
    c_p = 3994
    
    # Constants for surface heat penetration (from Table 2 of Paulson and Simpson, 1977)
    R = 0.62
    zeta1 = 0.6
    zeta2 = 20.0
    
    Z = ds_GEOM.Z.compute()
    RF = np.concatenate([ds_GEOM.Zp1.values[:-1],[np.nan]])
    q1 = R*np.exp(1.0/zeta1*RF[:-1]) + (1.0-R)*np.exp(1.0/zeta2*RF[:-1])
    q2 = R*np.exp(1.0/zeta1*RF[1:]) + (1.0-R)*np.exp(1.0/zeta2*RF[1:])
    # Create xarray data arrays
    q1 = xr.DataArray(q1,coords=[Z.k],dims=['k'])
    q2 = xr.DataArray(q2,coords=[Z.k],dims=['k'])
    
    # Correction for the 200m cutoff
    zCut = np.where(Z < -200)[0][0]
    q1[zCut:] = 0
    q2[zCut-1:] = 0
    
    ## Land masks
    # Make copy of hFacC
    mskC = 1 * ds_GEOM.hFacC.copy(deep=True).compute()
    
    # Change all fractions (ocean) to 1. land = 0
    mskC.values[mskC.values>0] = 1
    
    # Shortwave flux below the surface (W/m^2)
    forcH_subsurf = ((q1*(mskC==1)-q2*(mskC.shift(k=-1)==1))*ds_HFLUX["oceQsw"]).transpose('time','tile','k','j','i')
    
    # Surface heat flux (W/m^2)
    forcH_surf = ((ds_HFLUX["TFLUX"] - (1-(q1[0]-q2[0]))*ds_HFLUX["oceQsw"])\
                  *mskC[0]).transpose('time','tile','j','i').assign_coords(k=0).expand_dims('k')
    
    # Full-depth sea surface forcing (W/m^2)
    forcH = xr.concat([forcH_surf,forcH_subsurf[:,:,1:]], dim='k').transpose('time','tile','k','j','i')

    # Add geothermal heat flux to forcing field and convert from W/m^2 to degC/s
    G_surf_forcing = ((forcH)/(rhoconst*c_p))/(ds_GEOM.hFacC*ds_GEOM.drF)
    
    return G_surf_forcing

def calculate_temperature_tendencies(ds_snap, ds_3D, ds_GEOM):

    sTHETA = ds_snap["THETA"]*(1+ds_snap["ETAN"]/ds_GEOM.Depth)

    sTHETAp1 = sTHETA.isel(time = slice(1, None))
    sTHETA = sTHETA.isel(time = slice(0, -1))
    
    sTHETAp1.coords["time"] = ds_3D.coords["time"]
    sTHETA.coords["time"] = ds_3D.coords["time"]
    sTHETAdt = (sTHETAp1 - sTHETA) / 2.628e+6
    return sTHETAdt

def calc_heat_adv(ds_3D, ds_GEOM):
    cell_volumes = get_cell_volumes(ds_GEOM)
    cell_volumes = cell_volumes.where(cell_volumes > 0)
    
    ADV_XY = calc_2d_flux_convergence(ds_3D["ADVx_TH"], ds_3D["ADVy_TH"], ds_GEOM) 
    ADV_Z = calc_1d_flux_convergence(ds_3D["ADVr_TH"], ds_GEOM)
    return  (ADV_XY + ADV_Z) / cell_volumes

def calc_heat_diff(ds_3D, ds_GEOM):
    cell_volumes = get_cell_volumes(ds_GEOM)
    cell_volumes = cell_volumes.where(cell_volumes > 0)
    
    DIF_XY = calc_2d_flux_convergence(ds_3D["DFxE_TH"], ds_3D["DFyE_TH"], ds_GEOM)
    DIF_Z = calc_1d_flux_convergence(ds_3D["DFrE_TH"] + ds_3D["DFrI_TH"], ds_GEOM)
    return  (DIF_XY + DIF_Z) / cell_volumes

def generate_heat_budget_terms(ds_2D, ds_3D, ds_SNAP, ds_GEOM):
    heat_budget = calc_3d_surf_heat_flux(ds_2D.fillna(0.0), ds_GEOM).rename("G_SURF").to_dataset()
    heat_budget["G_GEO"] = calc_3d_geothermal_flux(ds_GEOM).fillna(0.0)
    
    heat_budget["G_DIF"] = calc_heat_diff(ds_3D.fillna(0.0), ds_GEOM)
    heat_budget["G_ADV"] = calc_heat_adv(ds_3D.fillna(0.0), ds_GEOM)
    
    heat_budget["RHS"] = heat_budget["G_ADV"] + heat_budget["G_DIF"] + heat_budget["G_GEO"] + heat_budget["G_SURF"]
    heat_budget["LHS"] = calculate_temperature_tendencies(ds_SNAP, ds_3D, ds_GEOM)

    return heat_budget

def volume_average_heat_budget_terms(ds, ds_GEOM, mask_3d): 
    ds_budget_terms = ["G_GEO", "G_DIF", "G_ADV", "RHS", "G_SURF"]
    cell_volumes = get_cell_volumes(ds_GEOM)
    weights = enforce_pos_def_mask(mask_3d) * cell_volumes
    weights = weights.where(weights > 0)

    ds_heat_budget = (ds[ds_budget_terms] * cell_volumes * mask_3d).sum(["tile", "i", "j", "k"]) / weights.sum(["tile", "i", "j", "k"])
    ds_heat_budget["LHS"] = (ds["LHS"] * weights).sum(["tile", "i", "j", "k"]) / weights.sum(["tile", "i", "j", "k"])
    
    return ds_heat_budget
