import xarray as xr


# Load the data
non_ag_xr = xr.open_dataarray('data/lumap_2d_all_lucc_5km_non_ag.nc', chunks='auto')
bio_map_area_ha = xr.open_dataarray('data/bio_area_ha.nc', chunks='auto')
bio_map_area_ha = bio_map_area_ha.reindex_like(non_ag_xr, method='nearest')

NON_AG_BIO_IMPACT = {
    '1 Conservation and natural environments':0,
    '2 Production from relatively natural environments':0.1,
    '3 Production from dryland agriculture and plantations':0.1,
    '4 Production from irrigated agriculture and plantations':0.1,
    '5 Intensive uses':1, 
    '6 Water':0, 
    'No data':0}

non_ag_bio_effects_land = xr.DataArray(
    list(NON_AG_BIO_IMPACT.values()), 
    dims=['PRIMARY_V7'],
    coords={'PRIMARY_V7': list(NON_AG_BIO_IMPACT.keys())}
)


non_ag_bio_land = non_ag_xr * non_ag_bio_effects_land * bio_map_area_ha

non_ag_bio_land.sum().values
