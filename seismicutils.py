import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm
from time import sleep
from numba import njit
from datetime import datetime


def return_regions():
    #useful dictionary to identify some regions in the world
    regions = {}
    regions['japan'] = (22,46,123,148)
    regions['italy'] = (36,47,6,19)
    regions['turkey'] = (36,41,26,46)
    regions['us'] = (25,51,-127,-64)
    return regions
regions = return_regions()

@njit(nogil=True)
def haversine_1d(pos_rads, origin_rads):
    '''
    pos_rads: will be N stations, with 2 coordinates
    origin_rads: will be only one point
    To compute a distance between two points / sets of points
    '''
    a = np.sin((origin_rads[0]-pos_rads[:,0])/2)**2 + np.cos(origin_rads[0])*np.cos(pos_rads[:,0])*((np.sin((origin_rads[1]-pos_rads[:,1])/2))**2)
    return 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
#                     fit_gps(out_array, 
#                     np.pi*o/180,
#                     v[i,...],
#                     n_discr_lat, n_discr_lon,
#                     np.pi*min_lat/180,
#                     np.pi*min_lon/180,
#                     cell_size_rads,
#                     8*cell_size_rads,
#                     0)   
@njit(nogil=True)
def fit_gps(out_array, stations_positions_rads, data, n_rows, n_cols, \
            latr_origin, lonr_origin, spacing_rads, sigma_rads, min_w):
    '''
    out_array:   (pixel,pixel,channel)  is modified in-place
        channel:     0,1,2 -> n,e,u zeroth-order
        channel:     (...) -> first order method (4 derivative + 2 velocities)
        channel:     -1 -> alpha  (the masking value) is stored there.
    data: (Nstations,3)  # GPS input data for a sinlge day.
    n_rows, n_cols: shape of out_array
    latr_origin, lonr_origin : origin of the image (point (0,0), or lower left corner of the image in lat,lon)
    spacing_rads: cell size (in radians)
    sigma_rads:   sigma of the gaussian used for interpolation
    min_w:  minimal alpha (confidence level) below which we send the value to 0 (discard it)
    '''
    ## (i,j): sweep over all pixels

    alpha_uncapped = np.zeros((out_array.shape[0],out_array.shape[1]))
    for i in range(n_rows):
        grid_latr = latr_origin + i*spacing_rads
        for j in range(n_cols):
            grid_lonr = lonr_origin + j*spacing_rads
            # grid_latr, grid_lonr: are absolute coordinates of the pixel
            ## d is the array of all distances between stations and the current pixel (in radians)
            d = haversine_1d(stations_positions_rads, np.array([grid_latr,grid_lonr]) )
            w = np.exp(-0.5*(d/sigma_rads)**2) ## size Nstations
            norm = w.sum()
            max_w = w.max()  ## when stations are far, the max will pick up essentially the weight relative to the closest station
            alpha_uncapped[i,j] = norm
            ## when stations are close, the max may be larger than 1, but it's ok, we don't care.
            if(norm > 0  and max_w >= min_w): ## exclude the pixel if no station is close enough (max_w too small)
                w /= norm
                mu_pos = np.dot(w,stations_positions_rads)  ## 0-th order method
                var_pos= np.dot(w,stations_positions_rads**2)- mu_pos**2
                cv_pos = np.dot(w,stations_positions_rads[:,0]*stations_positions_rads[:,1]) - mu_pos[0]*mu_pos[1]
                det_cov_matr = var_pos[0]*var_pos[1] - cv_pos*cv_pos
                mu_vel = np.dot(w,data) #zero-th order method, only averages ## product (N,)@(N,3) -> (3,)
                out_array[i,j,:3] = mu_vel
                out_array[i,j,-1] = max_w
                # out_array[i,j,-1] = np.minmum(1.0, norm) ## TODO: to be considered as a better choice for our trust...
                if(det_cov_matr != 0):
                    ## out_array: will be of shape (Npixel, Npixel, 10) if we want to go frist order.
                    if(out_array.shape[-1] > 4): # decide to go to first order
                        cv_y_north = np.dot(w,stations_positions_rads[:,0]*data[:,0]) - mu_pos[0]*mu_vel[0]
                        cv_y_east  = np.dot(w,stations_positions_rads[:,0]*data[:,1]) - mu_pos[0]*mu_vel[1]
                        cv_x_north = np.dot(w,stations_positions_rads[:,1]*data[:,0]) - mu_pos[1]*mu_vel[0]
                        cv_x_east  = np.dot(w,stations_positions_rads[:,1]*data[:,1]) - mu_pos[1]*mu_vel[1]
                        cov_inv = np.array([[var_pos[1], -cv_pos],[-cv_pos, var_pos[0]]])/det_cov_matr
                        b_north = np.array([cv_y_north, cv_x_north])
                        b_east  = np.array([cv_y_east , cv_x_east])
                        out_array[i,j,3:5] = cov_inv@b_north
                        out_array[i,j,5:7] = cov_inv@b_east
                        out_array[i,j,7] = out_array[i,j,0] + (grid_latr-mu_pos[0])*out_array[i,j,3]+ (grid_lonr-mu_pos[1])*out_array[i,j,4]
                        out_array[i,j,8] = out_array[i,j,1] + (grid_latr-mu_pos[0])*out_array[i,j,5]+ (grid_lonr-mu_pos[1])*out_array[i,j,6]
            else:
                ## out_array will remain nans
                continue
    return alpha_uncapped


def fit_constrained(stations_positions_rads, data,n_rows, n_cols, latr_origin, lonr_origin, spacing_rads,sigma_rads, reg_factor = 3, nu=0.5, index_ratio = 0.6):
    '''
    This interpolation assumes the top crust to be modelled as a 2D thin elastic sheet
    Then one interplate displacements based on that model.
    Reference to the original paper:
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016GL070340
    '''

    cutoff = 2*np.pi*spacing_rads*reg_factor
    N = stations_positions_rads.shape[0]
    bf_matr = np.zeros((2*N, 2*N))
    for i,i_pos in enumerate(stations_positions_rads):
        d_ij = haversine_1d(stations_positions_rads, i_pos) + cutoff
        for j,j_pos in enumerate(stations_positions_rads):
            q_ij = (3-nu)*np.log(d_ij[j]) + (1+nu)*((i_pos[1]-j_pos[1])**2)/d_ij[j]**2
            p_ij = (3-nu)*np.log(d_ij[j]) + (1+nu)*((i_pos[0]-j_pos[0])**2)/d_ij[j]**2
            w_ij =  (-(1+nu))*(i_pos[0]-j_pos[0])*(i_pos[1]-j_pos[1])/d_ij[j]**2
            bf_matr[2*i, 2*j] = q_ij
            bf_matr[2*i, 2*j+1] = w_ij
            bf_matr[2*i+1, 2*j] = w_ij
            bf_matr[2*i+1, 2*j+1] = p_ij
    uu, vv, vvh = np.linalg.svd(bf_matr)
    index_cut = int(index_ratio*bf_matr.shape[0])
    known_vec = np.stack([data[:,0],data[:,1]]).T.flatten()
    #body_forces = np.linalg.inv(bf_matr)@known_vec
    body_forces = ((vvh.conj().T)[:, :index_cut]@np.diag(1/vv[:index_cut])@(uu.conj().T)[:index_cut,:])@known_vec
    v = np.zeros((n_rows, n_cols, 3))
    for i in range(n_rows):
        grid_latr = latr_origin + i*spacing_rads
        for j in range(n_cols):
            grid_lonr = lonr_origin + j*spacing_rads
            grid_pos_rads = np.array([grid_latr,grid_lonr ])
            d = haversine_1d(stations_positions_rads,grid_pos_rads) + cutoff
            closes_station_idx = np.argmin(d)
            v[i,j,2] = np.exp(-0.5*((d[closes_station_idx]-cutoff)/sigma_rads)**2)
            q_s = (3-nu)*np.log(d) + (1+nu)*((grid_lonr-stations_positions_rads[:,1])**2)/d**2
            p_s = (3-nu)*np.log(d) + (1+nu)*((grid_latr-stations_positions_rads[:,0])**2)/d**2
            w_s =  (-(1+nu))*(grid_latr-stations_positions_rads[:,0])*(grid_lonr-stations_positions_rads[:,1])/d**2
            v[i,j,0] = np.sum(q_s*body_forces[::2]+ w_s*body_forces[1::2])
            v[i,j,1] = np.sum(w_s*body_forces[::2]+ p_s*body_forces[1::2])
    return v, body_forces


class SeismicUtils:
    
    @staticmethod
    def format_velset_filename(catalogType, asTimeWindow, minMainMag, nDays):
        filename = ""
        filename += "velocity_"
        filename += "cat="+catalogType
        filename += "_ASlag="+str(asTimeWindow)
        filename += "_MS="+str(minMainMag)
        filename += "_days="+str(nDays)
        filename += '.npy'
        return filename

    @staticmethod
    def format_dataset_filename(catalogType, asTimeWindow, minMainMag, cellSizeKm, sigmaInterp, nDays, minNumStations, soft_labels, regression, use_constrained_fit):
        filename = ""
        filename += "dataset_"
        filename += "cat="+catalogType
        filename += "_ASlag="+str(asTimeWindow)
        filename += "_MS="+str(minMainMag)
        filename += "_c="+str(cellSizeKm)
        filename += "_sigIt="+str(sigmaInterp)
        filename += "_days="+str(nDays)
        filename += "_minNStat="+str(minNumStations)

        if soft_labels:
            filename += "_softLabels="+str(soft_labels) 
        if regression:
            filename += "_regrTask="+str(regression) 
        if use_constrained_fit:
            filename += "_thinElasticSheet="+str(use_constrained_fit) 
        filename += '.npy'
        return filename


    @staticmethod
    def str_to_datetime(s, limit_for_2000):
        str_to_month = {'JAN':1, 'FEB':2, 'MAR':3, 'APR' : 4, 'MAY':5, 'JUN':6, 'JUL' : 7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
        d = int(s[5:])
        m = str_to_month[s[2:5]]
        y = s[:2]
        if(int(y) > limit_for_2000):
            y = int("19" + y)
        else:
            y = int("20" + y)
        return datetime(y, m, d)   ## change this to datetime.datetime ?

    @staticmethod
    @njit(nogil=True)
    def create_soft_labels(y, row_indices, col_indices, spacing, sigma, mode='max'):
        ## allows to soften the hard labels in {0,1} by a small Gaussian smoothing.
        ## we go through Aftershocks and add to the local pixel (i,j) a Gaussian (r=distance between i,j and the AS)
        ## call is:
        ## create_soft_labels(y, aftershocks_row[aftershocks_mask], aftershocks_col[aftershocks_mask], cell_size_rads, 2*cell_size_rads)
        ## y is initially full of zeros
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                for (r,c) in zip(row_indices, col_indices):
                    weight = np.exp(-0.5*((i-r)**2+(j-c)**2)*((spacing/sigma)**2))
                    if(mode=='max'):
                        y[i,j] = max(y[i,j],weight)
                    else:
                        y[i,j] += weight
        if(mode != 'max'):
            y = np.minimum(y,1.0)
    
    @staticmethod
    def usgs_query_radius(start_time, end_time, center_lat, center_lon, radius):
        base_link = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&minmagnitude=4"
        base_link += "&starttime="  + str(start_time.astype('datetime64[D]'))
        base_link += "&endtime="  + str(end_time.astype('datetime64[D]'))
        base_link += "&latitude=" + str(center_lat)
        base_link += "&longitude=" + str(center_lon)
        base_link += "&maxradiuskm=" + str(radius)
        return base_link

    @staticmethod
    def xyz_to_latlon(point, r0=6371):
        longitude_radians = np.arctan2(point[:,1],point[:,0])
        latitude_radians =  np.arctan2(point[:,2],np.sqrt(point[:,0]**2+point[:,1]**2))
        return np.rad2deg(np.vstack([latitude_radians, longitude_radians]).T)


    ngl_full_list  = "http://geodesy.unr.edu/NGLStationPages/llh.out"
    ngl_24hFinal = "http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
    ngl_24hRapid = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsRapid24hr.txt"
    ngl_5mRapid = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsRapid5min.txt"
    ngl_5mUltra = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsUltra5min.txt"
    gps_data_types = ['tenv', 'tenv3']
    gps_ref_types = ['IGS14', 'NA']

    @staticmethod
    def get_ngl_stations(url_list = "http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt", post_process = True):
        df_list = pd.read_csv(url_list, sep=r"\s+", on_bad_lines='skip', parse_dates=['Dtbeg', 'Dtend'])
        if(post_process):
            df_list.rename(columns={'Sta' : 'name', 'Lat(deg)' : 'lat', 'Long(deg)' : 'lon', 'Hgt(deg)' : 'height', 'X(m)' : 'x', 'Y(m)' : 'y', 'Z(m)' : 'z', 'Dtbeg' : 'begin', 'Dtend' : 'end'   }, inplace=True)
            df_list['lon'] =  (df_list['lon'] + 180)%360-180
            return df_list[['name', 'lat','lon', 'x','y','z','begin','end']]
        else:
            return df_list
    @staticmethod
    def get_ngl_gps_data(station_name, ref_type, data_type):
        base_url = "http://geodesy.unr.edu/gps_timeseries/" + data_type
        if(ref_type == "IGS14"):
            data_fr =  pd.read_csv(base_url + "/IGS14/" + station_name + "." + data_type, sep=r"\s+")
        else:
            data_fr =  pd.read_csv(base_url + "/plates/NA/" + station_name + ".NA." + data_type, sep=r"\s+")
        return data_fr
    

    @staticmethod
    def load_isc_catalog(file_name, is_header_commented=False, extract_minimal_features=True):
        default_header = ["date","lat","lon","smajax","sminax","strike","q","depth","unc","q","mw","unc","q","s","mo","fac","mo_auth","mpp","mpr","mrr","mrt","mtp","mtt","str1","dip1","rake1","str2","dip2","rake2","type","eventid"]
        if(not is_header_commented):
            catalog  = pd.read_csv(file_name, comment='#', sep=",", low_memory=False, parse_dates=['date'])
        else:
            catalog  = pd.read_csv(file_name, comment='#', sep=",", low_memory=False)
        catalog.columns = default_header
        if(extract_minimal_features):
            catalog = catalog[['date', 'lat', 'lon', 'mw']]
            catalog.rename(columns={'mw' : 'mag'}, inplace=True)
        return catalog
    
    @staticmethod
    def space_time_selection(catalog, min_mainshock_mag, min_event_mag, max_event_dist, dt_fs, dt_as, tol_overlap_dist,   date_name = "date", lat_name="lat", lon_name="lon", mag_name = "mag", r0=6371, verbose=True):
        catalog.sort_values(by=date_name, axis=0, inplace=True)
        sequences = []
        time = catalog[date_name].values
        space = catalog[[lat_name, lon_name]].values
        mag = catalog[mag_name].values
        mainshocks_indices = np.argwhere(mag >= min_mainshock_mag).flatten()
        distance_matrix = r0*haversine_distances(np.radians(space), np.radians(space[mainshocks_indices]))
        internal_idx = 0
        pbar =  tqdm(total=len(mainshocks_indices))

        for cnt, ev_idx in enumerate(mainshocks_indices):
            
            #if(verbose):
            #    print("Completed Perc.:", (100*(cnt/len(mainshocks_indices))))
            dt_event = time-time[ev_idx]
            seq_mask =  (distance_matrix[:, cnt] <= max_event_dist)
            seq_mask *=  (dt_event >= dt_fs)
            seq_mask *= (dt_event <= dt_as)
            seq_mask *= (mag>= min_event_mag)
            seq_indices = np.argwhere(seq_mask).flatten()

            seq = pd.DataFrame()
            seq[date_name] = time[seq_indices]
            seq["day"] = time[seq_indices].astype('datetime64[D]')
            seq[lat_name] = space[seq_indices,0]
            seq[lon_name] = space[seq_indices,1]
            seq[mag_name] = mag[seq_indices]
            mag_type = np.zeros(len(seq), dtype=int)
            dt_event_seq= dt_event[seq_indices].astype('timedelta64[s]')
            mag_type[dt_event_seq ==  np.timedelta64(0, 's') ] = 1
            mag_type[dt_event_seq > np.timedelta64(0, 's')] = 2
            seq["type"] = mag_type
            seq.sort_values(by=date_name, axis=0, inplace=True)
            found_any_conflict = False
            current_event_m_time = seq[seq.type==1][date_name].values[0]
            current_event_m_pos = seq[seq.type==1][[lat_name,lon_name]].values[0,:][None,:]
            
            for prev_seq in sequences:
                last_event_m_pos = prev_seq[prev_seq.type==1][[lat_name,lon_name]].values[0,:][None,:]
                space_dist = r0*haversine_distances(np.radians(last_event_m_pos), np.radians(current_event_m_pos))[0,0]
                # mainshocks are far apart, ignore
                if(space_dist >= tol_overlap_dist):
                    continue
                # here the events are close in space
                last_event_m_time =  prev_seq[prev_seq.type==1][date_name].values[0]
                time_lag = current_event_m_time - last_event_m_time
                # new event happens in the future
                if(time_lag > 0):
                    if(time_lag > dt_as): # no overlap, nice
                        found_any_conflict = False
                        break
                    else:
                        found_any_conflict = True
                        break
                else:
                    found_any_conflict = True
                    break


                  
            if(verbose):
                pbar.update(1)
                sleep(0.01)    
            if(found_any_conflict):
                continue
            else:
                seq['seq_id'] = np.ones(len(seq), dtype=int)*internal_idx
                internal_idx += 1
                sequences.append(seq)
            
        sequences = pd.concat(sequences)
        sequences.reset_index(inplace=True, drop=True)
        return sequences

    

