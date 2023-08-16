#
# Based on open source framework Sionna of 2021-2023 NVIDIA CORPORATION & AFFILIATES
# Based on implementation of scenario RMa in Sionna TR38.901
# Copyright Department of Communications Engineering, University of Bremen
#
"""3GPP TR38.811 dense urban channel scenario"""

import tensorflow as tf
import numpy as np

from sionna import SPEED_OF_LIGHT, PI
from sionna.utils import log10
from . import SystemLevelScenario

class DenseUrbanScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.811 dense urban channel model scenario.

    Parameters
    -----------

    carrier_frequency : float
        Carrier frequency [Hz]

    rx_array : PanelArray
        Panel array used by the receivers. All receivers share the same
        antenna array configuration.

    tx_array : PanelArray
        Panel array used by the transmitters. All transmitters share the
        same antenna array configuration.

    direction : str
        Link direction. Either "uplink" or "downlink".

    elevation_angle : float
        elevation angle of the LOS path of the satellite/HAPS vs. ground horizon in degrees

    enable_pathloss : bool
        If `True`, apply pathloss. Otherwise doesn't. Defaults to `True`.

    enable_shadow_fading : bool
        If `True`, apply shadow fading. Otherwise doesn't.
        Defaults to `True`.

    average_street_width : float
        Average street width [m]. Defaults to 5m.

    average_street_width : float
        Average building height [m]. Defaults to 20m.

    always_generate_lsp : bool
        If `True`, new large scale parameters (LSPs) are generated for every
        new generation of channel impulse responses. Otherwise, always reuse
        the same LSPs, except if the topology is changed. Defaults to
        `False`.

    dtype : Complex tf.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `tf.complex64`.
    """

    def __init__(self, carrier_frequency, ut_array, bs_array,
        direction, elevation_angle, enable_pathloss=True, enable_shadow_fading=True,
        average_street_width=20.0, average_building_height=5.0,
        dtype=tf.complex64):

        super().__init__(carrier_frequency, ut_array, bs_array,
            direction, elevation_angle, enable_pathloss, enable_shadow_fading, dtype)

        # Average street width [m]
        self._average_street_width = tf.constant(average_street_width,
            self._dtype.real_dtype)

        # Average building height [m]
        self._average_building_height = tf.constant(average_building_height,
            self._dtype.real_dtype)

    #########################################
    # Public methods and properties
    #########################################

    def clip_carrier_frequency_lsp(self, fc):
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation

        Input
        -----
        fc : float
            Carrier frequency [GHz]

        Output
        -------
        : float
            Clipped carrier frequency, that should be used for LSp computation
        """
        return fc

    @property
    def min_2d_in(self):
        r"""Minimum indoor 2D distance for indoor UTs [m]"""
        return tf.constant(0.0, self._dtype.real_dtype)

    @property
    def max_2d_in(self):
        r"""Maximum indoor 2D distance for indoor UTs [m]"""
        return tf.constant(10.0, self._dtype.real_dtype)

    @property
    def average_street_width(self):
        r"""Average street width [m]"""
        return self._average_street_width

    @property
    def average_building_height(self):
        r"""Average building height [m]"""
        return self._average_building_height

    @property
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to generate LoS
        status of outdoor UTs.

        Taken from table 6.6.1-1 in TR38.811

        [batch size, num_ut]"""
        #can' access get_param function with los before get_sample function. Manual circumvention
        #TO DO, this is currently a  Hotfix only taking the shape of previous calculation here, getting the shape in a prettier fashion in the future
        angle_str = str(round(self._elevation_angle/10.0)*10)
        los_p = self._params_los["LoS_p" + '_' + angle_str]

        #print("shape of loc ins dus: ", self._ut_loc.shape)

        distance_2d_out = self._distance_2d_out
        #los_probability = tf.exp(-(distance_2d_out-10.0)/1000.0)
        los_probability = tf.where(tf.math.less(distance_2d_out, 10.0),
            tf.constant(los_p, self._dtype.real_dtype), tf.constant(los_p, self._dtype.real_dtype))

        #los_probability = tf.zeros(shape = los_probability.shape) + los_p
        return los_probability

    @property
    def rays_per_cluster(self):
        r"""Number of rays per cluster"""
        return tf.constant(20, tf.int32)

    @property
    def los_parameter_filepath(self):
        r""" Path of the configuration file for LoS scenario"""
        assert (self.carrier_frequency >= 2e9 and self.carrier_frequency <= 4e9 or self.carrier_frequency >= 26e9 and self.carrier_frequency <= 40e9) 
        if self.direction == "uplink":
            if self.carrier_frequency >= 2e9 and self.carrier_frequency <= 4e9:
                return 'Dense_Urban_LOS_S_band_UL.json'
            else:
                return 'Dense_Urban_LOS_Ka_band_UL.json'
        else:
            if self.carrier_frequency >= 2e9 and self.carrier_frequency <= 4e9:
                return 'Dense_Urban_LOS_S_band.json'
            else:
                return 'Dense_Urban_LOS_Ka_band.json'
            

    @property
    def nlos_parameter_filepath(self):
        r""" Path of the configuration file for NLoS scenario"""
        assert (self.carrier_frequency >= 2e9 and self.carrier_frequency <= 4e9 or self.carrier_frequency >= 26e9 and self.carrier_frequency <= 40e9) 
        if self.direction == "uplink":
            if self.carrier_frequency >= 2e9 and self.carrier_frequency <= 4e9:
                return 'Dense_Urban_NLOS_S_band_UL.json'
            else:
                return 'Dense_Urban_NLOS_Ka_band_UL.json'
        else:
            if self.carrier_frequency >= 2e9 and self.carrier_frequency <= 4e9:
                return 'Dense_Urban_NLOS_S_band.json'
            else:
                return 'Dense_Urban_NLOS_Ka_band.json'

    #########################
    # Utility methods
    #########################
    #needs to be adapted, TO DO
    def _compute_lsp_log_mean_std(self):
        r"""Computes the mean and standard deviations of LSPs in log-domain"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        h_bs = self.h_bs
        h_bs = tf.expand_dims(h_bs, axis=2) # For broadcasting
        h_ut = self.h_ut
        h_ut = tf.expand_dims(h_ut, axis=1) # For broadcasting

        ## Mean
        # DS
        log_mean_ds = self.get_param("muDS")
        # ASD
        log_mean_asd = self.get_param("muASD")
        # ASA
        log_mean_asa = self.get_param("muASA")
        # SF.  Has zero-mean.
        log_mean_sf = tf.zeros([batch_size, num_bs, num_ut],
                                self._dtype.real_dtype)
        # K.  Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK")/10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD mean is of the form max(-1, A*d2D/1000 - 0.01*(hUT-1.5) + B)
        log_mean_zsd = (self.get_param("muZSD"))
        log_mean_zsd = tf.math.maximum(tf.constant(-1.0,
                                        self._dtype.real_dtype), log_mean_zsd)

        lsp_log_mean = tf.stack([log_mean_ds,
                                log_mean_asd,
                                log_mean_asa,
                                log_mean_sf,
                                log_mean_k,
                                log_mean_zsa,
                                log_mean_zsd], axis=3)

        ## STD
        # DS
        log_std_ds = self.get_param("sigmaDS")
        # ASD
        log_std_asd = self.get_param("sigmaASD")
        # ASA
        log_std_asa = self.get_param("sigmaASA")
        # SF. Given in dB in the 3GPP tables, hence the division by 10
        # O2I and NLoS cases just require the use of a predefined value
        log_std_sf_o2i_nlos = self.get_param("sigmaSF")/10.0
        # For LoS, two possible scenarion depending on the 2D location of the
        # user
        distance_breakpoint = (2.*PI*h_bs*h_ut*self.carrier_frequency/
            SPEED_OF_LIGHT)
        log_std_sf_los=tf.where(tf.math.less(distance_2d, distance_breakpoint),
            self.get_param("sigmaSF")/10.0, self.get_param("sigmaSF")/10.0)
        # Use the correct SF STD according to the user scenario: NLoS/O2I, or
        # LoS
        log_std_sf = tf.where(self.los, log_std_sf_los, log_std_sf_o2i_nlos)
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK")/10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        log_std_zsd = self.get_param("sigmaZSD")

        lsp_log_std = tf.stack([log_std_ds,
                               log_std_asd,
                               log_std_asa,
                               log_std_sf,
                               log_std_k,
                               log_std_zsa,
                               log_std_zsd], axis=3)

        self._lsp_log_mean = lsp_log_mean
        self._lsp_log_std = lsp_log_std

        # ZOD offset
        zod_offset = (tf.atan((35.-3.5)/distance_2d)
          - tf.atan((35.-1.5)/distance_2d))
        zod_offset = tf.where(self.los, tf.constant(0.0,self._dtype.real_dtype),
            zod_offset)
        self._zod_offset = zod_offset

    def _compute_pathloss_basic(self):

        r"""Computes the basic component of the pathloss [dB] according to TR38.811, section 6.6.2"""
        
        distance_2d = self.distance_2d
        distance_3d = self.distance_3d
        #print("distance 3d shape in dus: ", distance_3d.shape)
        fc = self.carrier_frequency/1e9 # Carrier frequency (GHz)
        angle_str = str(round(self._elevation_angle/10.0)*10)

        

        fspl = 32.45 + 20*log10(fc) + 20*log10(distance_3d)
        #print("this is fspl: ", fspl, " with shape ", fspl.shape)
        cl = self._params_nlos["CL" + '_' + angle_str]

        sigmaSF_los = self._params_los["sigmaSF" + '_' + angle_str]
        sigmaSF_nlos = self._params_nlos["sigmaSF" + '_' + angle_str]


        #TO DO this is a Hotfix and needs to be made prettier soon
        #as tf.random.normal cannot handle the None value of the first dimension of pl_b, this will be replaced by a manual override
        #SF_los = tf.random.normal(shape=(distance_2d.shape[1],distance_2d.shape[2]), mean = 0.0, stddev = sigmaSF_los)
        #SF_nlos = tf.random.normal(shape=(distance_2d.shape[1],distance_2d.shape[2]), mean = 0.0, stddev = sigmaSF_nlos)

        #SF_los = tf.reshape(SF_los, shape = (None, SF_los.shape[0], SF_los.shape[1]))
        #SF_nlos = tf.reshape(SF_nlos, shape = (None, SF_los.shape[0], SF_los.shape[1]))

        #SF_los = tf.map_fn(fn=lambda t : np.random.normal(loc = 0.0, scale = sigmaSF_los), elems=fspl)
        #SF_nlos = tf.map_fn(fn=lambda t : np.random.normal(loc = 0.0, scale = sigmaSF_nlos), elems=fspl)
        #tf.map_fn(fn=lambda t: myOp(t), elems=elems)
        #SF_los = [random.normal(mean = 0.0, loc = sigmaSF_los) for d in distance_2d]
        #SF_nlos = [random.normal(mean = 0.0, loc = sigmaSF_nlos) for d in distance_2d]
        #TO DO Hotfix, seems to only output the same value for the random function evytime, replace in the future
        SF_los = tf.where(self.los, np.random.normal(loc = 0.0, scale = sigmaSF_los), np.random.normal(loc = 0.0, scale = sigmaSF_los))
        SF_nlos = tf.where(self.los, np.random.normal(loc = 0.0, scale = sigmaSF_nlos), np.random.normal(loc = 0.0, scale = sigmaSF_nlos))

        pl_los = fspl + SF_los
        pl_nlos = fspl + SF_nlos + cl
        
        pl_b = tf.where(self.los, pl_los, pl_nlos)
        #print("from dus 811:")
        ##print("pl_d: ", pl_b)
        #print("SF_los: ", SF_los)
        #print("SF_nlos: ", SF_nlos)
        self._pl_b = pl_b