import os
import ctypes
import numpy as np
from typing import List, Union, Optional, Dict

class GPvrnn(object):
    __PATH_MAX = 4096
    libext = '.dylib' if os.uname().sysname == 'Darwin' else '.so'
    libpath = os.path.join(os.path.dirname(__file__), "../../build/lib/libpvrnn" + libext) # path to compiled PVRNN library
    lib = None
    obj = None
    _tasks = ("training", "testing", "error_regression", "online_error_regression", "planning")

    def __init__(self, config_file="", task="training", config_type="", epoch=-1, rng_seed=-1): # type: (str, str, str, int, int) -> None
        """Loads and initializes the library
           If config_file is blank, initialization is skipped"""
        self.lib = ctypes.CDLL(self.libpath)
        self.lib.getInstance.restype = ctypes.POINTER(ctypes.c_void_p)
        self.obj = self.lib.getInstance()
        if config_file:
            self.newModel(config_file, task, config_type, epoch, rng_seed)

    @property
    def n_seq(self): # type: () -> int
        return self.lib.n_seq(self.obj)

    @property
    def dims(self): # type: () -> int
        return self.lib.dims(self.obj)

    @property
    def max_timesteps(self): # type: () -> int
        return self.lib.max_timesteps(self.obj)

    @property
    def epoch(self): # type: () -> int
        return self.lib.current_epoch(self.obj)

    @property
    def save_epochs(self): # type: () -> int
        return self.lib.save_epochs(self.obj)

    @property
    def max_epochs(self): # type: () -> int
        return self.lib.max_epochs(self.obj)

    @property
    def softmax_quant(self): # type: () -> int
        return self.lib.softmax_quant(self.obj)

    @property
    def n_layers(self): # type: () -> int
        return self.lib.n_layers(self.obj)

    @property
    def window_length(self): # type: () -> int
        return self.lib.window_length(self.obj)

    @property
    def postdiction_window_length(self): # type: () -> int
        return self.lib.postdiction_window_length(self.obj)

    @property
    def planning_windown_length(self): # type: () -> int
        return self.lib.planning_window_length(self.obj)

    @property
    def meta_prior(self): # type: () -> List[float]
        w_buf = np.zeros(self.n_layers-1, dtype=np.float64)
        w = (ctypes.c_double * w_buf.size)(*w_buf)
        self.lib.getMetaPrior(self.obj, w)
        return np.frombuffer(w, np.float64).tolist()

    @property
    def d_neurons(self): # type: () -> List[int]
        d_units_buf = np.zeros(self.n_layers-1, dtype=np.int32)
        d_units = (ctypes.c_int * (self.n_layers-1))(*d_units_buf)
        self.lib.getDNeurons(self.obj, d_units)
        return np.frombuffer(d_units, np.int32).tolist()

    @property
    def z_units(self): # type: () -> List[int]
        z_units_buf = np.zeros(self.n_layers-1, dtype=np.int32)
        z_units = (ctypes.c_int * z_units_buf.size)(*z_units_buf)
        self.lib.getZUnits(self.obj, z_units)
        return np.frombuffer(z_units, np.int32).tolist()

    @property
    def data_path(self): # type: () -> str
        path = ctypes.create_string_buffer(self.__PATH_MAX * 8)
        self.lib.getDataPath(self.obj, path)
        return path.value.decode("ascii")

    @property
    def output_dir(self): # type: () -> str
        path = ctypes.create_string_buffer(self.__PATH_MAX * 8)
        self.lib.getOutputDir(self.obj, path)
        return path.value.decode("ascii")

    @property
    def training_path(self): # type: () -> str
        path = ctypes.create_string_buffer(self.__PATH_MAX * 8)
        self.lib.getTrainingPath(self.obj, path)
        return path.value.decode("ascii")

    @property
    def base_dir(self): # type: () -> str
        path = ctypes.create_string_buffer(self.__PATH_MAX * 8)
        self.lib.getBaseDir(self.obj, path)
        return path.value.decode("ascii")

    # Overload setter handler with data and er_data
    def __setattr__(self, name, value):
        if name == "data":
            """Load training data from a List[List[List[float]]] with shape (n_seq, max_timesteps, dims)"""
            data_in = np.asarray(value).flatten()
            self.lib.setData(self.obj, (ctypes.c_float * len(data_in))(*data_in))
        elif name == "er_data":
            """Load ER target data from a List[List[float]] with shape (max_timesteps,  dims)"""
            data_in = np.asarray(value).flatten()
            self.lib.setErData(self.obj, (ctypes.c_float * len(data_in))(*data_in))
        super(GPvrnn, self).__setattr__(name, value) # passthrough


    def newModel(self, config_file="", task="training", config_type="", epoch=-1, rng_seed=-1): # type: (str, str, str, int, int) -> None
        """Initializes a new model with the given config file. If config_file is blank, the model is left empty and will need to be manually initialized
           config_file: path to a config file. If blank, the configuration will need to be imported later (legacy method)
           config_type: can be 'glean', 'toml' or 'pvrnn'. If blank, attempt to guess the config type from file extension
           task: can be 'training', 'testing', 'error_regression', 'online_error_regression' or 'planning'
           epoch: the epoch to start the task from. -1 means start from the last saved epoch (for training, this will resume from last saved epoch
        """
        if (task not in self._tasks):
            print("newModel: Unknown task configuration", task)
        self.lib.newModel(self.obj, config_file.encode("ascii"), task.encode("ascii"), config_type.encode("ascii"), rng_seed)
        if config_file:
            if task == "training":
                self.trainInitialize(epoch)
            elif task == "testing":
                self.testInitialize(epoch)
            elif task == "error_regression":
                self.batchErInitialize(epoch)
            elif task == "online_error_regression":
                self.onlineErInitialize(epoch)
            elif task == "planning":
                self.planInitialize(epoch)

    def modelInitialize(self): # type: () -> None
        """First stage initialization of the model
           If model is being initialized manually, call after loading configuration"""
        self.lib.modelInitialize(self.obj)

    def trainInitialize(self, start_epoch=0): # type: (int) -> None
        """Initialize model for training
           Set startEpoch to -1 to resume from last saved model, or set to a specific saved epoch
           Leave startEpoch at 0 to restart training from scratch"""
        self.lib.trainInitialize(self.obj, start_epoch)

    def testInitialize(self, epoch=-1): # type (int) -> None
        """Initialize model for testing
           Set epoch to -1 to resume from last saved model, or set to a specific saved epoch"""
        self.lib.testInitialize(self.obj, epoch) # load last saved epoch

    def batchErInitialize(self, epoch=-1): # type: (int) -> None
        """Initialize model for batch error regression
           Set epoch to -1 to resume from last saved model, or set to a specific saved epoch"""
        self.lib.batchErInitialize(self.obj, epoch)

    def onlineErInitialize(self, epoch=-1): # type: (int) -> None
        """Initialize model for online error regression
           Set epoch to -1 to resume from last saved model, or set to a specific saved epoch"""
        self.lib.onlineErInitialize(self.obj, epoch)

    def planInitialize(self, epoch=-1): # type: (int) -> None
        """Initialize model for online planning
           Set epoch to -1 to resume from last saved model, or set to a specific saved epoch"""
        self.lib.planInitialize(self.obj, epoch)

    def train(self, background_epochs=0, greedy_train=True): # type: (int, bool) -> Optional[Dict[str, float]]
        """Train a set number of epochs and then return loss values. Setting background_epochs too high can cause unresponsiveness
           Calls internal train routine if background_epochs <= 0"""
        if background_epochs > 0:
            rec_loss = ctypes.c_double()
            reg_loss = ctypes.c_double()
            self.lib.trainBackground(self.obj, background_epochs, ctypes.byref(rec_loss), ctypes.byref(reg_loss), greedy_train)
            return {"total_batch_reconstruction_loss": rec_loss.value, "total_batch_regularization_loss": reg_loss.value}
        else:
            self.lib.train(self.obj)
            return None

    def test(self): # () -> None
        """Save output of prior and posterior generation for test routine"""
        self.lib.test(self.obj)

    def postGenAndSave(self):
        """For unit testing"""
        self.lib.postGenAndSave(self.obj)

    def priorGenAndSave(self, post_steps=0):
        """For unit testing"""
        self.lib.priorGenAndSave(self.obj, post_steps)

    def batchErrorRegression(self, sub_dir="", output_suffix=""): # type: (str, str) -> Dict[str, float]
        """Loads configured file and does error regression"""
        rec_loss = ctypes.c_double()
        reg_loss = ctypes.c_double()
        self.lib.batchErrorRegression(self.obj, ctypes.byref(rec_loss), ctypes.byref(reg_loss), sub_dir.encode("ascii"), output_suffix.encode("ascii"))
        return {"total_batch_reconstruction_loss": rec_loss.value, "total_batch_regularization_loss": reg_loss.value}

    def onlineErrorRegression(self, data_in, mask=None, sub_dir="", output_suffix=""): # type: (List[float], List[float], str, str) -> Dict[str, Union[List[float], float]]
        """Takes current timestep input and returns one step prediction after error regression. Maintains window internally
           Pass an optional mask of size dims * window_size to mask off dimensions from error calculation. Mask dims should be softmax dims if output is softmax
           Note: if g_window = 0 in config, this returns zeros until the postdiction window is filled"""
        rec_loss = ctypes.c_double()
        reg_loss = ctypes.c_double()
        data_out_buf = np.zeros(len(data_in), dtype=float)
        data_out = (ctypes.c_float * len(data_in))(*data_out_buf)
        if mask is not None:
            mask = np.asarray(mask).flatten()
            self.lib.maskedOnlineErrorRegression(self.obj, (ctypes.c_float * len(data_in))(*data_in), data_out, (ctypes.c_float * len(mask))(*mask), ctypes.byref(rec_loss), ctypes.byref(reg_loss), sub_dir.encode("ascii"), output_suffix.encode("ascii"))
        else:
            self.lib.onlineErrorRegression(self.obj, (ctypes.c_float * len(data_in))(*data_in), data_out, ctypes.byref(rec_loss), ctypes.byref(reg_loss), sub_dir.encode("ascii"), output_suffix.encode("ascii"))
        return {"generated_out": np.frombuffer(data_out, np.float32).tolist(), "total_batch_reconstruction_loss": rec_loss.value, "total_batch_regularization_loss": reg_loss.value}

    def plan(self, mask, data_in=None, dynamic=False, sub_dir="", output_suffix=""): # type: (List[List[float]], List[float], bool, str, str) -> Dict[str, float] # FIXME: masking should be optionally handled internally
        """Takes a mask of size dims * plan length to mask off reconstruction error during error regression"""
        rec_loss = ctypes.c_double()
        reg_loss = ctypes.c_double()
        mask = np.asarray(mask).flatten()
        if data_in is not None:
            self.lib.dynamicPlan(self.obj, (ctypes.c_float * len(data_in))(*data_in), (ctypes.c_float * len(mask))(*mask), dynamic, ctypes.byref(rec_loss), ctypes.byref(reg_loss), sub_dir.encode("ascii"), output_suffix.encode("ascii"))
        else:
            self.lib.plan(self.obj, (ctypes.c_float * len(mask))(*mask), ctypes.byref(rec_loss), ctypes.byref(reg_loss), sub_dir.encode("ascii"), output_suffix.encode("ascii"))
        return {"total_batch_reconstruction_loss": rec_loss.value, "total_batch_regularization_loss": reg_loss.value}

    def getPlanLoss(self, weighted_kld=True): # type: (bool) -> Dict[str, Union[List[float], List[List[float]]]]
        """Return reconstruction and regularization loss (per layer) per timestep. Optionally weight by metaprior"""
        rec_err_buf = np.zeros(self.window_length, dtype=np.float64)
        rec_err = (ctypes.c_double * rec_err_buf.size)(*rec_err_buf)
        self.lib.getFullErRecErr(self.obj, -1, rec_err)
        rec_loss = (np.frombuffer(rec_err, np.float64) / self.window_length).tolist()
        reg_loss = []
        for l in range(self.n_layers-1):
            kld_buf = np.zeros(self.window_length, dtype=np.float64)
            kld = (ctypes.c_double * kld_buf.size)(*kld_buf)
            self.lib.getFullErRegErr(self.obj, l, -1, kld)
            if weighted_kld:
                reg_loss.append((((np.frombuffer(kld, np.float64)/self.window_length))*self.meta_prior[l]).tolist())
            else:
                reg_loss.append((((np.frombuffer(kld, np.float64)/self.window_length))).tolist())
        return {"batch_reconstruction_loss": rec_loss, "batch_regularization_loss": reg_loss}

    def getPlanOutput(self): # type: () -> List[List[float]]
        """Retrieve posterior generation within the postdiction + planning window"""
        data_out_buf = np.zeros(self.dims * self.window_length, dtype=float)
        data_out = (ctypes.c_float * data_out_buf.size)(*data_out_buf)
        self.lib.getErOutput(self.obj, data_out)
        return np.reshape(np.frombuffer(data_out, np.float32), (self.window_length, self.dims)).tolist()

    def getPosteriorA(self): # type: () -> Dict[str, List[List[float]]]
        """Retrieve Adaptation values within window"""
        Amyu_buf = np.zeros(np.sum(self.z_units) * self.window_length, dtype=float)
        Amyu = (ctypes.c_float * Amyu_buf.size)(*Amyu_buf)
        Asigma_buf = np.zeros(np.sum(self.z_units) * self.window_length, dtype=float)
        Asigma = (ctypes.c_float * Asigma_buf.size)(*Asigma_buf)
        self.lib.getPosteriorA(self.obj, Amyu, Asigma)
        return {"myu": np.reshape(np.frombuffer(Amyu, np.float32), (self.window_length, np.sum(self.z_units))).tolist(),
                "sigma": np.reshape(np.frombuffer(Asigma, np.float32), (self.window_length, np.sum(self.z_units))).tolist()}

    def getPriorMyuSigma(self): # type: () -> Dict[str, List[List[float]]]
        """Retrieve myu and sigma values for prior"""
        myu_buf = np.zeros(np.sum(self.z_units) * self.window_length, dtype=float)
        myu = (ctypes.c_float * myu_buf.size)(*myu_buf)
        sigma_buf = np.zeros(np.sum(self.z_units) * self.window_length, dtype=float)
        sigma = (ctypes.c_float * sigma_buf.size)(*sigma_buf)
        self.lib.getPriorMyuSigma(self.obj, myu, sigma)
        return {"myu": np.reshape(np.frombuffer(myu, np.float32), (self.window_length, np.sum(self.z_units))).tolist(),
                "sigma": np.reshape(np.frombuffer(sigma, np.float32), (self.window_length, np.sum(self.z_units))).tolist()}

    def getPosteriorMyuSigma(self): # type: () -> Dict[str, List[List[float]]]
        """Retrieve myu and sigma values for posterior"""
        myu_buf = np.zeros(np.sum(self.z_units) * self.window_length, dtype=float)
        myu = (ctypes.c_float * myu_buf.size)(*myu_buf)
        sigma_buf = np.zeros(np.sum(self.z_units) * self.window_length, dtype=float)
        sigma = (ctypes.c_float * sigma_buf.size)(*sigma_buf)
        self.lib.getPosteriorMyuSigma(self.obj, myu, sigma)
        return {"myu": np.reshape(np.frombuffer(myu, np.float32), (self.window_length, np.sum(self.z_units))).tolist(),
                "sigma": np.reshape(np.frombuffer(sigma, np.float32), (self.window_length, np.sum(self.z_units))).tolist()}

    def priorGeneration(self): # type: () -> List[float]
        """One step prior generation"""
        data_out_buf = np.zeros(self.dims, dtype=float)
        data_out = (ctypes.c_float*self.dims)(*data_out_buf)
        self.lib.priorGeneration(self.obj, data_out)
        return np.frombuffer(data_out, np.float32).tolist()

    def importGConfig(self, cfg_path, task="training"): # type: (str, str) -> None
        """Manually import GLean config"""
        self.lib.importGConfig(self.obj, cfg_path.encode("ascii"), task.encode("ascii"))

    def importTOMLConfig(self, cfg_path, task="training"): # type: (str, str) -> None
        """Manually import TOML config"""
        self.lib.importTOMLConfig(self.obj, cfg_path.encode("ascii"), task.encode("ascii"))

    def reconfigureTOML(self, cfg_str, task="training"): # type: (str, str) -> None
        """Manually import TOML config"""
        self.lib.reconfigureTOML(self.obj, cfg_str.encode("ascii"), task.encode("ascii"))
