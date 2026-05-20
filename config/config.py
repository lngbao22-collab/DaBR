# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import logging
import re
import numpy as np
from tqdm import tqdm

from metrics import ClassificationMetricsCalculator, Metrics

# CUDA optimizations
use_gpu = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    use_gpu = True
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for better performance
    torch.backends.cudnn.deterministic = True  # For reproducibility
    torch.backends.cudnn.enabled = True  # Enable cuDNN
    
    logger = logging.getLogger(__name__)
    logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")

class MyDataParallel(nn.DataParallel):
    def _getattr__(self, name):
        return getattr(self.module, name)
  
            
def to_var(x):
    return Variable(torch.from_numpy(x).to(device))


class Config(object):
    def __init__(self):
        base_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../release/Base.so")
        )
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        """argtypes"""
        """'sample"""
        self.lib.sampling.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        """'valid"""
        self.lib.getValidHeadBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getValidTailBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.validHead.argtypes = [ctypes.c_void_p]
        self.lib.validTail.argtypes = [ctypes.c_void_p]
        """test link prediction"""
        self.lib.getHeadBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getTailBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.testHead.argtypes = [ctypes.c_void_p]
        self.lib.testTail.argtypes = [ctypes.c_void_p]
        """test triple classification"""
        self.lib.getValidBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getTestBatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.getBestThreshold.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.test_triple_classification.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        """restype"""
        self.lib.getValidHit10.restype = ctypes.c_float

        # for triple classification
        self.lib.test_triple_classification.restype = ctypes.c_float
        """set essential parameters"""

        self.in_path = "./"
        self.batch_size = 100
        self.bern = 0
        self.work_threads = 8
        self.hidden_size = 100
        self.negative_ent = 1
        self.negative_rel = 0
        self.ent_size = self.hidden_size
        self.rel_size = self.hidden_size
        self.margin = 1.0
        self.valid_steps = 5
        self.save_steps = 5
        self.opt_method = "SGD"
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.lmbda = 0.0
        self.lmbda_two = 0.0
        self.alpah = 0.001
        self.early_stopping_patience = 10
        self.nbatches = 100
        self.p_norm = 1
        self.test_link = True
        self.test_triple = False
        self.model = None
        self.trainModel = None
        self.testModel = None
        self.pretrain_model = None
        self.ent_dropout = 0
        self.rel_dropout = 0
        self.use_init_embeddings = False
        self.test_file_path = None
        self.triple_classification_path = ""
        self.valid_class_total = 0
        self.test_class_total = 0
        self.use_labeled_triple_classification = False
        
        # Timing and reporting attributes
        self.train_time = 0.0
        self.valid_time = 0.0
        self.test_time = 0.0
        self.total_time = 0.0
        self.best_epoch = 0
        self.best_mrr = 0.0
        self.last_test_metrics = {}

    def _native_function(self, name):
        try:
            return getattr(self.lib, name)
        except AttributeError:
            return None

    def _call_native_if_available(self, name, *args):
        func = self._native_function(name)
        if func is None:
            return None
        return func(*args)

    def init(self):
        self.lib.setInPath(
            ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2)
        )

        self.lib.setTestFilePath(
            ctypes.create_string_buffer(self.test_file_path.encode(), len(self.test_file_path) * 2)
        )

        if self.triple_classification_path:
            triple_path_func = self._native_function("setTripleClassificationPath")
            if triple_path_func is not None:
                triple_path_func(
                    ctypes.create_string_buffer(
                        self.triple_classification_path.encode(),
                        len(self.triple_classification_path) * 2,
                    )
                )
            else:
                print(
                    "Warning: native library does not expose setTripleClassificationPath; "
                    "triple classification will be disabled for this run."
                )
                self.triple_classification_path = ""

        self.lib.setBern(self.bern)
        self.lib.setWorkThreads(self.work_threads)
        self.lib.randReset()
        self.lib.importTrainFiles()
        self.lib.importTestFiles()
        self.lib.importTypeFiles()
        self.relTotal = self.lib.getRelationTotal()
        self.entTotal = self.lib.getEntityTotal()
        self.trainTotal = self.lib.getTrainTotal()
        self.testTotal = self.lib.getTestTotal()
        self.validTotal = self.lib.getValidTotal()
        self.valid_class_total = self._call_native_if_available(
            "getValidClassificationTotal"
        )
        self.test_class_total = self._call_native_if_available(
            "getTestClassificationTotal"
        )
        if self.valid_class_total is None:
            self.valid_class_total = 0
        if self.test_class_total is None:
            self.test_class_total = 0
        self.use_labeled_triple_classification = (
            bool(self.triple_classification_path)
            and self.valid_class_total > 0
            and self.test_class_total > 0
        )

        self.batch_size = int(self.trainTotal / self.nbatches)
        self.batch_seq_size = self.batch_size * (
            1 + self.negative_ent + self.negative_rel
        )
        self.batch_h = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_t = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_r = np.zeros(self.batch_seq_size, dtype=np.int64)
        self.batch_y = np.zeros(self.batch_seq_size, dtype=np.float32)
        self.batch_h_addr = self.batch_h.__array_interface__["data"][0]
        self.batch_t_addr = self.batch_t.__array_interface__["data"][0]
        self.batch_r_addr = self.batch_r.__array_interface__["data"][0]
        self.batch_y_addr = self.batch_y.__array_interface__["data"][0]

        self.valid_h = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_t = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_r = np.zeros(self.entTotal, dtype=np.int64)
        self.valid_h_addr = self.valid_h.__array_interface__["data"][0]
        self.valid_t_addr = self.valid_t.__array_interface__["data"][0]
        self.valid_r_addr = self.valid_r.__array_interface__["data"][0]

        self.test_h = np.zeros(self.entTotal, dtype=np.int64)
        self.test_t = np.zeros(self.entTotal, dtype=np.int64)
        self.test_r = np.zeros(self.entTotal, dtype=np.int64)
        self.test_h_addr = self.test_h.__array_interface__["data"][0]
        self.test_t_addr = self.test_t.__array_interface__["data"][0]
        self.test_r_addr = self.test_r.__array_interface__["data"][0]

        self.valid_pos_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_pos_h_addr = self.valid_pos_h.__array_interface__["data"][0]
        self.valid_pos_t_addr = self.valid_pos_t.__array_interface__["data"][0]
        self.valid_pos_r_addr = self.valid_pos_r.__array_interface__["data"][0]
        self.valid_neg_h = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_t = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_r = np.zeros(self.validTotal, dtype=np.int64)
        self.valid_neg_h_addr = self.valid_neg_h.__array_interface__["data"][0]
        self.valid_neg_t_addr = self.valid_neg_t.__array_interface__["data"][0]
        self.valid_neg_r_addr = self.valid_neg_r.__array_interface__["data"][0]

        self.test_pos_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_pos_h_addr = self.test_pos_h.__array_interface__["data"][0]
        self.test_pos_t_addr = self.test_pos_t.__array_interface__["data"][0]
        self.test_pos_r_addr = self.test_pos_r.__array_interface__["data"][0]
        self.test_neg_h = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_t = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_r = np.zeros(self.testTotal, dtype=np.int64)
        self.test_neg_h_addr = self.test_neg_h.__array_interface__["data"][0]
        self.test_neg_t_addr = self.test_neg_t.__array_interface__["data"][0]
        self.test_neg_r_addr = self.test_neg_r.__array_interface__["data"][0]

        self.valid_cls_pos_h = np.zeros(self.valid_class_total, dtype=np.int64)
        self.valid_cls_pos_t = np.zeros(self.valid_class_total, dtype=np.int64)
        self.valid_cls_pos_r = np.zeros(self.valid_class_total, dtype=np.int64)
        self.valid_cls_pos_h_addr = self.valid_cls_pos_h.__array_interface__["data"][0]
        self.valid_cls_pos_t_addr = self.valid_cls_pos_t.__array_interface__["data"][0]
        self.valid_cls_pos_r_addr = self.valid_cls_pos_r.__array_interface__["data"][0]
        self.valid_cls_neg_h = np.zeros(self.valid_class_total, dtype=np.int64)
        self.valid_cls_neg_t = np.zeros(self.valid_class_total, dtype=np.int64)
        self.valid_cls_neg_r = np.zeros(self.valid_class_total, dtype=np.int64)
        self.valid_cls_neg_h_addr = self.valid_cls_neg_h.__array_interface__["data"][0]
        self.valid_cls_neg_t_addr = self.valid_cls_neg_t.__array_interface__["data"][0]
        self.valid_cls_neg_r_addr = self.valid_cls_neg_r.__array_interface__["data"][0]

        self.test_cls_pos_h = np.zeros(self.test_class_total, dtype=np.int64)
        self.test_cls_pos_t = np.zeros(self.test_class_total, dtype=np.int64)
        self.test_cls_pos_r = np.zeros(self.test_class_total, dtype=np.int64)
        self.test_cls_pos_h_addr = self.test_cls_pos_h.__array_interface__["data"][0]
        self.test_cls_pos_t_addr = self.test_cls_pos_t.__array_interface__["data"][0]
        self.test_cls_pos_r_addr = self.test_cls_pos_r.__array_interface__["data"][0]
        self.test_cls_neg_h = np.zeros(self.test_class_total, dtype=np.int64)
        self.test_cls_neg_t = np.zeros(self.test_class_total, dtype=np.int64)
        self.test_cls_neg_r = np.zeros(self.test_class_total, dtype=np.int64)
        self.test_cls_neg_h_addr = self.test_cls_neg_h.__array_interface__["data"][0]
        self.test_cls_neg_t_addr = self.test_cls_neg_t.__array_interface__["data"][0]
        self.test_cls_neg_r_addr = self.test_cls_neg_r.__array_interface__["data"][0]
        self.relThresh = np.zeros(self.relTotal, dtype=np.float32)
        self.relThresh_addr = self.relThresh.__array_interface__["data"][0]

    def set_test_link(self, test_link):
        self.test_link = test_link

    def set_test_triple(self, test_triple):
        self.test_triple = test_triple

    def set_margin(self, margin):
        self.margin = margin

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_test_file_path(self, test_file_path):
        self.test_file_path = test_file_path

    def set_triple_classification_path(self, triple_classification_path):
        self.triple_classification_path = triple_classification_path

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_p_norm(self, p_norm):
        self.p_norm = p_norm

    def set_valid_steps(self, valid_steps):
        self.valid_steps = valid_steps

    def set_save_steps(self, save_steps):
        self.save_steps = save_steps

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_result_dir(self, result_dir):
        self.result_dir = result_dir

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda
        
    def set_lmbda_two(self, lmbda_two):
        self.lmbda_two = lmbda_two

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_bern(self, bern):
        self.bern = bern

    def set_init_embeddings(self, entity_embs, rel_embs):
        self.use_init_embeddings = True
        self.init_ent_embs = torch.from_numpy(entity_embs).to(device)
        self.init_rel_embs = torch.from_numpy(rel_embs).to(device)

    def set_dimension(self, dim):
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_ent_dimension(self, dim):
        self.ent_size = dim

    def set_rel_dimension(self, dim):
        self.rel_size = dim

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_ent_dropout(self, ent_dropout):
        self.ent_dropout = ent_dropout

    def set_rel_dropout(self, rel_dropout):
        self.rel_dropout = rel_dropout
        
    def set_early_stopping_patience(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def get_parameters(self, param_dict, mode="numpy"):
        for param in param_dict:
            param_dict[param] = param_dict[param].cpu()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = param_dict[param].numpy()
            elif mode == "list":
                res[param] = param_dict[param].numpy().tolist()
            else:
                res[param] = param_dict[param]
        return res

    def _capture_native_output(self, func, *args):
        stdout_fd = sys.stdout.fileno()
        saved_stdout_fd = os.dup(stdout_fd)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, stdout_fd)
        os.close(write_fd)

        try:
            result = func(*args)
            try:
                ctypes.CDLL(None).fflush(None)
            except Exception:
                pass
        finally:
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)

        captured = []
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            captured.append(chunk)
        os.close(read_fd)

        return result, b"".join(captured).decode(errors="replace")

    def _parse_link_prediction_output(self, native_output):
        section_pattern = re.compile(r"^(no type constraint results|type constraint results):$", re.MULTILINE)
        row_pattern = re.compile(
            r"^(l|r|averaged)\((raw|filter)\):\s+"
            r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)$"
        )

        parsed = {}
        current_section = None
        current_metric_names = None

        for raw_line in native_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            section_match = section_pattern.match(line)
            if section_match:
                current_section = section_match.group(1).replace(" ", "_")
                parsed[current_section] = {}
                current_metric_names = None
                continue
            if line.startswith("metric:"):
                current_metric_names = ["MRR", "MR", "Hit@10", "Hit@3", "Hit@1"]
                continue
            row_match = row_pattern.match(line)
            if row_match and current_section is not None and current_metric_names is not None:
                row_name = f"{row_match.group(1)}_{row_match.group(2)}"
                parsed[current_section][row_name] = {
                    metric_name: float(value)
                    for metric_name, value in zip(current_metric_names, row_match.groups()[2:])
                }

        summary = {}
        for section_name, section_metrics in parsed.items():
            averaged_rows = {}
            for row_name in ("averaged_raw", "averaged_filter"):
                if row_name in section_metrics:
                    averaged_rows[row_name] = section_metrics[row_name]
            if averaged_rows:
                summary[section_name] = averaged_rows

        return {
            "sections": parsed,
            "summary": summary,
        }

    def save_embedding_matrix(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + ".json")
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters(best_model, "list")))
        f.close()

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config=self)
        #self.trainModel = nn.DataParallel(self.trainModel, device_ids=[2,3,4])
        
        self.trainModel.to(device)
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.trainModel.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing")

    def set_test_model(self, model, path=None):
        print("Initializing test model...")
        self.model = model
        self.testModel = self.model(config=self)
        if path == None:
            path = os.path.join(self.result_dir, self.model.__name__ + ".ckpt")
        self.testModel.load_state_dict(torch.load(path, weights_only=True))
        self.testModel.to(device)
        self.testModel.eval()
        print("Finish initializing")

    def sampling(self):
        self.lib.sampling(
            self.batch_h_addr,
            self.batch_t_addr,
            self.batch_r_addr,
            self.batch_y_addr,
            self.batch_size,
            self.negative_ent,
            self.negative_rel,
        )

    def save_checkpoint(self, model, epoch):
        path = os.path.join(
            self.checkpoint_dir, self.model.__name__ + "-" + str(epoch) + ".ckpt"
        )
        torch.save(model, path)

    def save_best_checkpoint(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + ".ckpt")
        torch.save(best_model, path)

    def train_one_step(self):
        self.trainModel.train()
        self.trainModel.batch_h = to_var(self.batch_h)
        self.trainModel.batch_t = to_var(self.batch_t)
        self.trainModel.batch_r = to_var(self.batch_r)
        self.trainModel.batch_y = to_var(self.batch_y)
        
        self.optimizer.zero_grad()
        loss = self.trainModel()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainModel.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()

    def test_one_step(self, model, test_h, test_t, test_r):
        model.eval()
        with torch.no_grad():
            model.batch_h = to_var(test_h)
            model.batch_t = to_var(test_t)
            model.batch_r = to_var(test_r)
        return model.predict()

    def valid(self, model):
        self.lib.validInit()
        for i in range(self.validTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            self.lib.getValidHeadBatch(
                self.valid_h_addr, self.valid_t_addr, self.valid_r_addr
            )
            res = self.test_one_step(model, self.valid_h, self.valid_t, self.valid_r)

            self.lib.validHead(res.__array_interface__["data"][0])

            self.lib.getValidTailBatch(
                self.valid_h_addr, self.valid_t_addr, self.valid_r_addr
            )
            res = self.test_one_step(model, self.valid_h, self.valid_t, self.valid_r)
            self.lib.validTail(res.__array_interface__["data"][0])
        return self.lib.getValidHit10()


    def training_model(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        
        best_epoch = 0
        best_hit10 = 0.0
        best_mrr = 0.0
        best_model = None
        bad_counts = 0
        
        # Timing trackers
        total_start_time = time.time()
        train_time_total = 0.0
        valid_time_total = 0.0
        
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            # Training phase
            train_start = time.time()
            res = 0.0
            for batch in range(self.nbatches):
                self.sampling()
                loss = self.train_one_step()
                res += loss
            train_time_total += time.time() - train_start
            
            training_range.set_description("Epoch %d | loss: %.6f" % (epoch, res))
            
            if (epoch + 1) % self.save_steps == 0:
                training_range.set_description("Epoch %d has finished, saving..." % (epoch))
                self.save_checkpoint(self.trainModel.state_dict(), epoch)
            
            # Validation phase
            if (epoch + 1) % self.valid_steps == 0:
                training_range.set_description("Epoch %d validating..." % (epoch))
                valid_start = time.time()
                hit10 = self.valid(self.trainModel)
                valid_time_total += time.time() - valid_start
                
                if hit10 > best_hit10:
                    best_hit10 = hit10
                    best_mrr = hit10  # Store as placeholder, actual MRR from C++ lib
                    best_epoch = epoch
                    best_model = self.trainModel.state_dict()
                    bad_counts = 0
                    logger = logging.getLogger(__name__)
                    logger.info(f"Epoch {epoch}: Best model | Hit@10: {best_hit10:.6f}")
                else:
                    bad_counts += 1
                    logger = logging.getLogger(__name__)
                    logger.info(f"Epoch {epoch}: Hit@10: {hit10:.6f} | Bad count: {bad_counts}")
                
                if bad_counts == self.early_stopping_patience:
                    logger = logging.getLogger(__name__)
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        if best_model is None:
            best_model = self.trainModel.state_dict()
            best_epoch = self.train_times - 1
            valid_start = time.time()
            best_hit10 = self.valid(self.trainModel)
            valid_time_total += time.time() - valid_start
        
        self.train_time = train_time_total
        self.valid_time = valid_time_total
        self.best_epoch = best_epoch
        self.best_mrr = best_hit10
        
        logger = logging.getLogger(__name__)
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Best Hit@10: {best_hit10:.6f}")
        logger.info(f"Total training time: {train_time_total:.2f}s")
        logger.info(f"Total validation time: {valid_time_total:.2f}s")
        
        logger.info("Storing checkpoint of best result...")
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model)
        logger.info("Checkpoint stored")
        
        # Testing phase
        logger.info("Starting testing...")
        test_start = time.time()
        self.set_test_model(self.model)
        self.test()
        self.test_time = time.time() - test_start
        logger.info(f"Test time: {self.test_time:.2f}s")
        
        self.total_time = time.time() - total_start_time
        
        logger.info("Training completed successfully")
        return best_model

    def valid_triple_classification(self, model):
        use_labeled = self.use_labeled_triple_classification
        self.lib.getValidBatch(
            self.valid_cls_pos_h_addr if use_labeled else self.valid_pos_h_addr,
            self.valid_cls_pos_t_addr if use_labeled else self.valid_pos_t_addr,
            self.valid_cls_pos_r_addr if use_labeled else self.valid_pos_r_addr,
            self.valid_cls_neg_h_addr if use_labeled else self.valid_neg_h_addr,
            self.valid_cls_neg_t_addr if use_labeled else self.valid_neg_t_addr,
            self.valid_cls_neg_r_addr if use_labeled else self.valid_neg_r_addr,
        )
        res_pos = self.test_one_step(
            model,
            self.valid_cls_pos_h if use_labeled else self.valid_pos_h,
            self.valid_cls_pos_t if use_labeled else self.valid_pos_t,
            self.valid_cls_pos_r if use_labeled else self.valid_pos_r,
        )
        res_neg = self.test_one_step(
            model,
            self.valid_cls_neg_h if use_labeled else self.valid_neg_h,
            self.valid_cls_neg_t if use_labeled else self.valid_neg_t,
            self.valid_cls_neg_r if use_labeled else self.valid_neg_r,
        )
        self.lib.getBestThreshold(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

        return self.lib.test_triple_classification(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

    def training_triple_classification(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        
        best_epoch = 0
        best_acc = 0.0
        best_model = None
        bad_counts = 0
        
        # Timing trackers
        total_start_time = time.time()
        train_time_total = 0.0
        valid_time_total = 0.0
        
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            # Training phase
            train_start = time.time()
            res = 0.0
            for batch in range(self.nbatches):
                self.sampling()
                loss = self.train_one_step()
                res += loss
            train_time_total += time.time() - train_start
            
            training_range.set_description("Epoch %d | loss: %.6f" % (epoch, res))
            
            if (epoch + 1) % self.save_steps == 0:
                training_range.set_description("Epoch %d has finished, saving..." % (epoch))
                self.save_checkpoint(self.trainModel.state_dict(), epoch)
            
            # Validation phase
            if (epoch + 1) % self.valid_steps == 0:
                training_range.set_description("Epoch %d validating..." % (epoch))
                valid_start = time.time()
                acc = self.valid_triple_classification(self.trainModel)
                valid_time_total += time.time() - valid_start
                
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    best_model = self.trainModel.state_dict()
                    bad_counts = 0
                    logger = logging.getLogger(__name__)
                    logger.info(f"Epoch {epoch}: Best model | Accuracy: {best_acc:.6f}")
                else:
                    bad_counts += 1
                    logger = logging.getLogger(__name__)
                    logger.info(f"Epoch {epoch}: Accuracy: {acc:.6f} | Bad count: {bad_counts}")
                
                if bad_counts == self.early_stopping_patience:
                    logger = logging.getLogger(__name__)
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        if best_model is None:
            best_model = self.trainModel.state_dict()
            best_epoch = self.train_times - 1
            valid_start = time.time()
            best_acc = self.valid_triple_classification(self.trainModel)
            valid_time_total += time.time() - valid_start
        
        self.train_time = train_time_total
        self.valid_time = valid_time_total
        self.best_epoch = best_epoch
        self.best_mrr = best_acc
        
        logger = logging.getLogger(__name__)
        logger.info(f"Best epoch: {best_epoch}")
        logger.info(f"Best Accuracy: {best_acc:.6f}")
        logger.info(f"Total training time: {train_time_total:.2f}s")
        logger.info(f"Total validation time: {valid_time_total:.2f}s")
        
        logger.info("Storing checkpoint of best result...")
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model)
        logger.info("Checkpoint stored")
        
        # Testing phase
        logger.info("Starting testing...")
        test_start = time.time()
        self.set_test_model(self.model)
        self.test()
        self.test_time = time.time() - test_start
        logger.info(f"Test time: {self.test_time:.2f}s")
        
        self.total_time = time.time() - total_start_time
        
        logger.info("Training completed successfully")
        return best_model

    def link_prediction(self):
        print("The total of test triple is %d" % (self.testTotal))
        for i in range(self.testTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testHead(res.__array_interface__["data"][0])

            self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
            res = self.test_one_step(
                self.testModel, self.test_h, self.test_t, self.test_r
            )
            self.lib.testTail(res.__array_interface__["data"][0])
        _, native_output = self._capture_native_output(self.lib.test_link_prediction)

        logger = logging.getLogger(__name__)
        logger.info("\n%s", native_output.rstrip())
        return {
            "native_output": native_output,
            "metrics": self._parse_link_prediction_output(native_output),
        }

    def triple_classification(self):
        use_labeled = self.use_labeled_triple_classification
        self.lib.getValidBatch(
            self.valid_cls_pos_h_addr if use_labeled else self.valid_pos_h_addr,
            self.valid_cls_pos_t_addr if use_labeled else self.valid_pos_t_addr,
            self.valid_cls_pos_r_addr if use_labeled else self.valid_pos_r_addr,
            self.valid_cls_neg_h_addr if use_labeled else self.valid_neg_h_addr,
            self.valid_cls_neg_t_addr if use_labeled else self.valid_neg_t_addr,
            self.valid_cls_neg_r_addr if use_labeled else self.valid_neg_r_addr,
        )
        res_pos = self.test_one_step(
            self.testModel,
            self.valid_cls_pos_h if use_labeled else self.valid_pos_h,
            self.valid_cls_pos_t if use_labeled else self.valid_pos_t,
            self.valid_cls_pos_r if use_labeled else self.valid_pos_r,
        )
        res_neg = self.test_one_step(
            self.testModel,
            self.valid_cls_neg_h if use_labeled else self.valid_neg_h,
            self.valid_cls_neg_t if use_labeled else self.valid_neg_t,
            self.valid_cls_neg_r if use_labeled else self.valid_neg_r,
        )
        self.lib.getBestThreshold(
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

        self.lib.getTestBatch(
            self.test_cls_pos_h_addr if use_labeled else self.test_pos_h_addr,
            self.test_cls_pos_t_addr if use_labeled else self.test_pos_t_addr,
            self.test_cls_pos_r_addr if use_labeled else self.test_pos_r_addr,
            self.test_cls_neg_h_addr if use_labeled else self.test_neg_h_addr,
            self.test_cls_neg_t_addr if use_labeled else self.test_neg_t_addr,
            self.test_cls_neg_r_addr if use_labeled else self.test_neg_r_addr,
        )
        res_pos = np.asarray(
            self.test_one_step(
                self.testModel,
                self.test_cls_pos_h if use_labeled else self.test_pos_h,
                self.test_cls_pos_t if use_labeled else self.test_pos_t,
                self.test_cls_pos_r if use_labeled else self.test_pos_r,
            )
        ).reshape(-1)
        res_neg = np.asarray(
            self.test_one_step(
                self.testModel,
                self.test_cls_neg_h if use_labeled else self.test_neg_h,
                self.test_cls_neg_t if use_labeled else self.test_neg_t,
                self.test_cls_neg_r if use_labeled else self.test_neg_r,
            )
        ).reshape(-1)

        classifier = ClassificationMetricsCalculator()
        for score, rel in zip(
            res_pos,
            self.test_cls_pos_r if use_labeled else self.test_pos_r,
        ):
            threshold = self.relThresh[int(rel)]
            classifier.add_prediction(1, int(score <= threshold), float(-score))
        for score, rel in zip(
            res_neg,
            self.test_cls_neg_r if use_labeled else self.test_neg_r,
        ):
            threshold = self.relThresh[int(rel)]
            classifier.add_prediction(0, int(score <= threshold), float(-score))

        classification_metrics = classifier.get_metrics()

        accuracy, native_output = self._capture_native_output(
            self.lib.test_triple_classification,
            self.relThresh_addr,
            res_pos.__array_interface__["data"][0],
            res_neg.__array_interface__["data"][0],
        )

        classification_metrics["Accuracy"] = float(accuracy)

        logger = logging.getLogger(__name__)
        logger.info("\n%s", native_output.rstrip())
        logger.info("Triple classification metrics: %s", Metrics.format_metrics(classification_metrics))
        return {
            "accuracy": float(accuracy),
            "metrics": classification_metrics,
            "native_output": native_output,
        }

    def test(self):
        results = {}
        if self.test_link:
            results["link_prediction"] = self.link_prediction()
        if self.test_triple:
            results["triple_classification"] = self.triple_classification()
        self.last_test_metrics = results
        return results
    
    def _save_report(self, test_metrics=None):
        """Save training and testing report to file."""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "best_epoch": self.best_epoch,
            "best_mrr": float(self.best_mrr),
            "training_time_seconds": float(self.train_time),
            "validation_time_seconds": float(self.valid_time),
            "test_time_seconds": float(self.test_time),
            "total_time_seconds": float(self.total_time),
            "training_time_formatted": f"{int(self.train_time // 3600)}h {int((self.train_time % 3600) // 60)}m {self.train_time % 60:.2f}s",
            "validation_time_formatted": f"{int(self.valid_time // 3600)}h {int((self.valid_time % 3600) // 60)}m {self.valid_time % 60:.2f}s",
            "test_time_formatted": f"{int(self.test_time // 3600)}h {int((self.test_time % 3600) // 60)}m {self.test_time % 60:.2f}s",
            "total_time_formatted": f"{int(self.total_time // 3600)}h {int((self.total_time % 3600) // 60)}m {self.total_time % 60:.2f}s"
        }
        if test_metrics:
            report["test_metrics"] = test_metrics
        
        report_path = os.path.join(self.result_dir, "report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Report saved to {report_path}")
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Best Epoch: {self.best_epoch}")
        logger.info(f"Best MRR: {self.best_mrr:.6f}")
        logger.info(f"Training Time: {report['training_time_formatted']}")
        logger.info(f"Validation Time: {report['validation_time_formatted']}")
        logger.info(f"Test Time: {report['test_time_formatted']}")
        logger.info(f"Total Time: {report['total_time_formatted']}")
        if test_metrics:
            logger.info(f"Test Metrics: {test_metrics}")
        logger.info(f"{'='*60}")
