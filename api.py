import numpy as np
np.float = float
np.complex = complex
import os
import json
import uuid
import threading
import queue
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch

from lib_v5.vr_network.model_param_init import ModelParameters
from separate import SeperateVR, SeperateMDX, SeperateMDXC, SeperateDemucs, clear_gpu_cache

app = FastAPI(title="UVR API")

MODELS_DIR = "/app/models"
VR_PARAM_DIR = "/app/lib_v5/vr_network/modelparams"

jobs = {}
job_queue = queue.Queue()

class JobRequest(BaseModel):
    audio_file: str
    export_path: str
    model_name: str
    architecture: str # "VR Arc", "MDX-Net", "Demucs"
    parameters: Optional[Dict[str, Any]] = None

class APIModelData:
    def __init__(self, **kwargs):
        # Default all required attributes to typical/safe defaults
        self.is_pitch_change = False
        self.semitone_shift = 0
        self.is_match_frequency_pitch = False
        self.overlap = 0.25
        self.overlap_mdx = 0.25
        self.overlap_mdx23 = 8
        self.is_mdx_combine_stems = False
        self.is_mdx_c = False
        self.mdx_c_configs = None
        self.mdxnet_stem_select = 'Vocals'
        self.mixer_path = '/app/lib_v5/mixer.ckpt'
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_pre_proc_model = False
        self.is_secondary_model_activated = False
        self.is_secondary_model = False
        self.process_method = 'VR Arc'
        self.model_path = ''
        self.model_name = ''
        self.model_basename = ''
        self.wav_type_set = 'PCM_16'
        self.mp3_bit_set = '320k'
        self.save_format = 'WAV'
        self.is_gpu_conversion = 0 if torch.cuda.is_available() else -1
        self.is_normalization = False
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.is_ensemble_mode = False
        self.secondary_model = None
        self.primary_model_primary_stem = None
        self.primary_stem_native = 'Vocals'
        self.primary_stem = 'Vocals'
        self.secondary_stem = 'Instrumental'
        self.is_invert_spec = False
        self.is_deverb_vocals = False
        self.is_mixer_mode = False
        self.secondary_model_scale = 1.0
        self.is_demucs_pre_proc_model_inst_mix = False
        self.ensemble_primary_stem = 'Vocals'
        self.ensemble_secondary_stem = 'Instrumental'
        self.is_multi_stem_ensemble = False
        self.DENOISER_MODEL = None
        self.DEVERBER_MODEL = None
        self.vocal_split_model = None
        self.is_vocal_split_model = False
        self.is_save_inst_vocal_splitter = False
        self.is_inst_only_voc_splitter = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = False
        self.is_sec_bv_rebalance = False
        self.deverb_vocal_opt = None
        self.is_save_vocal_only = False
        self.device_set = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_use_opencl = False
        self.is_mdx_ckpt = False
        self.is_denoise = False
        self.is_denoise_model = False
        self.is_mdx_c_seg_def = False
        self.mdx_batch_size = 1
        self.compensate = 1.035
        self.mdx_segment_size = 256
        self.mdx_dim_f_set = 2048
        self.mdx_dim_t_set = 8
        self.mdx_n_fft_scale_set = 6144
        self.chunks = 0
        self.margin = 44100
        self.demucs_stems = 'Vocals'
        self.secondary_model_4_stem = None
        self.secondary_model_4_stem_scale = 1.0
        self.is_chunk_demucs = False
        self.segment = 'Default'
        self.demucs_version = 'v3'
        self.demucs_source_list = []
        self.demucs_source_map = {}
        self.is_demucs_combine_stems = False
        self.demucs_stem_count = 2
        self.pre_proc_model = None
        self.shifts = 2
        self.is_split_mode = False
        self.vr_model_param = None
        self.is_high_end_process = 'None'
        self.is_tta = False
        self.is_post_process = False
        self.batch_size = 1
        self.window_size = 512
        self.post_process_threshold = 0.2
        self.aggression_setting = 5
        
        # Override with any provided kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

def worker_thread():
    while True:
        job_id, req = job_queue.get()
        if job_id is None:
            break
        process_audio(job_id, req)
        job_queue.task_done()

threading.Thread(target=worker_thread, daemon=True).start()

@app.get("/models")
def list_models():
    """Returns a list of models available in the system grouped by architecture."""
    models = {
        "VR Arc": [f for f in os.listdir(os.path.join(MODELS_DIR, "VR_Models")) if not os.path.isdir(os.path.join(MODELS_DIR, "VR_Models", f))] if os.path.exists(os.path.join(MODELS_DIR, "VR_Models")) else [],
        "MDX-Net": [f for f in os.listdir(os.path.join(MODELS_DIR, "MDX_Net_Models")) if not os.path.isdir(os.path.join(MODELS_DIR, "MDX_Net_Models", f))] if os.path.exists(os.path.join(MODELS_DIR, "MDX_Net_Models")) else [],
        "Demucs": [f for f in os.listdir(os.path.join(MODELS_DIR, "Demucs_Models")) if not os.path.isdir(os.path.join(MODELS_DIR, "Demucs_Models", f))] if os.path.exists(os.path.join(MODELS_DIR, "Demucs_Models")) else [],
    }
    return models

def process_audio(job_id: str, req: JobRequest):
    try:
        jobs[job_id]["status"] = "processing"
        
        params = req.parameters or {}
        
        model_path = ""
        if req.architecture == "VR Arc":
            model_path = os.path.join(MODELS_DIR, "VR_Models", req.model_name)
        elif req.architecture == "MDX-Net":
            model_path = os.path.join(MODELS_DIR, "MDX_Net_Models", req.model_name)
        elif req.architecture == "Demucs":
            model_path = os.path.join(MODELS_DIR, "Demucs_Models", req.model_name)
            
        kwargs = {
            "process_method": req.architecture,
            "model_path": model_path,
            "model_name": req.model_name,
            "model_basename": os.path.splitext(req.model_name)[0],
        }
        
        if req.architecture == "VR Arc":
            param_name = params.get("vr_model_param", "1band_sr44100_hl512")
            param_path = os.path.join(VR_PARAM_DIR, f"{param_name}.json")
            if not os.path.exists(param_path):
                raise ValueError(f"VR model parameter file not found: {param_path}")
                
            kwargs["vr_model_param"] = ModelParameters(param_path)
            kwargs["primary_stem"] = params.get("primary_stem", "Instrumental")
            kwargs["secondary_stem"] = "Vocals" if kwargs["primary_stem"] == "Instrumental" else "Instrumental"
            kwargs["model_samplerate"] = kwargs["vr_model_param"].param['sr']
            
        model_data = APIModelData(**kwargs)
        
        for k, v in params.items():
            if k not in ["vr_model_param", "primary_stem"]:
                setattr(model_data, k, v)
        
        def set_progress_bar(step, step_status=None):
            jobs[job_id]["progress"] = step
            if step_status:
                jobs[job_id]["step_status"] = step_status
                
        def write_to_console(*args, **kwargs):
            jobs[job_id]["logs"].append(args)
            
        process_data = {
            'model_data': model_data, 
            'export_path': req.export_path,
            'audio_file_base': os.path.splitext(os.path.basename(req.audio_file))[0],
            'audio_file': req.audio_file,
            'set_progress_bar': set_progress_bar,
            'write_to_console': write_to_console,
            'process_iteration': lambda: None,
            'cached_source_callback': lambda *args, **kwargs: (None, None),
            'cached_model_source_holder': lambda *args, **kwargs: None,
            'list_all_models': [req.model_name],
            'is_ensemble_master': False,
            'is_4_stem_ensemble': False
        }
        
        if req.architecture == "VR Arc":
            seperator = SeperateVR(model_data, process_data)
        elif req.architecture == "MDX-Net":
            is_mdx_c = params.get("is_mdx_c", False)
            if is_mdx_c:
                seperator = SeperateMDXC(model_data, process_data)
            else:
                seperator = SeperateMDX(model_data, process_data)
        elif req.architecture == "Demucs":
            seperator = SeperateDemucs(model_data, process_data)
        else:
            raise ValueError(f"Unknown architecture: {req.architecture}")
            
        seperator.seperate()
        clear_gpu_cache()
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        import traceback
        traceback.print_exc()

@app.post("/jobs")
def create_job(req: JobRequest):
    """Submits a separation job to the internal background queue."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "logs": [],
        "error": None
    }
    job_queue.put((job_id, req))
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Retrieves the status of a specific job by ID."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
