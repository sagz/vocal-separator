import argparse
import os
import time

from separate import SeperateMDX, SeperateMDXC, SeperateDemucs, SeperateVR

class MockModelData:
    def __init__(self, **kwargs):
        self.is_pitch_change = False
        self.semitone_shift = 0
        self.is_match_frequency_pitch = False
        self.overlap = 0.25
        self.overlap_mdx = "Default"
        self.overlap_mdx23 = 8
        self.is_mdx_combine_stems = False
        self.is_mdx_c = False
        self.mdx_c_configs = None
        self.mdxnet_stem_select = "Vocals"
        self.mixer_path = "mixer.ckpt"
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_pre_proc_model = False
        self.is_secondary_model_activated = False
        self.is_secondary_model = False
        self.process_method = "MDX-Net"
        self.model_path = ""
        self.model_name = ""
        self.model_basename = ""
        self.wav_type_set = "PCM_16"
        self.mp3_bit_set = "320k"
        self.save_format = "WAV"
        self.is_gpu_conversion = 0
        self.is_normalization = False
        self.is_primary_stem_only = False
        self.is_secondary_stem_only = False
        self.is_ensemble_mode = False
        self.secondary_model = None
        self.primary_model_primary_stem = "Vocals"
        self.primary_stem_native = "Vocals"
        self.primary_stem = "Vocals"
        self.secondary_stem = "Instrumental"
        self.is_invert_spec = False
        self.is_deverb_vocals = False
        self.is_mixer_mode = False
        self.secondary_model_scale = 1.0
        self.is_demucs_pre_proc_model_inst_mix = False
        self.ensemble_primary_stem = "Vocals"
        self.is_multi_stem_ensemble = False
        self.DENOISER_MODEL = ""
        self.DEVERBER_MODEL = ""
        self.vocal_split_model = None
        self.is_vocal_split_model = False
        self.is_save_inst_vocal_splitter = False
        self.is_inst_only_voc_splitter = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.deverb_vocal_opt = "Vocals"
        self.is_save_vocal_only = False
        self.device_set = "0"
        self.is_use_opencl = False
        self.is_mdx_ckpt = False
        self.is_denoise = False
        self.is_denoise_model = False
        self.is_mdx_c_seg_def = False
        self.mdx_batch_size = 1
        self.compensate = 1.035
        self.mdx_segment_size = 256
        self.dim_f = 2048
        self.dim_t = 8
        self.mdx_n_fft_scale_set = 6144
        self.chunks = 0
        self.margin = 44100
        
        # Demucs specific
        self.demucs_stems = "Vocals"
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.is_chunk_demucs = False
        self.segment = "Default"
        self.demucs_version = "v4"
        self.demucs_source_list = []
        self.demucs_source_map = {}
        self.is_demucs_combine_stems = False
        self.demucs_stem_count = 2
        self.pre_proc_model = None
        self.shifts = 2
        self.is_split_mode = True
        
        # VR specific
        self.vr_model_param = None
        self.is_high_end_process = "None"
        self.is_tta = False
        self.is_post_process = False
        self.batch_size = 1
        self.window_size = 320
        self.post_process_threshold = 0.2
        self.aggression_setting = 0.5

        for k, v in kwargs.items():
            setattr(self, k, v)

def dummy_progress(step, max_step=None):
    pass

def dummy_console(text, base_text=''):
    print(base_text + text)

def cached_source_callback(arch_type, model_name=None):
    return None, None

def cached_model_source_holder(arch_type, sources, model_name=None):
    pass

def run_separation(audio_file, export_path, model_path, process_method="MDX-Net", **kwargs):
    model_name = os.path.basename(model_path)
    model_basename = os.path.splitext(model_name)[0]
    
    # We need MDX dim_f/dim_t settings here if we want to run MDX
    # For a real implementation we would load these from hashes, 
    # but for CLI let's pass them via kwargs or detect
    
    model_data = MockModelData(
        model_path=model_path,
        model_name=model_name,
        model_basename=model_basename,
        process_method=process_method,
        **kwargs
    )
    
    audio_file_base = os.path.splitext(os.path.basename(audio_file))[0]
    
    process_data = {
        'audio_file': audio_file,
        'audio_file_base': audio_file_base,
        'export_path': export_path,
        'set_progress_bar': dummy_progress,
        'write_to_console': dummy_console,
        'cached_source_callback': cached_source_callback,
        'cached_model_source_holder': cached_model_source_holder,
        'is_4_stem_ensemble': False,
        'list_all_models': [model_basename],
        'process_iteration': 1,
        'is_ensemble_master': False
    }
    
    if process_method == "MDX-Net":
        if model_data.is_mdx_c:
            separator = SeperateMDXC(model_data, process_data)
        else:
            separator = SeperateMDX(model_data, process_data)
    elif process_method == "Demucs":
        separator = SeperateDemucs(model_data, process_data)
    elif process_method == "VR Architecture":
        separator = SeperateVR(model_data, process_data)
    else:
        raise ValueError(f"Unknown process method {process_method}")
        
    start_time = time.time()
    separator.seperate()
    print(f"Separation completed in {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UVR CLI")
    parser.add_argument('--audio', required=True, help="Input audio file")
    parser.add_argument('--export', required=True, help="Export path")
    parser.add_argument('--model', required=True, help="Model path")
    parser.add_argument('--method', default="MDX-Net", help="Process method (MDX-Net, Demucs, VR Architecture)")
    
    # MDX Specifics
    parser.add_argument('--mdx-dim-f', type=int, default=2048)
    parser.add_argument('--mdx-dim-t', type=int, default=8)
    parser.add_argument('--mdx-n-fft-scale', type=int, default=6144)
    parser.add_argument('--mdx-compensate', type=float, default=1.035)

    args = parser.parse_args()
    
    os.makedirs(args.export, exist_ok=True)
    
    run_separation(
        args.audio, 
        args.export, 
        args.model, 
        process_method=args.method,
        mdx_dim_f_set=args.mdx_dim_f,
        mdx_dim_t_set=args.mdx_dim_t,
        mdx_n_fft_scale_set=args.mdx_n_fft_scale,
        compensate=args.mdx_compensate
    )
