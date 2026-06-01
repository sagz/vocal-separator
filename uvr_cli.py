import argparse
import os
import sys
import json
import hashlib
from lib_v5.inference_config import ModelData
from lib_v5.vr_network.model_param_init import ModelParameters
import separate

def get_hash(model_path):
    try:
        with open(model_path, 'rb') as f:
            f.seek(-10000 * 1024, 2)
            return hashlib.md5(f.read()).hexdigest()
    except:
        with open(model_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="UVR Headless CLI")
    parser.add_argument('--audio_file', required=True, help="Path to input audio file")
    parser.add_argument('--export_path', required=True, help="Path to output directory")
    parser.add_argument('--model_path', required=True, help="Path to model file")
    parser.add_argument('--process_method', required=True, choices=['VR Architecture', 'MDX-Net', 'Demucs'], help="Architecture type")
    
    # Model configuration for non-interactive registration
    parser.add_argument('--model_config', help="Path to JSON config for unknown models")
    
    # Additional CLI args mapping to InferenceConfig
    parser.add_argument('--primary_stem', default="Instrumental", help="Primary stem (e.g., Instrumental, Vocals)")
    parser.add_argument('--is_gpu_conversion', action='store_true', help="Use GPU")
    # ... more params as needed to run a basic separation
    
    args = parser.parse_args()
    
    process_method_map = {
        'VR Architecture': 'VR Arc',
        'MDX-Net': 'MDX-Net',
        'Demucs': 'Demucs'
    }
    process_method = process_method_map[args.process_method]
    
    if not os.path.exists(args.audio_file):
        print(f"Error: audio file {args.audio_file} not found")
        sys.exit(1)
        
    if not os.path.exists(args.export_path):
        os.makedirs(args.export_path, exist_ok=True)
        
    if not os.path.exists(args.model_path):
        print(f"Error: model file {args.model_path} not found")
        sys.exit(1)
        
    model_hash = get_hash(args.model_path)
    
    # Load model config from JSON or defaults
    model_settings = {}
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config, 'r') as f:
            model_settings = json.load(f)
            # If the json contains a map of hashes, extract the right one
            if model_hash in model_settings:
                model_settings = model_settings[model_hash]
    
    # Populate the config object
    kwargs = {
        'process_method': process_method,
        'model_path': args.model_path,
        'model_name': os.path.basename(args.model_path),
        'model_basename': os.path.splitext(os.path.basename(args.model_path))[0],
        'primary_stem': model_settings.get('primary_stem', args.primary_stem),
        'is_gpu_conversion': 0 if args.is_gpu_conversion else -1,
        'is_normalization': False,
        'is_primary_stem_only': False,
        'is_secondary_stem_only': False,
        'is_ensemble_mode': False,
        'is_pitch_change': False,
        'semitone_shift': 0,
        'is_match_frequency_pitch': False,
        'overlap': 0.25,
        'overlap_mdx': 'Default',
        'overlap_mdx23': 8,
        'is_mdx_combine_stems': False,
        'is_mdx_c': False,
        'mdxnet_stem_select': 'All Stems',
        'model_samplerate': model_settings.get('model_samplerate', 44100),
        'model_capacity': model_settings.get('model_capacity', 32),
        'is_vr_51_model': False,
        'is_pre_proc_model': False,
        'wav_type_set': 'PCM_16',
        'mp3_bit_set': '320k',
        'save_format': 'WAV',
        'is_invert_spec': False,
        'is_deverb_vocals': False,
        'is_mixer_mode': False,
        'is_demucs_pre_proc_model_inst_mix': False,
        'device_set': 'cuda' if args.is_gpu_conversion else 'cpu',
        # Set specific settings from model config
        'vr_model_param': ModelParameters(os.path.join('lib_v5', 'vr_network', 'modelparams', f"{model_settings.get('vr_model_param', '4band_v3')}.json")),
        'mdx_dim_t_set': model_settings.get('mdx_dim_t_set', 8),
        'mdx_dim_f_set': model_settings.get('mdx_dim_f_set', 3072),
        'mdx_n_fft_scale_set': model_settings.get('mdx_n_fft_scale_set', 6144),
        'compensate': model_settings.get('compensate', 1.035),
        'demucs_stems': model_settings.get('demucs_stems', 'All Stems'),
        'demucs_source_list': model_settings.get('demucs_source_list', []),
        'demucs_source_map': model_settings.get('demucs_source_map', {}),
        'demucs_stem_count': model_settings.get('demucs_stem_count', 4),
        'is_demucs_combine_stems': False,
        'is_chunk_demucs': False,
        'segment': 'Default',
        'shifts': 2,
        'margin': 44100,
        'batch_size': 1,
        'mdx_batch_size': 1,
        'window_size': 320,
        'aggression_setting': 0.1,
        'is_tta': False,
        'is_post_process': False,
        'is_high_end_process': 'None',
        'post_process_threshold': 0.2,
        'is_multi_stem_ensemble': False,
    }
    
    # Allow model settings to override kwargs entirely
    for k, v in model_settings.items():
        if k in kwargs:
            kwargs[k] = v
    
    model_data = ModelData(**kwargs)
    
    def set_progress_bar(step):
        print(f"Progress: {step*100:.1f}%")
        
    def write_to_console(text):
        print(text)
        
    process_data = {
        'audio_file': args.audio_file,
        'audio_file_base': os.path.splitext(os.path.basename(args.audio_file))[0],
        'export_path': args.export_path,
        'set_progress_bar': set_progress_bar,
        'write_to_console': write_to_console,
        'cached_source_callback': lambda *args, **kwargs: (None, None),
        'cached_model_source_holder': lambda *args, **kwargs: None,
        'is_ensemble_master': False,
        'is_4_stem_ensemble': False,
        'list_all_models': [args.model_path],
        'process_iteration': lambda: None
    }
    
    print(f"Starting separation for {args.audio_file} using {process_method}")
    
    if process_method == 'VR Arc':
        separator = separate.SeperateVR(model_data, process_data)
        separator.seperate()
    elif process_method == 'MDX-Net':
        separator = separate.SeperateMDX(model_data, process_data)
        separator.seperate()
    elif process_method == 'Demucs':
        separator = separate.SeperateDemucs(model_data, process_data)
        separator.seperate()
    else:
        print("Unknown process method")
        sys.exit(1)
        
    print("Separation complete.")

if __name__ == '__main__':
    main()
