import argparse
import hashlib
import os
import subprocess
import time

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True)
    parser.add_argument('--export', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--method', default="MDX-Net")
    parser.add_argument('--reference-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.export, exist_ok=True)
    
    start = time.time()
    cmd = [
        "python3", "cli.py",
        "--audio", args.audio,
        "--export", args.export,
        "--model", args.model,
        "--method", args.method
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    inference_time = time.time() - start
    print(f"Inference Time: {inference_time:.2f} seconds")

    # Compare with reference dir
    # We expect identical files
    audio_base = os.path.splitext(os.path.basename(args.audio))[0]
    generated_files = [f for f in os.listdir(args.export) if f.startswith(audio_base) and f.endswith(".wav")]
    
    if not generated_files:
        print("No output files generated.")
        exit(1)
        
    all_match = True
    for f in generated_files:
        gen_path = os.path.join(args.export, f)
        ref_path = os.path.join(args.reference_dir, f)
        
        if not os.path.exists(ref_path):
            print(f"Warning: Reference file {f} does not exist.")
            all_match = False
            continue
            
        gen_hash = get_file_hash(gen_path)
        ref_hash = get_file_hash(ref_path)
        
        if gen_hash != ref_hash:
            print(f"FAIL: {f} hash mismatch! Generated: {gen_hash}, Reference: {ref_hash}")
            all_match = False
        else:
            print(f"PASS: {f} hash matches.")
            
    if all_match:
        print("All outputs bit-exact match reference.")
        exit(0)
    else:
        print("Some outputs did not match reference.")
        exit(1)

if __name__ == '__main__':
    main()
