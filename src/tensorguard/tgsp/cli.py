
import argparse
import sys
import json
import os
import secrets
import struct
import hashlib
import tarfile
import tempfile
import logging
from typing import List, Dict

from .manifest import PackageManifest
from .tar_deterministic import create_deterministic_tar
from .format import write_tgsp_package_v1, read_tgsp_header
from ..crypto.kem import generate_hybrid_keypair, decap_hybrid
from ..crypto.sig import generate_hybrid_sig_keypair
from ..crypto.payload import PayloadDecryptor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_keygen(args):
    out = args.out
    os.makedirs(out, exist_ok=True)
    
    if args.type == "signing":
        pub, priv = generate_hybrid_sig_keypair()
        with open(os.path.join(out, "signing.priv"), "w") as f:
            json.dump(priv, f)
        with open(os.path.join(out, "signing.pub"), "w") as f:
            json.dump(pub, f)
        print(f"Generated Hybrid-Dilithium Signing Key in {out}")
        
    elif args.type == "encryption":
        pub, priv = generate_hybrid_keypair()
        with open(os.path.join(out, "encryption.priv"), "w") as f:
            json.dump(priv, f)
        with open(os.path.join(out, "encryption.pub"), "w") as f:
            json.dump(pub, f)
        print(f"Generated Hybrid-Kyber Encryption Key in {out}")

def run_build(args):
    # 1. Manifest
    manifest = PackageManifest(
        tgsp_version="1.0",
        package_id=secrets.token_hex(8),
        model_name=args.model_name,
        model_version=args.model_version,
        author_id="cli-user",
        payload_hash="pending"
    )
    
    # 2. Recipients
    recipients_public_keys = []
    
    if args.recipients:
        for r_str in args.recipients:
            # format: [optional_label:]path_to_pub_json
            path = r_str
            if ":" in r_str:
                path = r_str.split(":")[-1] # handle C:\ paths or labels
                # If the whole thing exists, use it (maybe no label)
                if not os.path.exists(path) and os.path.exists(r_str):
                    path = r_str
            
            if os.path.exists(path):
                with open(path, "r") as f:
                    pk = json.load(f)
                recipients_public_keys.append(pk)
            else:
                logger.warning(f"Recipient key not found: {r_str} (resolved to {path})")
         
    # 3. Payload Stream (Tar)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        create_deterministic_tar(args.input_dir, tf.name)
        tf_path = tf.name
        
    # 4. Signing Key
    if args.signing_key:
        with open(args.signing_key, "r") as f:
            sk = json.load(f)
        sk_id = "key_1"
    else:
        raise ValueError("TGSP v1.0 requires signing key (Hybrid PQC)")

    if not args.signing_pub:
        raise ValueError("TGSP v1.0 requires signing public key (Hybrid PQC)")
    with open(args.signing_pub, "r") as f:
        signing_pub = json.load(f)
        
    # 5. Write Container
    with open(tf_path, "rb") as payload_stream:
        evt = write_tgsp_package_v1(
            args.out,
            manifest,
            payload_stream,
            recipients_public_keys,
            sk,
            signing_pub,
            sk_id,
        )
        
    from ..evidence.store import get_store
    get_store().save_event(evt)
    
    os.unlink(tf_path)
    print(f"TGSP v1.0 Built: {args.out}")
    print(json.dumps(evt, indent=2))

def run_inspect(args):
    data = read_tgsp_header(args.file)
    print(json.dumps(data["header"], indent=2))
    print(f"Manifest Version: {data['manifest'].get('model_version')}")
    print(f"PQC Mode: {data['header']['crypto'].get('kem')}")

def run_open(args):
    data = read_tgsp_header(args.file)
    
    if not args.key:
        print("Private key required to open")
        return

    with open(args.key, "r") as f:
        sk = json.load(f)
        
    dek = None
    for rec in data["recipients"]:
        try:
            ss_hybrid = decap_hybrid(sk, rec["encap"])
            
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
            wrapper = rec["wrapper"]
            nonce = bytes.fromhex(wrapper["nonce"])
            ct = bytes.fromhex(wrapper["ct"])
            
            aead = ChaCha20Poly1305(ss_hybrid)
            dek = aead.decrypt(nonce, ct, None)
            break
        except Exception as e:
            continue
            
    if not dek:
        raise ValueError("Failed to decrypt (No matching recipient or invalid key)")
        
    h = data["header"]
    nonce_base = bytes.fromhex(h["crypto"]["nonce_base"])
    m_hash = h["hashes"]["manifest"]
    r_hash = h["hashes"]["recipients"]
    
    decryptor = PayloadDecryptor(dek, nonce_base, m_hash, r_hash)
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_tar = os.path.join(args.out_dir, "payload_decrypted_temp.tar")
    
    with open(args.file, "rb") as f:
        f.seek(data["payload_offset"])
        total_read = 0
        with open(out_tar, "wb") as out_f:
            while total_read < data["payload_len"]:
                chunk = decryptor.decrypt_chunk_from_stream(f)
                if not chunk: break
                out_f.write(chunk)
                total_read += (4 + len(chunk) + 16)
        
    with tarfile.open(out_tar, "r") as tr:
        tr.extractall(args.out_dir)
    os.remove(out_tar)
    print(f"Payload decrypted and extracted to {args.out_dir}")

# --- Compatibility Shims for QA Suite ---

def create_tgsp(args):
    """Shim for tests: maps old Args class to run_build."""
    # Ensure manifest has compat fields if provided
    if not hasattr(args, 'model_name'): args.model_name = "llama-3-8b"
    if not hasattr(args, 'model_version'): args.model_version = "1.0.0"
    if not hasattr(args, 'input_dir'):
        # Extract from payload if needed
        # args.payload filter: ["adapter1:weights:path"]
        if hasattr(args, 'payload') and args.payload:
            p0 = args.payload[0]
            if ":" in p0:
                args.input_dir = os.path.dirname(p0.split(":")[-1])
            else:
                args.input_dir = os.path.dirname(p0)
        else:
            args.input_dir = "."
            
    if hasattr(args, 'recipient'):
        args.recipients = args.recipient
    else:
        args.recipients = []
        
    if hasattr(args, 'producer_signing_key'):
        args.signing_key = args.producer_signing_key
    
    return run_build(args)

def verify_tgsp(args):
    """Shim for tests: maps VerifyArgs to verify_tgsp_container."""
    from .format import verify_tgsp_container
    # In compat mode, we might not have a public key passed, 
    # so it fails if not self-signed.
    return verify_tgsp_container(args.in_file)

def decrypt_tgsp(args):
    """Shim for tests: maps DecryptArgs to run_open."""
    if hasattr(args, 'recipient_private_key'):
        args.key = args.recipient_private_key
    if hasattr(args, 'in_file'):
        args.file = args.in_file
    if hasattr(args, 'outdir'):
        args.out_dir = args.outdir
    return run_open(args)

def main():
    parser = argparse.ArgumentParser()
    subps = parser.add_subparsers(dest="cmd")
    
    kg = subps.add_parser("keygen")
    kg.add_argument("--type", choices=["signing", "encryption"], required=True)
    kg.add_argument("--out", required=True)
    
    bd = subps.add_parser("build")
    bd.add_argument("--input-dir", required=True)
    bd.add_argument("--out", required=True)
    bd.add_argument("--model-name", default="unknown")
    bd.add_argument("--model-version", default="1.0.0")
    bd.add_argument("--recipients", nargs="+", help="Paths to recipient public key JSONs")
    bd.add_argument("--signing-key", required=True, help="Path to signing private key JSON")
    bd.add_argument("--signing-pub", required=True, help="Path to signing public key JSON")
    
    ins = subps.add_parser("inspect")
    ins.add_argument("--file", required=True)
    
    op = subps.add_parser("open")
    op.add_argument("--file", required=True)
    op.add_argument("--key", required=True)
    op.add_argument("--out-dir", required=True)
    
    args = parser.parse_args()
    if args.cmd == "keygen": run_keygen(args)
    elif args.cmd == "build": run_build(args)
    elif args.cmd == "inspect": run_inspect(args)
    elif args.cmd == "open": run_open(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
